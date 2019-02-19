"""The module for training ENAS."""
import contextlib
import glob
import math
import os
import parse

import numpy as np
import random
import scipy.signal
from tensorboard import TensorBoard
import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable

import json
import models
import utils

from models.shared_rnn import RNN
from models.shared_cnn import CNN
from regularized_controllers.supervised import SupervisedController

from train_scripts.regular_trainer import Trainer, _get_optimizer, discount, _get_no_grad_ctx_mgr

logger = utils.get_logger()

class SupervisedTrainer(Trainer):
    """A class to wrap training code."""

    def __init__(self, args, dataset):
        """Constructor for training algorithm.

        Args:
            args: From command line, picked up by `argparse`.
            dataset: Currently only `data.text.Corpus` is supported.

        Initializes:
            - Data: train, val and test.
            - Model: shared and controller.
            - Inference: optimizers for shared and controller parameters.
            - Criticism: cross-entropy loss for training the shared model.
        """
        self.args = args
        self.controller_step = 0
        self.cuda = args.cuda
        self.dataset = dataset
        self.epoch = 0
        self.shared_step = 0
        self.start_epoch = 0
        self.best_ppl = float("inf")

        logger.info('regularizing:')
        for regularizer in [('activation regularization',
                             self.args.activation_regularization),
                            ('temporal activation regularization',
                             self.args.temporal_activation_regularization),
                            ('norm stabilizer regularization',
                             self.args.norm_stabilizer_regularization)]:
            if regularizer[1]:
                logger.info(f'{regularizer[0]}')

        if args.network_type == "rnn":
            self.train_data = utils.batchify(dataset.train,
                                             args.batch_size,
                                             self.cuda)
            # NOTE(brendan): The validation set data is batchified twice
            # separately: once for computing rewards during the Train Controller
            # phase (valid_data, batch size == 64), and once for evaluating ppl
            # over the entire validation set (eval_data, batch size == 1)
            self.valid_data = utils.batchify(dataset.valid,
                                             args.batch_size,
                                             self.cuda)
            self.eval_data = utils.batchify(dataset.valid,
                                            args.test_batch_size,
                                            self.cuda)
            self.test_data = utils.batchify(dataset.test,
                                            args.test_batch_size,
                                            self.cuda)

        self.max_length = self.args.shared_rnn_max_length

        if args.use_tensorboard:
            self.tb = TensorBoard(args.model_dir)
        else:
            self.tb = None
        self.build_model()

        if self.args.load_path:
            self.load_model()

        shared_optimizer = _get_optimizer(self.args.shared_optim)
        controller_optimizer = _get_optimizer(self.args.controller_optim)

        self.shared_optim = shared_optimizer(
            self.shared.parameters(),
            weight_decay=self.args.shared_l2_reg,
            # momentum=0.99,
            # nesterov=True,
            lr=self.args.controller_lr)
        self.args.shared_decay_after = 10e8
        #self.args.entropy_coeff = 1e-6
            #shared_optimizer(
            #self.shared.parameters(),
            #lr=self.shared_lr,
            #weight_decay=self.args.shared_l2_reg)  # TODO: NOTE THAT I ADDED MOMENTUM AND NESTEROV HERE'''

        self.controller_optim = controller_optimizer(
            self.controller.parameters(),
            lr=self.args.controller_lr)

        self.shared_prior_update = None
        self.controller_prior_update = None

        self.ce = nn.CrossEntropyLoss()

    def build_model(self):
        """Creates and initializes the shared and controller models."""
        if self.args.network_type == 'rnn':
            self.shared = RNN(self.args, self.dataset)
        elif self.args.network_type == 'cnn':
            self.shared = CNN(self.args, self.dataset)
        else:
            raise NotImplementedError(f'Network type '
                                      f'`{self.args.network_type}` is not '
                                      f'defined')

        self.controller = SupervisedController(self.args)

        if self.args.num_gpu == 1:
            self.shared.cuda()
            self.controller.cuda()
        elif self.args.num_gpu > 1:
            raise NotImplementedError('`num_gpu > 1` is in progress')

    def train_controller(self):
        """Fixes the shared parameters and updates the controller parameters.

        The controller is updated with a score function gradient estimator
        (i.e., REINFORCE), with the reward being c/valid_ppl, where valid_ppl
        is computed on a minibatch of validation data.

        A moving average baseline is used.

        The controller is trained for 2000 steps per epoch (i.e.,
        first (Train Shared) phase -> second (Train Controller) phase).
        """
        model = self.controller
        model.train()
        # TODO(brendan): Why can't we call shared.eval() here? Leads to loss
        # being uniformly zero for the controller.
        # self.shared.eval()

        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        hidden = self.shared.init_hidden(self.args.batch_size)
        total_loss = 0
        valid_idx = 0#random.randint(0, self.valid_data.size(0) - 1 - self.max_length)#0

        if self.epoch >= self.args.controller_supervised_store_epoch:
            architectures = self.controller.generate_architectures(1000)
            self.controller.store_architectures(architectures)

        for step in range(self.args.controller_max_step):
            # sample models
            dags, log_probs, entropies = self.controller.sample(
                with_details=True)

            if self.epoch >= self.args.controller_supervised_train_epoch:
                supervised_loss = self.controller.predict_architectures(self.controller.architectures)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            # NOTE(brendan): No gradients should be backpropagated to the
            # shared model during controller training, obviously.
            with _get_no_grad_ctx_mgr():
                rewards, hidden, ppl = self.get_reward(dags,
                                                  np_entropies,
                                                  hidden,
                                                  valid_idx)

            # discount
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv_history.extend(adv)

            # policy loss
            loss = -log_probs * utils.get_variable(adv,
                                                   self.cuda,
                                                   requires_grad=False)
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            if self.epoch >= self.args.controller_supervised_train_epoch:
                loss += supervised_loss

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)

            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)

            if ((step % self.args.log_step) == 0) and (step > 0):
                self._summarize_controller_train(total_loss,
                                                 adv_history,
                                                 entropy_history,
                                                 reward_history,
                                                 avg_reward_base,
                                                 dags)

                reward_history, adv_history, entropy_history = [], [], []
                total_loss = 0

            self.controller_step += 1

            prev_valid_idx = valid_idx
            valid_idx = ((valid_idx + self.max_length) % (self.valid_data.size(0) - 1))
            # TODO: I CHANGED THIS.
            # NOTE(brendan): Whenever we wrap around to the beginning of the
            # validation data, we reset the hidden states.
            # TODO: I TOOK OUT HIDDEN RESET
            if prev_valid_idx > valid_idx:
                hidden = self.shared.init_hidden(self.args.batch_size)

    def evaluate(self, source, dag, name, batch_size=1, max_num=None, tb=True):
        """Evaluate on the validation set.

        NOTE(brendan): We should not be using the test set to develop the
        algorithm (basic machine learning good practices).
        """
        self.shared.eval()
        self.controller.eval()

        if max_num is None:
            data = source
        else:
            data = source[:max_num * self.max_length]

        total_loss = 0
        hidden = self.shared.init_hidden(batch_size)

        pbar = range(0, data.size(0) - 1, self.max_length)
        for count, idx in enumerate(pbar):
            inputs, targets = self.get_batch(data, idx, volatile=True)
            output, hidden, _ = self.shared(inputs,
                                            dag,
                                            hidden=hidden,
                                            is_train=False)
            output_flat = output.view(-1, self.dataset.num_tokens)
            total_loss += len(inputs) * self.ce(output_flat, targets).data
            hidden.detach_()
            ppl = math.exp(utils.to_item(total_loss) / (count + 1) / self.max_length)

        val_loss = utils.to_item(total_loss) / len(data)
        ppl = math.exp(val_loss)

        if tb:
            if self.args.mode == "train_scratch":
                param_group_name = "eval_scratch"
            else:
                param_group_name = "eval"

            self.tb.scalar_summary('{}/{}_loss'.format(param_group_name, name), val_loss, self.epoch)
            self.tb.scalar_summary('{}/{}_ppl'.format(param_group_name, name), ppl, self.epoch)

        logger.info(f'val eval | loss: {val_loss:8.2f} | ppl: {ppl:8.2f}')

        return ppl

    def _summarize_controller_train(self,
                                    total_loss,
                                    adv_history,
                                    entropy_history,
                                    reward_history,
                                    avg_reward_base,
                                    dags):
        """Logs the controller's progress for this training epoch."""
        cur_loss = total_loss / self.args.log_step

        avg_adv = np.mean(adv_history)
        avg_entropy = np.mean(entropy_history)
        avg_reward = np.mean(reward_history)

        if avg_reward_base is None:
            avg_reward_base = avg_reward

        logger.info(
            f'controller | epoch {self.epoch:3d} | lr {self.controller_lr:.5f} '
            f'| R {avg_reward:.5f} | entropy {avg_entropy:.4f} '
            f'| loss {cur_loss:.5f}')

        # Tensorboard
        if self.tb is not None:
            self.tb.scalar_summary('controller/loss',
                                   cur_loss,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward',
                                   avg_reward,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward-B_per_epoch',
                                   avg_reward - avg_reward_base,
                                   self.controller_step)
            self.tb.scalar_summary('controller/entropy',
                                   avg_entropy,
                                   self.controller_step)
            self.tb.scalar_summary('controller/adv',
                                   avg_adv,
                                   self.controller_step)


