"""The module for training ENAS."""
import contextlib
import glob
import math
import os

import numpy as np
import random
import scipy.signal
from tensorboard import TensorBoard
import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable

import models
import utils

from models.shared_rnn import RNN
from models.shared_cnn import CNN
from flexible.controller import FlexibleController

from train_scripts.regular_trainer import Trainer, _get_optimizer, _get_no_grad_ctx_mgr, discount

logger = utils.get_logger()


class FlexibleTrainer(Trainer):
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
            #momentum=0.99,
            #nesterov=True,
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

        self.controller = FlexibleController(self.args)

        if self.args.num_gpu == 1:
            self.shared.cuda()
            self.controller.cuda()
        elif self.args.num_gpu > 1:
            raise NotImplementedError('`num_gpu > 1` is in progress')

    def train(self):
        """Cycles through alternately training the shared parameters and the
        controller, as described in Section 2.2, Training ENAS and Deriving
        Architectures, of the paper.

        From the paper (for Penn Treebank):

        - In the first phase, shared parameters omega are trained for 400
          steps, each on a minibatch of 64 examples.

        - In the second phase, the controller's parameters are trained for 2000
          steps.
        """
        if self.args.shared_initial_step > 0:
            self.train_shared(self.args.shared_initial_step)
            self.train_controller()

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. Training the shared parameters omega of the child models
            self.train_shared()

            # 2. Training the controller parameters theta
            self.train_controller()

            if self.epoch % self.args.save_epoch == 0:
                with _get_no_grad_ctx_mgr():
                    best_dag = self.derive()
                    eval_ppl = self.evaluate(self.eval_data,
                                  best_dag,
                                  'val_best',
                                  max_num=None) # Can now evaluate on entire dataset since we're using
                    # [-1,64] instead of [-1, 1] shape, which makes things way faster

                if eval_ppl < self.best_ppl:
                    self.best_ppl = eval_ppl
                    self.save_model(self.shared_path, self.controller_path)

            if self.epoch >= self.args.shared_decay_after:
                utils.update_lr(self.shared_optim, self.shared_lr)

        # Added to test the model on the entire validation and entire test set after training is done.
        # load_model() is called first to load the best saved params from file as the ones that are
        # currently in memory could be very overfit.
        self.load_model()
        self.test()
        self.indicate_training_complete()

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
        for step in range(self.args.controller_max_step):
            # sample models
            dags, log_probs, entropies = self.controller.sample(
                with_details=True)

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
                # Extending the baseline shape by padding with 0s means the values in those extended positions
                # correspond to a different amount of time steps than values in earlier positions. I.e. if we only
                # sample large DAGs occasionally, the baseline for the later controller actions will be less mature
                # than that for earlier actions.
                # Recall that the point of a baseline, in addition to variance reduction, it to increase the
                # probability of only the best actions. Without it, assuming we have all positive rewards,
                # all action probabilities are increased.
                if baseline.shape[0] < rewards.shape[0]:
                    padding = np.zeros(rewards.shape[0] - baseline.shape[0])

                    baseline = decay * np.concatenate((baseline, padding)) + (1 - decay) * rewards
                # Extending the rewards shape by padding with 0s means that the sampled architecture is smaller than
                # the largest one seen thus far. Thus, we only want the earlier controller actions to have their
                # baseline updated. Therefore, we pad not with 0s, but with what the current baseline vector is past
                # the shape of the reward vector.
                elif baseline.shape[0] > rewards.shape[0]:
                    # padding = np.zeros(baseline.shape[0] - rewards.shape[0])
                    padding = baseline[rewards.shape[0]:]

                    baseline = decay * baseline + (1 - decay) * np.concatenate((rewards, padding))
                else:
                    baseline = decay * baseline + (1 - decay) * rewards

            # Even though we had to use some padding tricks to update the baseline above, we never actually padded
            # the reward or baseline, just the terms used in the baseline update. Thus, here if the shape of the
            # baseline and reward are still off, then we have to do some padding. 
            if baseline.shape[0] < rewards.shape[0]:
                # In this scenario, the sampled DAG is larger than any DAG seen so far -> more actions. So
                # zero-padding makes sense.
                padding = np.zeros(rewards.shape[0] - baseline.shape[0])
                baseline = np.concatenate((baseline, padding))
            elif rewards.shape[0] < baseline.shape[0]:
                # In this scenario, the sampled DAG is smaller than the biggest DAG seen so for -> less actions. So
                # zero padding does not make sense since that means that the advantage computed for the actions that
                # weren't seen when creating this DAG would be negative, when in fact it should be 0 since they
                # simply were not observed this episode. To make the advantage for these tail actions be 0,
                # we pad with the baseline, rather than with 0s.
                # padding = np.zeros(baseline.shape[0] - rewards.shape[0])
                padding = baseline[rewards.shape[0]:]
                rewards = np.concatenate((rewards, padding))

            # Compute the advantage of the current actions over the baseline
            adv = rewards - baseline
            adv_history.extend(adv)

            # policy loss
            # So the log probabilities simply correspond to the probabilities of the selected actions.
            # I think by this point it is not possible for log_probs to be longer than adv based on the padding
            # we did earlier with the rewards and baseline. However, it might be shorter than adv. in which case
            # I think zero padding makes sense since we want those padded actions to have 0 effect on the loss.
            if log_probs.shape[0] < adv.shape[0]:
                padding = torch.zeros(adv.shape[0] - log_probs.shape[0])

                if self.args.cuda:
                    padding = padding.cuda()

                log_probs = torch.cat((log_probs, padding))
            elif adv.shape[0] < log_probs.shape[0]:
                padding = torch.zeros(log_probs.shape[0] - adv.shape[0])

                if self.args.cuda:
                    padding = padding.cuda()

                adv = torch.cat((adv, padding))

            loss = -log_probs * utils.get_variable(adv,
                                                   self.cuda,
                                                   requires_grad=False)
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

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
