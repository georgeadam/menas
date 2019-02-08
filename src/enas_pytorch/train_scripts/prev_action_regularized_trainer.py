"""The module for training ENAS shared params only with a random controller."""
import json
import os
import utils
import torch
import numpy as np
import math
import torch.nn as nn

from train_scripts.regular_trainer import Trainer, _get_no_grad_ctx_mgr, discount, TensorBoard, _get_optimizer
from models.shared_cnn import CNN
from models.shared_rnn import RNN
from models.prev_action_regularized_controller import PrevActionRegularizedController

logger = utils.get_logger()


class PrevActionRegularizedTrainer(Trainer):
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
            lr=self.args.controller_lr)
        self.args.shared_decay_after = 10e8

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

        self.controller = PrevActionRegularizedController(self.args)

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
        self.determine_dag_sampled_during_training()
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
        mse_history = []

        hidden = self.shared.init_hidden(self.args.batch_size)
        total_loss = 0
        valid_idx = 0#random.randint(0, self.valid_data.size(0) - 1 - self.max_length)#0
        for step in range(self.args.controller_max_step):
            # sample models
            dags, log_probs, entropies, mses = self.controller.sample(
                with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            np_mses = mses.data.cpu().numpy()
            # NOTE(brendan): No gradients should be backpropagated to the
            # shared model during controller training, obviously.
            with _get_no_grad_ctx_mgr():
                rewards, hidden, ppl = self.get_reward(dags,
                                                       np_entropies,
                                                       np_mses,
                                                       hidden,
                                                       valid_idx)

            # discount
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)
            mse_history.extend(np_mses)

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

            mses = torch.cat((torch.Tensor([0.0]).cuda(), mses))
            loss += mses
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
                                                 mse_history,
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

    def get_reward(self, dag, entropies, mses, hidden, valid_idx=0):
        """Computes the perplexity of a single sampled model on a minibatch of
        validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()
            mses = mses.data.cpu().numpy()

        inputs, targets = self.get_batch(self.valid_data,
                                         valid_idx,
                                         self.max_length,
                                         volatile=True)
        valid_loss, hidden, _ = self.get_loss(inputs, targets, hidden, dag)
        valid_loss = utils.to_item(valid_loss.data)

        valid_ppl = math.exp(valid_loss)

        # TODO: we don't know reward_c
        if self.args.ppl_square:
            # TODO: but we do know reward_c=80 in the previous paper
            R = self.args.reward_c / valid_ppl ** 2
        else:
            R = self.args.reward_c / valid_ppl

        # Interesting how R is just a scalar, and that the entropies are the only thing quantities which are distinct
        # at the different time steps (decisions) that the controller makes. It brings up the question if when we the
        # controller creates an architecture, should each individual choice be considered as an action, or should the
        # generated architecture as a whole be considered as an action?
        if self.args.entropy_mode == 'reward':
            rewards = R + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = R * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards, hidden, valid_ppl

    def _summarize_controller_train(self,
                                    total_loss,
                                    adv_history,
                                    entropy_history,
                                    mse_history,
                                    reward_history,
                                    avg_reward_base,
                                    dags):
        """Logs the controller's progress for this training epoch."""
        cur_loss = total_loss / self.args.log_step

        avg_adv = np.mean(adv_history)
        avg_entropy = np.mean(entropy_history)
        avg_mse = np.mean(mse_history)
        avg_reward = np.mean(reward_history)

        if avg_reward_base is None:
            avg_reward_base = avg_reward

        logger.info(
            f'controller | epoch {self.epoch:3d} | lr {self.controller_lr:.5f} '
            f'| R {avg_reward:.5f} | entropy {avg_entropy:.4f} | mse {avg_mse:.7f}'
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
            self.tb.scalar_summary('controller/mse',
                                   avg_mse,
                                   self.controller_step)

    def derive(self, sample_num=None, valid_idx=0, create_image=True):
        """TODO(brendan): We are always deriving based on the very first batch
        of validation data? This seems wrong...
        """
        if sample_num is None:
            sample_num = self.args.derive_num_sample

        dags, _, entropies, mses = self.controller.sample(sample_num, with_details=True)

        best_ppl = float("inf")
        best_dag = None
        for dag in dags:
            # We can now evaluate performance on entire dataset, not just first batch, so we use
            # get_perplexity_multibatch
            # R, _, ppl = self.get_reward(dag, entropies, hidden, valid_idx)
            # if R.max() > max_R:
            #     max_R = R.max()
            #     best_dag = dag
            ppl = self.get_perplexity_multibatch(self.eval_data, dag, self.args.batch_size, 1)

            if ppl < best_ppl:
                best_ppl = ppl
                best_dag = dag

        logger.info(f'derive | best PPL: {best_ppl:8.6f}')
        fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
                 f'{best_ppl:6.4f}-best.png')

        if create_image:
            path = os.path.join(self.args.model_dir, 'networks', fname)
            utils.draw_network(best_dag, path)
            self.tb.image_summary('derive/best', [path], self.epoch)

        json_architecture_path = os.path.join(self.args.model_dir, 'derived_architecture.json')

        with open(json_architecture_path, 'w') as fp:
            json.dump(best_dag, fp, indent=4, sort_keys=True)

        return best_dag