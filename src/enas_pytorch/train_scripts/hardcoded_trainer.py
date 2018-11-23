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
from hardcoded_networks.controller import HardcodedController

from train_scripts.regular_trainer import Trainer, _get_optimizer, _get_no_grad_ctx_mgr

logger = utils.get_logger()


class HardcodedTrainer(Trainer):
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

        self.controller = HardcodedController(self.args)

        if self.args.num_gpu == 1:
            self.shared.cuda()
            self.controller.cuda()
        elif self.args.num_gpu > 1:
            raise NotImplementedError('`num_gpu > 1` is in progress')
