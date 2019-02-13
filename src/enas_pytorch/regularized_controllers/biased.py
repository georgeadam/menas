"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F

import utils
import math

from network_construction.utils import _construct_dags
from models.controller import Controller

class BiasedController(Controller):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    TODO(brendan): RL controllers do not necessarily have much to do with
    language models.

    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self, args):
        super(BiasedController, self).__init__(args)
        self.init_weights()


    def init_weights(self):
        params = self.lstm.named_parameters()
        params = [(n,p) for n, p in params]

        idx = self.args.controller_hid

        params[2][1][idx: 2 * idx].data.fill_(5)
        params[3][1][idx: 2 * idx].data.fill_(5)

        return
