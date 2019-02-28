"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F

import utils
import math

from network_construction.utils import _construct_dags
from models.controller import Controller

class TiedWeightsController(Controller):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    TODO(brendan): RL controllers do not necessarily have much to do with
    language models.

    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args

        if self.args.network_type == 'rnn':
            # NOTE(brendan): `num_tokens` here is just the activation function
            # for every even step,
            self.num_tokens = [len(args.shared_rnn_activations)]
            for idx in range(self.args.num_blocks):
                self.num_tokens += [idx + 1,
                                    len(args.shared_rnn_activations)]
            self.func_names = args.shared_rnn_activations
        elif self.args.network_type == 'cnn':
            self.num_tokens = [len(args.shared_cnn_types),
                               self.args.num_blocks]
            self.func_names = args.shared_cnn_types

        num_total_tokens = sum(self.num_tokens)

        self.encoder = torch.nn.Embedding(num_total_tokens,
                                          args.controller_hid)
        self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)

        # TODO(brendan): Perhaps these weights in the decoder should be
        # shared? At least for the activation functions, which all have the
        # same size.
        self.decoders = {"activations": None, "connections": None}
        self.decoders["activations"] = torch.nn.Linear(args.controller_hid, self.num_tokens[0])
        self.decoders["connections"] = torch.nn.Linear(args.controller_hid, self.num_tokens[-2])

        self._decoders = torch.nn.ModuleList([self.decoders["activations"], self.decoders["connections"]])

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
        self.most_tokens = max(len(args.shared_rnn_activations), args.num_blocks)
        self.primes = self.generate_primes()
        self.hashes = []

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.args.controller_hid),
                self.args.cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for key, decoder in self.decoders.items():
            decoder.bias.data.fill_(0)

    def forward(self,  # pylint:disable=arguments-differ
                inputs,
                hidden,
                block_idx,
                is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)

        if block_idx % 2 == 0:
            decoder = self.decoders["activations"]
            subset = decoder.out_features
        else:
            decoder = self.decoders["connections"]
            subset = math.ceil(block_idx / 2)

        logits = decoder(hx)
        logits = logits[:, :subset]

        logits /= self.args.softmax_temperature

        # exploration
        if self.args.mode == 'train':
            logits = (self.args.tanh_c*torch.tanh(logits))

        return logits, (hx, cx)
