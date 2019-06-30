"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F

import utils
import math

from network_construction.utils import _construct_dags
from models.controller import Controller


class SupervisedController(Controller):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    TODO(brendan): RL controllers do not necessarily have much to do with
    language models.

    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self, args):
        super(SupervisedController, self).__init__(args)

        self.stored_architectures = None
        self.max_architectures = 10000

    def predict_architectures(self, architectures):
        inputs = architectures[:, 0]
        hidden = self.static_init_hidden[architectures.shape[0]]

        softmax = torch.nn.CrossEntropyLoss(reduction="none")
        losses = []

        for i in range(architectures.shape[1] - 1):
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          i + 1,
                                          is_embed=False)

            probs = F.softmax(logits, dim=-1)
            _, preds = torch.max(probs, dim=1)

            loss = softmax(probs, architectures[:, i + 1])
            losses.append(loss)

            inputs = architectures[:, i + 1]

        return torch.mean(torch.cat(losses))

    def store_architectures(self, architectures):
        if self.stored_architectures is None:
            self.stored_architectures = architectures
        else:
            if len(self.stored_architectures) < self.max_architectures:
                self.stored_architectures = torch.cat([self.stored_architectures, architectures])
            else:
                idxes = torch.randperm(len(architectures))
                idxes = idxes[:100]
                updates = architectures[idxes]
                self.stored_architectures = torch.cat([self.stored_architectures[:-100], updates])

    @property
    def architectures(self):
        idxes = torch.randperm(len(self.stored_architectures))
        idxes = idxes[:100]

        return self.stored_architectures[idxes]