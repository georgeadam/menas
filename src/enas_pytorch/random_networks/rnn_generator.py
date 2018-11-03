import torch.nn.functional as F

import torch
import collections
import utils
import os

from models.controller import _construct_dags

Node = collections.namedtuple('Node', ['id', 'name'])

class RandomArchitectureGenerator():
    def __init__(self, args):
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

    def sample(self, batch_size=1, with_details=False, save_dir=None):
        activations = []
        prev_nodes = []

        min_logit = -30.0
        max_logit = 30.0

        for block_idx in range(2 * (self.args.num_blocks - 1) + 1):
            logits = (min_logit - max_logit) * torch.rand(batch_size, self.num_tokens[block_idx]) + max_logit

            probs = F.softmax(logits, dim=-1)
            action = probs.multinomial(num_samples=1)

            # 0: function, 1: previous node
            mode = block_idx % 2

            if mode == 0:
                activations.append(action[:, 0])
            elif mode == 1:
                prev_nodes.append(action[:, 0])

        prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
        activations = torch.stack(activations).transpose(0, 1)

        dags = _construct_dags(prev_nodes,
                               activations,
                               self.func_names,
                               self.args.num_blocks)

        if save_dir is not None:
            for idx, dag in enumerate(dags):
                utils.draw_network(dag,
                                   os.path.join(save_dir, f'graph{idx}.png'))

        return dags