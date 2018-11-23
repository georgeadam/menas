from models.controller import Controller
import torch
import torch.nn.functional as F

import utils

from network_construction.utils import _construct_dags
import os

from math import ceil


class FlexibleController(Controller):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args

        if self.args.network_type == 'rnn':
            # NOTE(brendan): `num_tokens` here is just the activation function
            # for every even step,
            self.num_tokens = [args.num_blocks, len(args.shared_rnn_activations)]
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
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.args.controller_hid),
                self.args.cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def sample(self, batch_size=1, with_details=False, save_dir=None, return_hidden=False):
        """First samples the number of nodes
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        # Sample the number of nodes first
        logits, hidden = self.forward(inputs, hidden, 0, is_embed=True)
        probs = F.softmax(logits, dim=-1)
        num_blocks = probs.multinomial(num_samples=1).data
        num_blocks[num_blocks < 3] = 3
        list_num_blocks = num_blocks.tolist()
        list_num_blocks = set([temp[0] for temp in list_num_blocks if temp[0] > 2])
        activations = {key: [] for key in list_num_blocks}
        entropies = {key: [] for key in list_num_blocks}
        log_probs = {key: [] for key in list_num_blocks}
        prev_nodes = {key: [] for key in list_num_blocks}

        inputs = utils.get_variable(
            num_blocks[:, 0] + sum(self.num_tokens[:0]),
            requires_grad=False)

        for block_idx in range(0, 2 * (self.args.num_blocks - 1) + 1):
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx + 1,
                                          False)

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            # TODO(brendan): .mean() for entropy?
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(1, utils.get_variable(action, requires_grad=False))

            # TODO(brendan): why the [:, 0] here? Should it be .squeeze(), or
            # .view()? Same below with `action`.
            mode = (block_idx) % 2

            for nb in list_num_blocks:
                idxes = (num_blocks == nb) * (num_blocks > (ceil(block_idx / 2)))

                if len(idxes.nonzero().shape) == 1:
                    continue

                idxes = idxes.nonzero()[:, 0]

                sub_entropy = entropy[idxes]

                sub_action = action[idxes]
                sub_selected_log_prob = selected_log_prob[idxes]

                entropies[nb].append(sub_entropy)
                log_probs[nb].append(sub_selected_log_prob[:, 0])

                # 0: function, 1: previous node
                if mode == 0:
                    activations[nb].append(sub_action[:, 0])
                elif mode == 1:
                    prev_nodes[nb].append(sub_action[:, 0])

            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:mode]),
                requires_grad=False)

        list_log_probs = []
        list_entropies = []
        for nb in list_num_blocks:
            prev_nodes[nb] = torch.stack(prev_nodes[nb]).transpose(0, 1)
            activations[nb] = torch.stack(activations[nb]).transpose(0, 1)
            list_log_probs += log_probs[nb]
            list_entropies += entropies[nb]

        dags = []

        for nb in list_num_blocks:
            temp_dags = _construct_dags(prev_nodes[nb],
                                   activations[nb],
                                   self.func_names,
                                   nb)
            dags += temp_dags

        if save_dir is not None:
            for idx, dag in enumerate(dags):
                utils.draw_network(dag,
                                   os.path.join(save_dir, f'graph{idx}.png'))

        if with_details:
            if return_hidden:
                return dags, torch.cat(list_log_probs), torch.cat(list_entropies), hidden[0]
            else:
                return dags, torch.cat(list_log_probs), torch.cat(list_entropies)

        if return_hidden:
            return dags, hidden[0]
        else:
            return dags