import torch
import utils as utils
import os

import torch.nn.functional as F

from network_construction.utils import _construct_dags
from models.controller import Controller

from hardcoded_networks.chain import generate_prev_idx as generate_prev_idx_chain
from hardcoded_networks.tree import generate_prev_idx as generate_prev_idx_tree


class HardcodedController(Controller):
    def __init__(self, args):
        super().__init__(args)

        if args.architecture == "chain":
            self.generate_prev_idx = generate_prev_idx_chain
        elif args.architecture == "tree":
            self.generate_prev_idx = generate_prev_idx_tree

    def sample(self, batch_size=1, with_details=False, save_dir=None, return_hidden=False):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        activations = []
        entropies = []
        log_probs = []
        prev_nodes = []
        # NOTE(brendan): The RNN controller alternately outputs an activation,
        # followed by a previous node, for each block except the last one,
        # which only gets an activation function. The last node is the output
        # node, and its previous node is the average of all leaf nodes.
        for block_idx in range(2 * (self.args.num_blocks - 1) + 1):
            mode = block_idx % 2

            if mode == 0:
                logits, hidden = self.forward(inputs,
                                              hidden,
                                              block_idx,
                                              is_embed=(block_idx == 0))

                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)

                entropy = -(log_prob * probs).sum(1, keepdim=False)

                action = probs.multinomial(num_samples=1).data

                selected_log_prob = log_prob.gather(1, utils.get_variable(action, requires_grad=False))

                entropies.append(entropy)
                log_probs.append(selected_log_prob[:, 0])

                inputs = utils.get_variable(
                    action[:, 0] + sum(self.num_tokens[:mode]),
                    requires_grad=False)
                activations.append(action[:, 0])
            elif mode == 1:
                hardcoded_prev = self.generate_prev_idx(block_idx)
                action = (torch.ones(batch_size, 1) * hardcoded_prev).long()

                if self.args.cuda:
                    action = action.cuda()

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

        if with_details:
            if return_hidden:
                return dags, torch.cat(log_probs), torch.cat(entropies), hidden[0]
            else:
                return dags, torch.cat(log_probs), torch.cat(entropies)

        if return_hidden:
            return dags, hidden[0]
        else:
            return dags
