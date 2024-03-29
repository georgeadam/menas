"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F

import utils
import math

from network_construction.utils import _construct_dags


class PrevActionRegularizedController(torch.nn.Module):
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
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)

        self.regularizers = []
        for i in range(len(self.num_tokens)):
            if i > 0:
                regularizer = torch.nn.Linear(args.controller_hid, self.num_tokens[i - 1])
                self.regularizers.append(regularizer)

        self._regularizers = torch.nn.ModuleList(self.regularizers)

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
        for decoder in self.decoders:
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
        logits = self.decoders[block_idx](hx)

        logits /= self.args.softmax_temperature

        # exploration
        if self.args.mode == 'train':
            logits = (self.args.tanh_c*torch.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None, return_hidden=False,
               random_hidden_state=False, return_hashes=False):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        if random_hidden_state:
            inputs = torch.randn(inputs.shape).cuda()
            temp1 = torch.randn(hidden[0].shape)
            temp2 = torch.randn(hidden[1].shape)

            if self.args.cuda:
                temp1, temp2 = temp1.cuda(), temp2.cuda()

            hidden = (temp1, temp2)

        activations = []
        entropies = []
        log_probs = []
        prev_nodes = []
        max_probabilities = []
        prev_probs = None
        mses = []
        # NOTE(brendan): The RNN controller alternately outputs an activation,
        # followed by a previous node, for each block except the last one,
        # which only gets an activation function. The last node is the output
        # node, and its previous node is the average of all leaf nodes.
        for block_idx in range(2*(self.args.num_blocks - 1) + 1):
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx,
                                          is_embed=(block_idx == 0))

            # Sample DAG actions
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            # TODO(brendan): .mean() for entropy?
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(1, utils.get_variable(action, requires_grad=False))

            # TODO(brendan): why the [:, 0] here? Should it be .squeeze(), or
            # .view()? Same below with `action`.
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            # 0: function, 1: previous node
            mode = block_idx % 2
            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:mode]),
                requires_grad=False)

            if mode == 0:
                activations.append(action[:, 0])
            elif mode == 1:
                prev_nodes.append(action[:, 0])

            if return_hidden:
                max_probabilities.append(torch.max(probs, 1)[0].view(-1, 1))

            # Regularize hidden state to predict previous action
            if block_idx > 0:
                prev_action_logits = self.regularizers[block_idx - 1](hidden[0])
                prev_action_probs = F.softmax(prev_action_logits, dim=-1)
                mse = F.mse_loss(prev_action_probs, prev_probs).view(1)
                mses.append(mse)

            prev_probs = probs

        prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
        activations = torch.stack(activations).transpose(0, 1)

        hashes = self.hash_dags(prev_nodes, activations)
        self.hashes += hashes

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
                return dags, torch.cat(log_probs), torch.cat(entropies), torch.cat(mses), hidden[0], \
                       torch.cat(max_probabilities, dim=1).detach()
            elif return_hashes:
                return dags, torch.cat(log_probs), torch.cat(entropies), torch.cat(mses), hashes
            else:
                return dags, torch.cat(log_probs), torch.cat(entropies), torch.cat(mses)

        if return_hidden:
            return dags, hidden[0], torch.cat(max_probabilities, dim=1).detach()
        else:
            return dags

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False))

    def hash_dags(self, prev_nodes, activations):
        hashes = []

        for i in range(len(prev_nodes)):
            pn = prev_nodes[i]
            act = activations[i]
            hash = 1
            block_num = 0

            for j in range(len(pn)):
                node = pn[j]
                hash *= self.primes[node + block_num * self.most_tokens]
                block_num += 1

                activation = act[j]
                hash *= self.primes[activation + block_num * self.most_tokens]
                block_num += 1

            hashes.append(hash)

        return hashes

    def generate_primes(self):
        num_primes = 2 * (self.most_tokens ** 2) + self.most_tokens

        primes = []
        j = 2

        for i in range(num_primes):
            if len(primes) == num_primes:
                break

            while True:
                if self.is_prime(j):
                    primes.append(j)
                    j += 1

                    break

                j += 1

        return primes

    def is_prime(self, n):
        for i in range(2, math.ceil(math.sqrt(n)) + 1):
            if n % i == 0:
                return False

        return True