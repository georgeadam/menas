import collections
import utils
import os

from hardcoded_networks.chain import generate_chain
from hardcoded_networks.tree import generate_tree

Node = collections.namedtuple('Node', ['id', 'name'])


class HardcodedArchitectureGenerator():
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

        if args.architecture == "chain":
            self.generate_function = generate_chain
        elif args.architecture == "tree":
            self.generate_function = generate_tree

    def sample(self, batch_size=1, with_details=False, save_dir=None):
        dags = []

        for i in range(0, batch_size):
            dags.append(self.generate_function(self.args.num_blocks, self.func_names))

        if save_dir is not None:
            for idx, dag in enumerate(dags):
                utils.draw_network(dag,
                                   os.path.join(save_dir, f'graph{idx}.png'))

        return dags