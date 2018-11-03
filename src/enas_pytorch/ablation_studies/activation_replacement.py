"""Entry point."""
import torch

from data.image import Image
from data.text import Corpus

from configs import config_ours as config
from train_scripts import regular_trainer as trainer
import utils as utils

from models.controller import Node


import collections

logger = utils.get_logger()


def replace_all_activations(dag, replacement="tanh"):
    new_dag = collections.defaultdict(list)

    for idx, nodes in dag.items():
        temp_nodes = []

        for node in nodes:
            if node.name != "avg":
                temp_node = Node(node.id, replacement)
            else:
                temp_node = Node(node.id, "avg")

            temp_nodes.append(temp_node)

        new_dag[idx] = temp_nodes

    return new_dag


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True

    if args.network_type == 'rnn':
        dataset = Corpus(args.data_path)
    elif args.network_type == 'cnn':
        dataset = Image(args)
    else:
        raise NotImplementedError(f"{args.dataset} is not supported")

    trnr = trainer.Trainer(args, dataset)

    dag = trnr.derive()

    ppl = trnr.get_perplexity_multibatch(trnr.eval_data, dag, 1)
    print("Original performance on entire validation set: {}".format(ppl))

    activations = ["tanh", "relu", "sigmoid", "identity"]

    for activation in activations:
        temp_dag = replace_all_activations(dag, activation)
        ppl = trnr.get_perplexity_multibatch(trnr.eval_data, temp_dag, 1)

        print("Validation PPL when using all {} activations is: {}".format(activation, ppl))


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)