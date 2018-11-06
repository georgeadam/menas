"""Entry point."""
import torch

from data.image import Image
from data.text import Corpus

from configs import config_ours as config
from train_scripts import regular_trainer as trainer
import utils as utils

from models.controller import Node

import collections
import json
import os
from dotenv import find_dotenv, load_dotenv

from settings import ROOT_DIR

logger = utils.get_logger()


def remove_node(dag, remove_idx):
    # TODO (Alex): Fix to remove remaining nodes without parents
    new_dag = collections.defaultdict(list)

    for idx, nodes in dag.items():
        temp_nodes = []

        if idx == remove_idx:
            continue

        for node in nodes:
            if node.id == remove_idx:
                temp_node = None
            else:
                temp_node = Node(node.id, node.name)

            if temp_node is not None:
                temp_nodes.append(temp_node)

        if len(temp_nodes) > 0:
            new_dag[idx] = temp_nodes

    return new_dag


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)
    load_dotenv(find_dotenv(), override=True)

    node_ablation_dir = os.environ.get("NODE_ABLATION_DIR")
    model_dir = os.path.basename(args.model_dir)
    save_dir = os.path.join(ROOT_DIR, node_ablation_dir, model_dir)

    train_args = utils.load_args(args.model_dir)
    utils.makedirs(save_dir)

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

    nodes = set(range(1, args.num_blocks))
    results = {}

    for idx in nodes:
        temp_dag = remove_node(dag, idx)
        ppl = trnr.get_perplexity_multibatch(trnr.eval_data, temp_dag, 1)

        print("Validation PPL when removed node {} is: {}".format(idx, ppl))

    with open(os.path.join(save_dir, "params.json"), "w") as fp:
        json.dump(train_args.__dict__, fp, indent=4, sort_keys=True)

    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)