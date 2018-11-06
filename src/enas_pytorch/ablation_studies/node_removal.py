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


def remove_node_hard(dag, remove_idx):
    """
    Given a DAG and an index to remove, this function creates a new DAG from scratch with the node having remove_idx
    removed, as well as any nodes it feeds into, and any nodes that feed into it. This has a cascading effect for
    the nodes that remove_idx feeds into, and the nodes after that, etc.

    Args:
        dag: A dictionary of Nodes.
        remove_idx: The idx of the Node to be removed.

    Returns:
        A new DAG with the desired node, it's parents, child, grandchildren, etc. removed.
    """
    # TODO (Alex): Fix to remove remaining nodes without parents
    new_dag = collections.defaultdict(list)
    child_idxes = [remove_idx]

    for idx, nodes in dag.items():
        temp_nodes = []

        if idx in child_idxes:
            child_idxes.remove(idx)

            for node in nodes:
                if node.name != 'avg' and node.name != 'h[t]':
                    child_idxes.append(node.id)

            continue

        for node in nodes:
            if node.id in child_idxes:
                temp_node = None
            else:
                temp_node = Node(node.id, node.name)

            if temp_node is not None:
                temp_nodes.append(temp_node)

        if len(temp_nodes) > 0:
            new_dag[idx] = temp_nodes

    return new_dag


def remove_node_reconnect(dag, remove_idx):
    """
    Given a DAG and an index to remove, this function creates a new DAG from scratch with the node having remove_id
    removed and its parents reconnected to its children when possible.

    Args:
        dag: A dictionary of Nodes.
        remove_idx: The idx of the Node to be removed.

    Returns:
        A new DAG with the desired node removed, and its parents reconnected to its children.
    """
    new_dag = collections.defaultdict(list)
    child_nodes = []
    parent_nodes = []

    for idx, nodes in dag.items():
        temp_nodes = []

        if idx == remove_idx:
            for node in nodes:
                child_nodes.append(Node(node.id, node.name))

            continue

        for node in nodes:
            if node.id == remove_idx:
                parent_nodes.append(Node(node.id, node.name))
                temp_node = None
            else:
                temp_node = Node(node.id, node.name)

            if temp_node is not None:
                temp_nodes.append(temp_node)

        if len(temp_nodes) > 0:
            new_dag[idx] = temp_nodes

    new_dag[parent_nodes[0].id] = child_nodes

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
        temp_dag = remove_node_hard(dag, idx)
        ppl = trnr.get_perplexity_multibatch(trnr.eval_data, temp_dag, 1)

        results[idx] = ppl

        print("Validation PPL when removed node {} is: {}".format(idx, ppl))

    with open(os.path.join(save_dir, "params.json"), "w") as fp:
        json.dump(train_args.__dict__, fp, indent=4, sort_keys=True)

    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)