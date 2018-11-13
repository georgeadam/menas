"""Entry point."""
import torch

from data.image import Image
from data.text import Corpus

from configs import config_ablation as config
from train_scripts import regular_trainer
from train_scripts import random_trainer
from train_scripts import hardcoded_trainer
import utils as utils

from network_construction.utils import Node

import collections
import json
import os
from dotenv import find_dotenv, load_dotenv

from dotmap import DotMap
from settings import ROOT_DIR

logger = utils.get_logger()


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

    for idx, nodes in dag.items():
        temp_nodes = []

        if idx == remove_idx:
            for node in nodes:
                if node.id != max(dag.keys()) or len(nodes) == 1:
                    child_nodes.append(Node(node.id, node.name))

            continue

        for node in nodes:
            if node.id == remove_idx:
                parent_idx = idx
                temp_node = None
            else:
                temp_node = Node(node.id, node.name)

            if temp_node is not None:
                temp_nodes.append(temp_node)

        if len(temp_nodes) > 0:
            new_dag[idx] = temp_nodes

    if parent_idx not in new_dag.keys() and len(child_nodes) > 0:
        new_dag[parent_idx] = child_nodes
    elif len(child_nodes) > 0:
        new_dag[parent_idx] += child_nodes

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
    train_args = DotMap(train_args)
    original_mode = train_args.mode
    train_args.mode = "derive"
    train_args.load_path = args.load_path
    utils.makedirs(save_dir)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True

    if train_args.network_type == 'rnn':
        dataset = Corpus(train_args.data_path)
    elif train_args.network_type == 'cnn':
        dataset = Image(train_args)
    else:
        raise NotImplementedError(f"{train_args.dataset} is not supported")

    if train_args.train_type == "enas":
        trnr = regular_trainer.Trainer(train_args, dataset)
    elif train_args.trian_type == 'random':
        trnr = random_trainer.RandomTrainer(train_args, dataset)
    elif train_args.train_type == "hardcoded":
        trnr = hardcoded_trainer.HardcodedTrainer(train_args, dataset)

    dag = trnr.derive()

    validation_ppl = trnr.get_perplexity_multibatch(trnr.eval_data, dag)
    print("Original performance on entire validation set: {}".format(validation_ppl))
    test_ppl = trnr.get_perplexity_multibatch(trnr.test_data, dag)
    print("Original performance on entire test set: {}".format(test_ppl))

    nodes = set(range(1, train_args.num_blocks))
    results = {"validation": {"original_performance": validation_ppl}, "test": {"original_performance": test_ppl}}

    for idx in nodes:
        temp_dag = remove_node_reconnect(dag, idx)
        validation_ppl = trnr.get_perplexity_multibatch(trnr.eval_data, temp_dag)
        test_ppl = trnr.get_perplexity_multibatch(trnr.test_data, temp_dag)

        results["validation"][idx] = validation_ppl
        results["test"][idx] = test_ppl

        print("Validation PPL when removed node {} is: {}".format(idx, validation_ppl))
        print("Test PPL when removed node {} is: {}".format(idx, test_ppl))

    train_args.mode = original_mode

    with open(os.path.join(save_dir, "params.json"), "w") as fp:
        json.dump(train_args.toDict(), fp, indent=4, sort_keys=True)

    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)