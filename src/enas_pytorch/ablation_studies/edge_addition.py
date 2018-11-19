"""This ablation  is more like edge replacement. In the case where an edge is adding going to a node that
already has another edge going into it, that existing edge just gets replaced due to logic """
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


def add_edge(dag, from_idx, to_idx, activation="relu"):
    new_dag = collections.defaultdict(list)

    # Construct a new dag from the old dag to avoid overwriting things
    for idx, nodes in dag.items():
        temp_nodes = []

        for node in nodes:
            temp_node = Node(node.id, node.name)

            temp_nodes.append(temp_node)

        new_dag[idx] = temp_nodes

    from_nodes = new_dag[from_idx]

    # Check if edge already exists and don't add another one if so
    for node in from_nodes:
        if node.id == to_idx or to_idx == max(dag.keys()):
            return False

    # If new edge points to the last node in cell, use the avg activation instead of the specified activation
    new_child = None

    if to_idx == max(dag.keys()):
        new_child = Node(to_idx, "avg")
    elif (len(new_dag[from_idx]) == 1 and new_dag[from_idx][0].id != max(dag.keys())) or len(new_dag[from_idx]) > 1:
        new_child = Node(to_idx, activation)

    if new_child is None:
        return False
    else:
        new_dag[from_idx].append(new_child)

    return new_dag


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)
    load_dotenv(find_dotenv(), override=True)

    node_ablation_dir = os.environ.get("EDGE_ADDITION_DIR")
    model_dir = os.path.basename(args.model_dir)
    save_dir = os.path.join(ROOT_DIR, node_ablation_dir, model_dir)

    train_args = utils.load_args(args.model_dir)
    train_args = DotMap(train_args)
    original_mode = train_args.mode
    original_test_batch_size = train_args.test_batch_size  # To make backwards compatible for models whose configs had
    # test batch size of 1
    train_args.mode = "derive"
    train_args.load_path = args.load_path
    train_args.test_batch_size = train_args.batch_size
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

    if train_args.train_type == "enas" or train_args.train_type == "ours" or train_args.train_type == "orig":
        trnr = regular_trainer.Trainer(train_args, dataset)
    elif train_args.train_type == 'random':
        trnr = random_trainer.RandomTrainer(train_args, dataset)
    elif train_args.train_type == "hardcoded":
        trnr = hardcoded_trainer.HardcodedTrainer(train_args, dataset)

    dag = trnr.derive(create_image=False)

    validation_ppl = trnr.get_perplexity_multibatch(trnr.eval_data, dag)
    print("Original performance on entire validation set: {}".format(validation_ppl))
    test_ppl = trnr.get_perplexity_multibatch(trnr.test_data, dag)
    print("Original performance on entire test set: {}".format(test_ppl))

    results = {"validation": {"original_performance": validation_ppl}, "test": {"original_performance": test_ppl}}
    activations = ["tanh", "relu", "sigmoid", "identity"]

    for activation in activations:
        for from_idx in range(4, train_args.num_blocks):
            for to_idx in range(from_idx + 1, train_args.num_blocks + 1):
                temp_dag = add_edge(dag, from_idx, to_idx, activation)

                if temp_dag == False:
                    continue

                validation_ppl = trnr.get_perplexity_multibatch(trnr.eval_data, temp_dag)
                test_ppl = trnr.get_perplexity_multibatch(trnr.test_data, temp_dag)

                results["validation"]["{}_{}_{}".format(from_idx, to_idx, activation)] = validation_ppl
                results["test"]["{}_{}_{}".format(from_idx, to_idx, activation)] = test_ppl

                print("Validation PPL when edge added from node {} to {} with activation {} is: {}".format(from_idx,
                                                                                                           to_idx,
                                                                                                           activation,
                                                                                                           validation_ppl))
                print("Test PPL when edge added from node {} to {} with activation {} is: {}".format(from_idx,
                                                                                                     to_idx,
                                                                                                     activation,
                                                                                                     validation_ppl))

                with open(os.path.join(save_dir, "results.json"), "w") as fp:
                    json.dump(results, fp, indent=4, sort_keys=True)

    train_args.mode = original_mode
    train_args.test_batch_size = original_test_batch_size

    with open(os.path.join(save_dir, "params.json"), "w") as fp:
        json.dump(train_args.toDict(), fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)