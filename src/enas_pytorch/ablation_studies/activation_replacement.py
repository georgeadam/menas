"""Entry point."""
import torch

from data.image import Image
from data.text import Corpus

from configs import config_ablation as config
from train_scripts import regular_trainer
from train_scripts import random_trainer
from train_scripts import hardcoded_trainer
import utils as utils

from models.controller import Node

import collections
import os
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

import json

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
    load_dotenv(find_dotenv(), override=True)

    activation_ablation_dir = os.environ.get("ACTIVATION_ABLATION_DIR")
    model_dir = os.path.basename(args.model_dir)
    save_dir = os.path.join(ROOT_DIR, activation_ablation_dir, model_dir)

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

    if args.train_type == "enas":
        trnr = regular_trainer.Trainer(args, dataset)
    elif args.trian_type == 'random':
        trnr = random_trainer.RandomTrainer(args, dataset)
    elif args.train_type == "hardcoded":
        trnr = hardcoded_trainer.HardcodedTrainer(args, dataset)

    dag = trnr.derive()

    ppl = trnr.get_perplexity_multibatch(trnr.eval_data, dag, 1)
    print("Original performance on entire validation set: {}".format(ppl))

    results = {"orig_performance": ppl}
    activations = ["tanh", "relu", "sigmoid", "identity"]

    for activation in activations:
        temp_dag = replace_all_activations(dag, activation)
        ppl = trnr.get_perplexity_multibatch(trnr.eval_data, temp_dag, 1)
        results[activation] = ppl

        print("Validation PPL when using all {} activations is: {}".format(activation, ppl))

    with open(os.path.join(save_dir, "params.json"), "w") as fp:
        json.dump(train_args, fp, indent=4, sort_keys=True)

    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)