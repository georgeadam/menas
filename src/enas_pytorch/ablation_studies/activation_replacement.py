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
import os
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

from dotmap import DotMap
import json

logger = utils.get_logger()


def replace_all_activations(dag, replacement="tanh"):
    new_dag = collections.defaultdict(list)

    for idx, nodes in dag.items():
        if idx < 0:
            continue

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
    train_args = DotMap(train_args)
    original_mode = train_args.mode
    train_args.mode = "derive"
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

    results = {"validation": {"original_performance": validation_ppl}, "test": {"original_performance": test_ppl}}
    activations = ["tanh", "relu", "sigmoid", "identity"]

    for activation in activations:
        temp_dag = replace_all_activations(dag, activation)
        validation_ppl = trnr.get_perplexity_multibatch(trnr.eval_data, temp_dag)
        test_ppl = trnr.get_perplexity_multibatch(trnr.test_data, temp_dag)

        results["validation"][activation] = validation_ppl
        results["test"][activation] = test_ppl

        print("Validation PPL when using all {} activations is: {}".format(activation, validation_ppl))
        print("Test PPL when using all {} activations is: {}".format(activation, test_ppl))

    train_args.mode = original_mode

    with open(os.path.join(save_dir, "params.json"), "w") as fp:
        json.dump(train_args.toDict(), fp, indent=4, sort_keys=True)

    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)