"""This script looks at the correlation between performance on the first batch of size [35, 64] of the validation
set, and the validation performance on the entire validation set of size [len(validation), 1]. The point is to see if
the way that the supposed best DAG is evaluated in derive() makes sense in terms of the big picture.
"""
import torch

from data.image import Image
from data.text import Corpus

from configs import config_ablation as config
from train_scripts import regular_trainer
from train_scripts import random_trainer
from train_scripts import hardcoded_trainer
from train_scripts import flexible_trainer
import utils as utils

from network_construction.utils import Node

from scipy.stats import spearmanr, pearsonr

import collections
import os
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

from dotmap import DotMap
import json

import time

logger = utils.get_logger()


def count_common_attributes(dag1, dag2):
    common_activations = 0
    common_connections = 0

    for idx in dag1.keys():
        nodes1 = dag1[idx]
        nodes2 = dag2[idx]

        for node1 in nodes1:
            for node2 in nodes2:
                if node1.id == node2.id:
                    common_connections += 1

                if node1.id == node2.id and node1.name == node2.name:
                    common_activations += 1

    return common_activations, common_connections


def cosine_similarity(x, y):
    numerator = torch.dot(x, y)
    denominator = torch.norm(x, 2) * torch.norm(y, 2)

    return numerator / denominator


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)
    load_dotenv(find_dotenv(), override=True)

    perplexity_correlation_dir = os.environ.get("PERPLEXITY_SPEED_DIR")
    model_dir = os.path.basename(args.model_dir)
    save_dir = os.path.join(ROOT_DIR, perplexity_correlation_dir, model_dir)

    train_args = utils.load_args(args.model_dir)
    train_args = DotMap(train_args)
    original_mode = train_args.mode
    train_args.mode = "derive"
    train_args.load_path = args.load_path
    train_args.test_batch_size = 1
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
    elif train_args.train_type == 'random': # Does not work for random yet since random has no controller.
        trnr = random_trainer.RandomTrainer(train_args, dataset)
    elif train_args.train_type == "hardcoded":
        trnr = hardcoded_trainer.HardcodedTrainer(train_args, dataset)
    elif train_args.train_type == "flexible":
        trnr = flexible_trainer.FlexibleTrainer(train_args, dataset)

    dags = trnr.derive_many(100)

    batched_times = []
    batched_ppls = []

    flattened_times = []
    flattened_ppls = []

    results = {"batched": {"times": batched_times, "ppls": batched_ppls}, "flattened": {"times": flattened_times,
                                                                                        "ppls": flattened_ppls}}

    for i, dag in enumerate(dags):
        batched_start = time.time()
        batched_ppl = trnr.get_perplexity_multibatch(trnr.valid_data, dag)
        batched_end = time.time()

        batched_time = batched_end - batched_start
        batched_times.append(batched_time)
        batched_ppls.append(batched_ppl)

        flattened_start = time.time()
        flattened_ppl = trnr.get_perplexity_multibatch(trnr.eval_data, dag)
        flattened_end = time.time()

        flattened_time = flattened_end - flattened_start
        flattened_times.append(flattened_time)
        flattened_ppls.append(flattened_ppl)

        print("Batched PPL: {} took {} to compute".format(batched_ppl, batched_end - batched_start))
        print("Flattened PPL: {} took {} to compute".format(flattened_ppl, flattened_end - flattened_start))

    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

    with open(os.path.join(save_dir, "params.json"), "w") as fp:
        json.dump(train_args.toDict(), fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)