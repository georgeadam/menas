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
import utils as utils

from network_construction.utils import Node

from scipy.stats import spearmanr, pearsonr

import collections
import os
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

from dotmap import DotMap
import json

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

    perplexity_correlation_dir = os.environ.get("PERPLEXITY_CORRELATION_DIR")
    model_dir = os.path.basename(args.model_dir)
    save_dir = os.path.join(ROOT_DIR, perplexity_correlation_dir, model_dir)

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

    if train_args.train_type == "enas" or train_args.train_type == "ours" or train_args.train_type == "orig":
        trnr = regular_trainer.Trainer(train_args, dataset)
    elif train_args.trian_type == 'random': # Does not work for random yet since random has no controller.
        trnr = random_trainer.RandomTrainer(train_args, dataset)
    elif train_args.train_type == "hardcoded":
        trnr = hardcoded_trainer.HardcodedTrainer(train_args, dataset)

    results = {"spearmanr": {}, "pearsonr": {}}
    dags, hiddens = trnr.derive_many(10)
    eval_ppls = []
    derive_ppls = []
    full_ppls = []

    for i, dag in enumerate(dags):
        print(i)
        hidden = trnr.shared.init_hidden(trnr.args.batch_size)
        eval_ppl = trnr.evaluate(trnr.eval_data, dag, "val best", max_num=trnr.args.batch_size, tb=False)
        rewards, hidden, derive_ppl = trnr.get_reward(dag, torch.tensor(1.0), hidden, 0)
        full_ppl = trnr.get_perplexity_multibatch(trnr.eval_data, dag)

        eval_ppls.append(eval_ppl)
        derive_ppls.append(derive_ppl)
        full_ppls.append(full_ppl)

        print(eval_ppls)
        print(derive_ppls)
        print(full_ppls)

    for type in [spearmanr, pearsonr]:
        results[type.__name__]["eval_derive"] = type(eval_ppls, derive_ppls)
        results[type.__name__]["eval_full"] = type(eval_ppls, full_ppls)
        results[type.__name__]["derive_full"] = type(derive_ppls, full_ppls)

    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

    with open(os.path.join(save_dir, "params.json"), "w") as fp:
        json.dump(train_args.toDict(), fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)