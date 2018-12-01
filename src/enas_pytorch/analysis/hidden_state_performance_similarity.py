"""Entry point."""
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

from scipy.stats import spearmanr

import collections
import os
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

from dotmap import DotMap
import json

logger = utils.get_logger()


def cosine_similarity(x, y):
    numerator = torch.dot(x, y)
    denominator = torch.norm(x, 2) * torch.norm(y, 2)

    return numerator / denominator


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)
    load_dotenv(find_dotenv(), override=True)

    hidden_state_analysis_dir = os.environ.get("HIDDEN_STATE_SIMILARITY_PERFORMANCE_DIR")
    model_dir = os.path.basename(args.model_dir)
    save_dir = os.path.join(ROOT_DIR, hidden_state_analysis_dir, model_dir)

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
    elif train_args.train_type == 'random': # Does not work for random yet since random has no controller.
        trnr = random_trainer.RandomTrainer(train_args, dataset)
    elif train_args.train_type == "hardcoded":
        trnr = hardcoded_trainer.HardcodedTrainer(train_args, dataset)
    elif train_args.train_type == "flexible":
        trnr = flexible_trainer.FlexibleTrainer(train_args, dataset)

    dags, hiddens, probabilities = trnr.derive_many(100, return_hidden=True)
    cosine_similarities = {}
    l2_distances = {}
    validation_ppls = {}

    distances_list = []
    ppl_diff_list = []
    cosines_list = []

    for i in range(len(dags)):
        validation_ppl_i = trnr.get_perplexity_multibatch(trnr.eval_data, dags[i])

        for j in range(i + 1, len(dags)):
            if i != j:
                validation_ppl_j = trnr.get_perplexity_multibatch(trnr.eval_data, dags[j])
                cosine_sim = cosine_similarity(hiddens[i], hiddens[j]).item()
                cosine_similarities["{}_{}".format(i, j)] = cosine_sim
                l2_distance = torch.norm(hiddens[i] - hiddens[j], 2).item()
                l2_distances["{}_{}".format(i, j)] = l2_distance
                validation_ppls["{}_{}".format(i, j)] = "{}_{}".format(validation_ppl_i, validation_ppl_j)

                distances_list.append(l2_distance)
                ppl_diff_list.append(abs(validation_ppl_i - validation_ppl_j))
                cosines_list.append(cosine_sim)

    distance_performance_correlation = spearmanr(distances_list, ppl_diff_list)
    cosine_performance_correlation = spearmanr(cosines_list, ppl_diff_list)

    results = {"cosine_similarities": cosine_similarities,
               "l2_distances": l2_distances,
               "validation_ppls": validation_ppls,
               "distance_performance_correlation": distance_performance_correlation,
               "cosine_performance_correlation": cosine_performance_correlation}

    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

    with open(os.path.join(save_dir, "params.json"), "w") as fp:
        json.dump(train_args.toDict(), fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)