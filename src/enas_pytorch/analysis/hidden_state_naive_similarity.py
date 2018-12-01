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

    hidden_state_analysis_dir = os.environ.get("HIDDEN_STATE_SIMILARITY_NAIVE_DIR")
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
    common = {}
    cosine_similarities = {}
    l2_distances = {}

    connections_list = []
    activations_list = []
    distances_list = []
    cosines_list = []

    for i in range(len(dags)):
        for j in range(i + 1, len(dags)):
            if i != j:
                common_activations, common_connections = count_common_attributes(dags[i], dags[j])
                common["{}_{}".format(i, j)] = {"activations": common_activations, "connections": common_connections}
                cosine_sim = cosine_similarity(hiddens[i], hiddens[j]).item()
                cosine_similarities["{}_{}".format(i, j)] = cosine_sim
                l2_distance = torch.norm(hiddens[i] - hiddens[j], 2).item()
                l2_distances["{}_{}".format(i, j)] = l2_distance

                connections_list.append(common_connections)
                activations_list.append(common_activations)
                distances_list.append(l2_distance)
                cosines_list.append(cosine_sim)

    connection_cor_l2 = spearmanr(connections_list, distances_list)
    activation_cor_l2 = spearmanr(activations_list, distances_list)

    connection_cor_cosine = spearmanr(connections_list, cosines_list)
    activation_cor_cosine = spearmanr(activations_list, cosines_list)

    results = {"connections_l2_distance_spearman": connection_cor_l2,
               "activations_l2_distance_spearman": activation_cor_l2, "common_attributes": common,
               "cosine_similarities": cosine_similarities,
               "l2_distances": l2_distances, "connections_cosine_spearman": connection_cor_cosine,
               "activations_cosine_spearman": activation_cor_cosine}

    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

    with open(os.path.join(save_dir, "params.json"), "w") as fp:
        json.dump(train_args.toDict(), fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)