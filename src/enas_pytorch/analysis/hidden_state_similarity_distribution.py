"""Entry point."""
import torch

from data.image import Image
from data.text import Corpus

from configs import config_ablation as config
from train_scripts import regular_trainer
from train_scripts import random_trainer
from train_scripts import hardcoded_trainer
from train_scripts import flexible_trainer
from train_scripts import supervised_trainer
import utils as utils

import os
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

from dotmap import DotMap
import json

from visualization.density_plots import density_plot
from visualization.line_plots import line_plot
import numpy as np

logger = utils.get_logger()


def cosine_similarity(x, y):
    numerator = torch.dot(x, y)
    denominator = torch.norm(x, 2) * torch.norm(y, 2)

    return numerator / denominator


def get_top_networks(distances, dags, n, direction="closest"):
    temp_keys, temp_values = distances.keys(), distances.values()

    keys, values = [], []

    # Keep only unique values of distance
    for key, value in zip(temp_keys, temp_values):
        if value not in values:
            keys.append(key)
            values.append(value)

    # Negate distances if we want to get the DAGs that are farthest apart rather than closest together
    if direction == "closest":
        values = np.array(values)
    else:
        values = - np.array(values)

    sort_idx = np.argsort(values)

    top_distances = []
    top_networks = []

    for i in range(n):
        top_distances.append(abs(values[sort_idx[i]]))
        top_networks.append(keys[sort_idx[i]])

    top_dags = []

    for i in range(len(top_networks)):
        left = int(top_networks[i].split("_")[0])
        right = int(top_networks[i].split("_")[1])

        top_dags.append((dags[left], dags[right]))

    return top_distances, top_dags


def plot_top_networks(top_distances, top_networks, save_dir, direction="closest"):
    for i in range(len(top_distances)):
        left = top_networks[i][0]
        right = top_networks[i][1]

        fname = "{}_{}_{}.png".format(direction, top_distances[i], "left")
        path = os.path.join(save_dir, fname)
        utils.draw_network(left, path)

        fname = "{}_{}_{}.png".format(direction, top_distances[i], "right")
        path = os.path.join(save_dir, fname)
        utils.draw_network(right, path)


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)
    load_dotenv(find_dotenv(), override=True)

    hidden_state_analysis_dir = os.environ.get("HIDDEN_STATE_SIMILARITY_PLOTS_DIR")
    model_dir = os.path.basename(args.model_dir)
    save_dir = os.path.join(ROOT_DIR, hidden_state_analysis_dir, model_dir)
    distance_file_path = os.path.join(save_dir, "distance_density.png")

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
    elif train_args.train_type == "supervised_regularized":
        trnr = supervised_trainer.SupervisedTrainer(train_args, dataset)

    dags, hiddens, probabilities = trnr.derive_many(500, return_hidden=True)
    cosine_similarities = {}
    l2_distances = {}

    distances_list = []
    cosines_list = []

    for i in range(len(dags)):
        print(i)
        for j in range(i + 1, len(dags)):
            if i != j:
                cosine_sim = cosine_similarity(hiddens[i], hiddens[j]).item()
                cosine_similarities["{}_{}".format(i, j)] = cosine_sim
                l2_distance = torch.norm(hiddens[i] - hiddens[j], 2).item()
                l2_distances["{}_{}".format(i, j)] = l2_distance

                distances_list.append(l2_distance)
                cosines_list.append(cosine_sim)

    density_plot(distances_list, train_args.train_type.capitalize(), "L2 Distances", distance_file_path)

    networks_dir = os.path.join(save_dir, "networks")
    utils.makedirs(networks_dir)
    closest_top_distances, closest_top_networks = get_top_networks(l2_distances, dags, 5, "closest")
    plot_top_networks(closest_top_distances, closest_top_networks, networks_dir, "closest")

    farthest_top_distances, farthest_top_networks = get_top_networks(l2_distances, dags, 5, "farthest")
    plot_top_networks(farthest_top_distances, farthest_top_networks, networks_dir, "farthest")

    with open(os.path.join(save_dir, "params.json"), "w") as fp:
        json.dump(train_args.toDict(), fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)