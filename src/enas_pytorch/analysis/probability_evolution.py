"""Entry point."""
import torch

from data.image import Image
from data.text import Corpus

from configs import config_ablation as config
from train_scripts import regular_trainer
from train_scripts import random_trainer
from train_scripts import hardcoded_trainer
from train_scripts import flexible_trainer
from train_scripts import prev_action_regularized_trainer
from train_scripts import performance_regularized_trainer
from train_scripts import biased_regularized_trainer
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


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)
    load_dotenv(find_dotenv(), override=True)

    hidden_state_analysis_dir = os.environ.get("HIDDEN_STATE_SIMILARITY_PLOTS_DIR")
    model_dir = os.path.basename(args.model_dir)
    save_dir = os.path.join(ROOT_DIR, hidden_state_analysis_dir, model_dir)
    probability_density_file_path = os.path.join(save_dir, "probability_density.png")
    probability_lineplot_file_path = os.path.join(save_dir, "probability_lineplot.png")
    probability_density_file_random_state_path = os.path.join(save_dir, "probability_density_random_state.png")
    probability_lineplot_file_random_state_path = os.path.join(save_dir, "probability_lineplot_random_state.png")


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
    elif train_args.train_type == "action_regularized":
        trnr = prev_action_regularized_trainer.PrevActionRegularizedTrainer(train_args, dataset)
    elif train_args.train_type == "performance_regularized":
        trnr = performance_regularized_trainer.PerformanceRegularizedTrainer(train_args, dataset)
    elif train_args.train_type == "biased_regularized":
        trnr = biased_regularized_trainer.BiasedRegularizedTrainer(train_args, dataset)
    elif train_args.train_type == "supervised_regularized":
        trnr = supervised_trainer.SupervisedTrainer(train_args, dataset)

    dags, hiddens, probabilities = trnr.controller.sample(100, with_details=False, return_hidden=True,
                                                          random_hidden_state=False)

    density_plot(probabilities.view(-1).cpu(), train_args.train_type.capitalize(), "Probabilities",
                 probability_density_file_path, label_stats=False)
    line_plot(torch.arange(probabilities.shape[1]).repeat(probabilities.shape[0], 1), probabilities.cpu(),
              train_args.train_type.capitalize(), "Time Step", "Probability", probability_lineplot_file_path)

    dags, hiddens, probabilities = trnr.controller.sample(100, with_details=False, return_hidden=True,
                                                          random_hidden_state=True)

    density_plot(probabilities.view(-1).cpu(), train_args.train_type.capitalize(), "Probabilities",
                 probability_density_file_random_state_path, label_stats=False)
    line_plot(torch.arange(probabilities.shape[1]).repeat(probabilities.shape[0], 1), probabilities.cpu(),
              train_args.train_type.capitalize(), "Time Step", "Probability",
              probability_lineplot_file_random_state_path)

    with open(os.path.join(save_dir, "params.json"), "w") as fp:
        json.dump(train_args.toDict(), fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)