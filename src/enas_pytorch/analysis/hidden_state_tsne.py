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
from sklearn import manifold

from visualization.scatterplots import scatterplot

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

    hidden_state_plots_dir = os.environ.get("HIDDEN_STATE_SIMILARITY_PLOTS_DIR")
    model_dir = os.path.basename(args.model_dir)
    save_dir = os.path.join(ROOT_DIR, hidden_state_plots_dir, model_dir)
    save_path = os.path.join(save_dir, "performance_tsne.png")

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

    dags, hiddens, probabilities = trnr.derive_many(1000, return_hidden=True)
    hiddens = hiddens.data.cpu().numpy()
    validation_ppls = []

    for i in range(len(dags)):
        validation_ppl = trnr.get_perplexity_multibatch(trnr.eval_data, dags[i])
        validation_ppls.append(validation_ppl)

    tsne5 = manifold.TSNE(perplexity=5, random_state=0)
    tsne10 = manifold.TSNE(perplexity=10, random_state=0)
    tsne20 = manifold.TSNE(perplexity=20, random_state=0)
    tsne30 = manifold.TSNE(perplexity=30,  random_state=0)

    y5 = tsne5.fit_transform(hiddens)
    y10 = tsne10.fit_transform(hiddens)
    y20 = tsne20.fit_transform(hiddens)
    y30 = tsne30.fit_transform(hiddens)


    Ys = [y5, y10, y20, y30]
    colors = [validation_ppls for i in range(len(Ys))]
    titles = ["Perplexity: 5", "Perplexity: 10", "Perplexity: 20", "Perplexity30"]

    scatterplot(Ys, titles, colors, save_path)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)