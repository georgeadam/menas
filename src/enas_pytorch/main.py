"""Entry point."""
import os

import torch

import sys
sys.path.insert(-1, os.path.abspath("."))
sys.path.insert(-1, os.path.abspath(".."))
sys.path.insert(-1, os.path.abspath("../.."))

import src.enas_pytorch.data.image
import src.enas_pytorch.data as data
import src.enas_pytorch.config as config
import src.enas_pytorch.utils as utils
import src.enas_pytorch.trainer as trainer

logger = utils.get_logger()


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True

    if args.network_type == 'rnn':
        dataset = data.text.Corpus(args.data_path)
    elif args.network_type == 'cnn':
        dataset = data.image.Image(args)
    else:
        raise NotImplementedError(f"{args.dataset} is not supported")

    trnr = trainer.Trainer(args, dataset)

    if args.mode == 'train':
        utils.save_args(args)
        trnr.train()
    elif args.mode == 'derive':
        assert args.load_path != "", ("`--load_path` should be given in "
                                      "`derive` mode")
        trnr.derive()
    else:
        if not args.load_path:
            raise Exception("[!] You should specify `load_path` to load a "
                            "pretrained model")
        trnr.test()


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)
    # python main.py --network_type rnn --dataset ptb --controller_optim adam --controller_lr 0.00035 --shared_optim sgd --shared_lr 20.0 --entropy_coeff 0.0001 --num_gpu 0
