"""Entry point."""
import torch

#import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__)))

from data.image import Image
from data.text import Corpus

import config as config
import trainer as trainer
import utils as utils

logger = utils.get_logger()


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True

    if args.network_type == 'rnn':
        dataset = Corpus(args.data_path)
    elif args.network_type == 'cnn':
        dataset = Image(args)
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
    # srun --gres=gpu:1 -c 2 -l -w dgx1 -p gpuc python main.py --network_type rnn --dataset ptb --controller_optim adam --controller_lr 0.00035 --shared_optim sgd --shared_lr 20.0 --entropy_coeff 0.0001 --num_gpu 1
    # python main.py --network_type rnn --dataset ptb --controller_optim adam --controller_lr 0.00035 --shared_optim sgd --shared_lr 20.0 --entropy_coeff 0.0001 --num_gpu 0
