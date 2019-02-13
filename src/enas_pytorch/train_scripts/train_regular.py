"""Entry point."""
import torch

import sys, os
sys.path.insert(0, os.path.realpath(__file__)[:-len('/train_scripts/train_regular.py')])
# TODO: This a dumb solution to adding the parent directories parent to the path.  I want to do this without setting an environment variable for super easy deployment.

from data.image import Image
from data.text import Corpus

from configs import config_orig as config
from train_scripts import regular_trainer, hardcoded_trainer, random_trainer, flexible_trainer, preset_trainer, \
    prev_action_regularized_trainer, performance_regularized_trainer, biased_regularized_trainer
import utils as utils

logger = utils.get_logger()

# python train_scripts/train_regular.py --mode test --load_path ptb_enas_2018-11-06_13-08-37
#@utils.slurmify
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

    if args.train_type == 'ours' or args.train_type == 'orig' or args.train_type == 'tf':
        trnr = regular_trainer.Trainer(args, dataset)
    elif args.train_type == 'random':
        trnr = random_trainer.RandomTrainer(args, dataset)
    elif args.train_type == 'hardcoded':
        trnr = hardcoded_trainer.HardcodedTrainer(args, dataset)
    elif args.train_type == 'flexible':
        trnr = flexible_trainer.FlexibleTrainer(args, dataset)
    elif args.train_type == 'preset':
        trnr = preset_trainer.PresetTrainer(args, dataset)
    elif args.train_type == 'action_regularized':
        trnr = prev_action_regularized_trainer.PrevActionRegularizedTrainer(args, dataset)
    elif args.train_type == 'performance_regularized':
        trnr = performance_regularized_trainer.PerformanceRegularizedTrainer(args, dataset)
    elif args.train_type == 'biased_regularized':
        trnr = biased_regularized_trainer.BiasedRegularizedTrainer(args, dataset)

    if args.mode == 'train':
        utils.save_args(args)
        trnr.train()
    elif args.mode == 'derive':
        assert args.load_path != "", ("`--load_path` should be given in "
                                      "`derive` mode")
        trnr.derive()
    elif args.mode == "test":
        if not args.load_path:
            raise Exception("[!] You should specify `load_path` to load a "
                            "pretrained model")
        trnr.test()
    elif args.mode == "train_scratch":
        assert args.load_path != "", ("--load_path should be given in derive mode")

        utils.save_args(args, "scratch_params")
        trnr.train_scratch()


if __name__ == "__main__":
    args, unparsed = config.get_args()
    if args.mode == 'test':
        # This could be invoked with:
        # python train_scripts/train_regular.py --mode test --load_path ptb_enas_2018-11-07_21-56-24
        while True:  # Spin in a loop graphin the test result for tensorboard.
            args, unparsed = config.get_args()
            main(args)
    main(args)
    # srun --gres=gpu:1 -c 2 -l -w dgx1 -p gpuc python train_regular.py --mode test --load_path ptb_2018-10-30_20-42-11 --num_gpu 1
    # srun --gres=gpu:1 -c 2 -l -w dgx1 -p gpuc python train_scripts/train_regular.py --network_type rnn --dataset ptb --controller_optim adam --controller_lr 0.00035 --shared_optim adam --shared_lr 0.00035 --entropy_coeff 0.0001 --num_gpu 1
    # python train_regular.py --network_type rnn --dataset ptb --controller_optim adam --controller_lr 0.00035 --shared_optim adam --shared_lr 0.00035 --entropy_coeff 0.0001 --num_gpu 0
