from configs.config_orig import parser, data_arg, learn_arg, net_arg, misc_arg, logger
from configs.config_orig import add_argument_group, str2bool

from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

import os

# Network

# Controller

# Data

# Training / test parameters
learn_arg.add_argument('--mode', type=str, default='derive',
                       choices=['train', 'derive', 'test'],
                       help='train: Training ENAS, derive: Deriving Architectures')

# Deriving Architectures
learn_arg.add_argument('--derive_num_sample', type=int, default=100)


# Misc
misc_arg.add_argument('--load_path', type=str, default='ptb_enas_2018-11-06_23-18-19')
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--num-workers', type=int, default=2)
misc_arg.add_argument('--random_seed', type=int, default=12345)
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True)


def get_args():
    args, unparsed = parser.parse_known_args()

    d = vars(args)

    load_dotenv(find_dotenv(), override=True)
    d["data_dir"] = os.path.join(ROOT_DIR, args.data_dir)
    d["log_dir"] = os.path.join(ROOT_DIR, args.log_dir)

    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        logger.info(f"Unparsed args: {unparsed}")

    return args, unparsed