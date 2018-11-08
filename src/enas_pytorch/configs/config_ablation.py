from configs.config_orig import parser, data_arg, learn_arg, net_arg, misc_arg, logger
from configs.config_orig import add_argument_group, str2bool, get_args_helper

# Network

# Controller

# Data
data_arg.add_argument('--dataset', type=str, default='ptb')

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
misc_arg.add_argument("--train_type", type=str, default="enas",
                      choices=['enas', 'random', 'hardcoded'])


def get_args():
    return get_args_helper(parser, logger)