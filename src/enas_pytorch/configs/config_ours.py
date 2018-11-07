from configs.config_orig import parser, data_arg, learn_arg, net_arg, misc_arg, logger
from configs.config_orig import add_argument_group, str2bool, get_args_helper

# Network

# Controller

# Data
data_arg.add_argument('--dataset', type=str, default='ptb')

# Training / test parameters
learn_arg.add_argument('--mode', type=str, default='train',
                       choices=['train', 'derive', 'test'],
                       help='train: Training ENAS, derive: Deriving Architectures')
learn_arg.add_argument('--max_epoch', type=int, default=int(150000))  # TODO: I changed this

# Controller
learn_arg.add_argument('--controller_max_step', type=int, default=60,
                       help='step for controller parameters')  # TODO: Changed from 2000 to 1

# Shared parameters
# TODO: Changed from 0 to 399
learn_arg.add_argument('--shared_max_step', type=int, default=20,
                       help='step for shared parameters') # TODO: CHANGED FROM 400 to 1
# NOTE(brendan): Should be 10 for CNN architectures.
learn_arg.add_argument('--shared_num_sample', type=int, default=1,
                       help='# of Monte Carlo samples')
learn_arg.add_argument('--shared_lr', type=float, default=10.0)
learn_arg.add_argument('--shared_decay_after', type=float, default=int(15 * (2000 / 50)))  # TODO: THIS SHOULD CHANGE, SINCE I CHANGED NUMBER OF ITERATIONS

# Deriving Architectures
learn_arg.add_argument('--derive_num_sample', type=int, default=100)


# Misc
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=20)
misc_arg.add_argument('--save_epoch', type=int, default=10)
misc_arg.add_argument('--max_save_num', type=int, default=4)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--num-workers', type=int, default=2)
misc_arg.add_argument('--random_seed', type=int, default=12345)
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True)
misc_arg.add_argument("--train_type", type=str, default="enas")


def get_args():
    return get_args_helper(parser, logger)