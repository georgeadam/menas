from configs.config_orig import parser, data_arg, learn_arg, net_arg, misc_arg, logger
from configs.config_orig import add_argument_group, str2bool, get_args_helper

# Network

# Data
data_arg.add_argument('--dataset', type=str, default='ptb')


# Training / test parameters
learn_arg.add_argument('--mode', type=str, default='train',
                       choices=['train', 'derive', 'test'],
                       help='train: Training ENAS, derive: Deriving Architectures')
learn_arg.add_argument('--test_batch_size', type=int, default=1)
learn_arg.add_argument('--max_epoch', type=int, default=int(1500))  # TODO: I changed this

# Shared parameters
learn_arg.add_argument('--shared_max_step', type=int, default=400,
                       help='step for shared parameters') # TODO: CHANGED FROM 400 to 1
# NOTE(brendan): Should be 10 for CNN architectures.
learn_arg.add_argument('--shared_optim', type=str, default='adam')
learn_arg.add_argument('--shared_lr', type=float, default=0.00035)

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
misc_arg.add_argument('--random_seed', type=int, default=555)
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True)
misc_arg.add_argument("--train_type", type=str, default="hardcoded")

hardcoded_arg = add_argument_group('Hardcoded')
hardcoded_arg.add_argument("--architecture", type=str, default='tree')


def get_args():
    return get_args_helper(parser, logger)