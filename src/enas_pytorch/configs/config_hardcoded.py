from configs.helpers import str2bool


def add_arguments(net_arg, data_arg, misc_arg, learn_arg, parser):
    # Network

    # Data
    data_arg.add_argument('--dataset', type=str, default='ptb')

    # Training / test parameters
    learn_arg.add_argument('--mode', type=str, default='train',
                           choices=['train', 'derive', 'test', 'train_scratch'],
                           help='train: Training ENAS, derive: Deriving Architectures')
    learn_arg.add_argument('--max_epoch', type=int, default=int(1500))  # TODO: I changed this

    # Shared parameters
    learn_arg.add_argument('--shared_max_step', type=int, default=400,
                           help='step for shared parameters')  # TODO: CHANGED FROM 400 to 1
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

    hardcoded_arg = parser.add_argument_group('Hardcoded')
    hardcoded_arg.add_argument("--architecture", type=str, default='tree')