from configs.helpers import str2bool


def add_arguments(net_arg, data_arg, learn_arg, misc_arg):
    data_arg.add_argument('--dataset', type=str, default='ptb')

    # Training / test parameters
    learn_arg.add_argument('--mode', type=str, default='train',
                           choices=['train', 'derive', 'test', 'train_scratch'],
                           help='train: Training ENAS, derive: Deriving Architectures')
    learn_arg.add_argument('--shared_num_sample', type=int, default=1,
                           help='# of Monte Carlo samples')
    misc_arg.add_argument('--load_path', type=str, default='')

    # Shared parameters
    learn_arg.add_argument('--shared_optim', type=str, default='adam')
    learn_arg.add_argument('--shared_lr', type=float, default=0.00035)
    learn_arg.add_argument('--shared_decay_after', type=float, default=10e8)  # TODO: DON'T DECAY

    learn_arg.add_argument('--entropy_coeff', type=float, default=1e-6)

    # Deriving Architectures
    learn_arg.add_argument('--derive_num_sample', type=int, default=100)

    # Misc
