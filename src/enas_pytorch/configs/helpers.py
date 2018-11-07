def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(parser, name):
    arg = parser.add_argument_group(name)
    return arg
