import collections

Node = collections.namedtuple('Node', ['id', 'name'])


def generate_chain(num_blocks, func_names):
    dag = collections.defaultdict(list)

    dag[-1] = [Node(0, func_names[0])]
    dag[-2] = [Node(0, func_names[0])]

    for idx in range(0, num_blocks):
        dag[idx] = [Node(idx + 1, func_names[0])]

    leaf_nodes = set(range(num_blocks)) - dag.keys()

    for idx in leaf_nodes:
        dag[idx] = [Node(num_blocks, 'avg')]

    last_node = Node(num_blocks + 1, 'h[t]')
    dag[num_blocks] = [last_node]

    return dag