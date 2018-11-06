import collections
import math

Node = collections.namedtuple('Node', ['id', 'name'])


def generate_tree(num_blocks, func_names):
    dag = collections.defaultdict(list)

    dag[-1] = [Node(0, func_names[0])]
    dag[-2] = [Node(0, func_names[0])]

    node_count = 0

    for idx in range(0, num_blocks):
        dag[idx] = [Node(idx + 1, func_names[0])]

        if node_count == num_blocks - 1:
            break

        # If root node of tree, just link to next two nodes
        if idx == 0:
            dag[idx] = []
            dag[idx].append(Node(idx + 1, func_names[0]))
            node_count += 1

            if node_count == num_blocks - 1:
                break

            dag[idx].append(Node(idx + 2, func_names[0]))

            node_count += 1
        else:
            # If internal node of tree, use the following logic. We need to consider the total number of nodes at the
            # depth of the current node, as well as the number of nodes at the same depth, but to the left of the
            # current node. The sum of these two terms if used as the offset where we will link to the next nodes in
            # the tree.
            depth = math.floor(math.log(idx + 1, 2))

            num_nodes_at_depth = math.pow(2, depth - 1)
            num_nodes_before_idx = (idx + 1) - num_nodes_at_depth

            offset = int(num_nodes_at_depth + num_nodes_before_idx)

            dag[idx] = []
            dag[idx].append(Node(idx + offset, func_names[0]))
            node_count += 1

            if node_count == num_blocks - 1:
                break

            dag[idx].append(Node(idx + offset + 1, func_names[0]))

            node_count += 1

    leaf_nodes = set(range(num_blocks)) - dag.keys()

    for idx in leaf_nodes:
        dag[idx] = [Node(num_blocks, 'avg')]

    last_node = Node(num_blocks + 1, 'h[t]')
    dag[num_blocks] = [last_node]

    return dag