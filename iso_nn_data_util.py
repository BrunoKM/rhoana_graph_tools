import numpy as np
from sympy.utilities.iterables import multiset_permutations
import networkx as nx
import itertools

from graph_utils import rand_permute_adj_matrix, is_isomorphic_from_adj


def generate_automorphism_dict(num_nodes, edges_range, directed=False, dtype=np.float64):
    """
    Generate a dictionary with graphs belonging to the same isomorphism class.
    :param num_nodes: int
    :param edges_range: a generator with the integer number of edges to consider
    :param directed: bool
    :param dtype: type of the adjacency matrix array
    :return: dictionary with class ids (arranged integers) as keys and list of
    adjacency matrices as values
    """
    class_dict = dict()
    ids = 0
    for num_edges in edges_range:
        # Create a temporary class dictionary to not redundantly compare graphs
        # to those with different number of edges
        temp_class_dict = dict()

        if directed:
            num_possible_edges = num_nodes ** 2 - num_nodes  # Number of possible edges
        else:
            num_possible_edges = int((num_nodes ** 2 - num_nodes) / 2)  # Number of possible edges

        for edge_vals in multiset_permutations([1] * num_edges + [0] * (num_possible_edges - num_edges)):
            # Graph construction implementation differs depending on whether directed or undirected
            if directed:
                # Add the self-loop connection elements which are set to 0
                for i in range(0, num_nodes ** 2, num_nodes + 1):
                    edge_vals.insert(i, 0)
                graph = np.array(edge_vals, dtype=dtype).reshape([num_nodes, num_nodes])
            else:
                graph = np.zeros([num_nodes, num_nodes], dtype=dtype)
                count = 0
                for i in range(0, num_nodes):
                    # Fill in the upper triangular part of the adjacency matrix
                    graph[i, i + 1:] = edge_vals[count:count + num_nodes - i - 1]
                    count += num_nodes - i - 1
                # copy the upper triangular part into the lower triangular entries
                graph += graph.T

            # Check whether belongs to any automorphism groups in class_dict
            automorphism_in_dict = False
            for class_id in temp_class_dict.keys():
                class_representative = temp_class_dict[class_id][0]
                if is_isomorphic_from_adj(class_representative, graph):
                    automorphism_in_dict = True
                    temp_class_dict[class_id].append(graph)
                    break
            # If automorphism class not already in dictionary, add it
            if not automorphism_in_dict:
                temp_class_dict[ids] = [graph]
                ids += 1

        # Merge the temp class dictionary into existing one
        class_dict = {**class_dict, **temp_class_dict}

    return class_dict


def generate_automorphism_dataset_eval_data(num_nodes, edges_range, directed=False, dtype=np.float64):
    """
    Generate a dataset for embedding evaluation and visualisation. Returns two lists: list of unique names and
    a list of graphs.
    """
    d = generate_automorphism_dict(num_nodes, edges_range, directed=directed, dtype=dtype)
    names = []
    graphs = []
    for class_id, graph_list in d.items():
        i = 0
        for graph in graph_list:
            name = f"class{class_id}_graph{i}"
            names.append(name)
            graphs.append(graph)
            i += 1
    return names, graphs


def generate_batch(batch_size, num_nodes, directed=False):
    # todo: rewrite and encapsulate pieces
    assert batch_size % 4 == 0

    batch_input_1 = np.ndarray(shape=(batch_size, num_nodes, num_nodes), dtype=np.float32)
    batch_input_2 = np.ndarray(shape=(batch_size, num_nodes, num_nodes), dtype=np.float32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.float32)

    # Generate two random graphs
    adj_mat_1 = np.random.randint(0, 2, size=[num_nodes] * 2)
    adj_mat_2 = np.random.randint(0, 2, size=[num_nodes] * 2)
    # Ensure the diagonal is zero
    adj_mat_1 -= adj_mat_1 * np.eye(num_nodes, dtype=adj_mat_1.dtype)
    adj_mat_2 -= adj_mat_2 * np.eye(num_nodes, dtype=adj_mat_2.dtype)
    if not directed:
        adj_mat_1[np.tril_indices(num_nodes)] = adj_mat_1.T[np.tril_indices(num_nodes)]
        adj_mat_2[np.tril_indices(num_nodes)] = adj_mat_2.T[np.tril_indices(num_nodes)]

    # Check whether the two graphs are isomorphic
    if is_isomorphic_from_adj(adj_mat_1, adj_mat_2):
        # If isomorphic, use the following sophisticated method to make them not isomorphic
        idx_to_change = [np.random.randint(0, num_nodes), np.random.randint(0, num_nodes - 1)]
        idx_to_change[1] += 1 if idx_to_change[1] == idx_to_change[0] else 0  # Ensures index is off diagonal
        idx_to_change = tuple(idx_to_change)
        adj_mat_2[idx_to_change] = not adj_mat_2[idx_to_change]
        # If undirected, symmetry has to be maintained
        if not directed:
            idx_complement = (idx_to_change[1], idx_to_change[0])
            adj_mat_2[idx_complement] = not adj_mat_2[idx_complement]

    adj_mats = (adj_mat_1, adj_mat_2)
    # Generate some random permutations of the input graphs
    i = 0
    # For each batch make an equal number of example with [g1, g1], [g1, g2], [g2, g1], [g2, g2]
    for j, k in itertools.product(range(2), repeat=2):
        for _ in range(batch_size // 4):
            batch_input_1[i, :, :] = rand_permute_adj_matrix(adj_mats[j])
            batch_input_2[i, :, :] = rand_permute_adj_matrix(adj_mats[k])
            # Set label to 1 if graphs isomorphic, to 0 otherwise
            labels[i] = 1 if j == k else 0
            i += 1

    return batch_input_1, batch_input_2, labels


def generate_example(num_nodes, directed=False, only_negative=True):
    label = np.ndarray(shape=(1,), dtype=np.float32)

    # Generate two random graphs
    adj_mat_1 = np.random.randint(0, 2, size=[num_nodes] * 2)
    adj_mat_2 = np.random.randint(0, 2, size=[num_nodes] * 2)
    # Ensure the diagonal is zero
    adj_mat_1 -= adj_mat_1 * np.eye(num_nodes, dtype=adj_mat_1.dtype)
    adj_mat_2 -= adj_mat_2 * np.eye(num_nodes, dtype=adj_mat_2.dtype)
    if not directed:
        adj_mat_1[np.tril_indices(num_nodes)] = adj_mat_1.T[np.tril_indices(num_nodes)]
        adj_mat_2[np.tril_indices(num_nodes)] = adj_mat_2.T[np.tril_indices(num_nodes)]

    # Check whether the two graphs are isomorphic
    isomorphic = is_isomorphic_from_adj(adj_mat_1, adj_mat_2)
    if isomorphic and only_negative:
        # If isomorphic, use the following sophisticated method to make them not isomorphic
        idx_to_change = [np.random.randint(0, num_nodes), np.random.randint(0, num_nodes - 1)]
        idx_to_change[1] += 1 if idx_to_change[1] == idx_to_change[0] else 0  # Ensures index is off diagonal
        idx_to_change = tuple(idx_to_change)
        adj_mat_2[idx_to_change] = not adj_mat_2[idx_to_change]
        # If undirected, symmetry has to be maintained
        if not directed:
            idx_complement = (idx_to_change[1], idx_to_change[0])
            adj_mat_2[idx_complement] = not adj_mat_2[idx_complement]
        isomorphic = False

    label[0] = int(isomorphic)
    return adj_mat_1.astype(np.float32), adj_mat_2.astype(np.float32), label
