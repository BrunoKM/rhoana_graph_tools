import numpy as np
import networkx as nx
import itertools
import h5py
import time

from graph_utils import rand_permute_adj_matrix, is_isomorphic_from_adj, graph_edit_distance_from_adj
from sympy.utilities.iterables import multiset_permutations


def generate_ged(save_path, num_nodes, total_examples, total_graphs, ged_function, split=(0.7, 0.1, 0.2), directed=False):
    """
    Generate the graph edit distance (ged) dataset.
    """
    num_train_graphs, num_val_graphs, num_test_graphs = map(lambda x: int(x * total_graphs), split)
    examples_per_set = list(map(lambda x: int(x * total_examples), split))

    graph_list = generate_graph_class_list(num_nodes, total_graphs, directed, dtype=np.int8)

    train_graphs = graph_list[:num_train_graphs]
    val_graphs = graph_list[num_train_graphs:num_train_graphs + num_val_graphs]
    test_graphs = graph_list[num_train_graphs + num_val_graphs:]
    graphs_sets = [train_graphs, val_graphs, test_graphs]

    hdf5_file = h5py.File(save_path, mode='w')
    set_names = ['train', 'val', 'test']

    for set_name, graphs, num_examples in zip(set_names, graphs_sets, examples_per_set):

        hdf5_file.create_dataset(set_name + '_graph1', [num_examples, num_nodes, num_nodes], np.int8)
        hdf5_file.create_dataset(set_name + '_graph2', [num_examples, num_nodes, num_nodes], np.int8)
        hdf5_file.create_dataset(set_name + '_labels', [num_examples], np.float64)

        num_graphs = len(graphs)
        for i in range(num_examples):
            start_time = time.time()

            # Pick two graphs at random
            selection = np.random.choice(num_graphs, 2)
            graph1, graph2 = graphs[selection[0]], graphs[selection[1]]
            distance = ged_function(graph1, graph2)

            # Save the values to the hdf5 file
            hdf5_file[set_name + '_graph1'][i, ...] = graph1
            hdf5_file[set_name + '_graph2'][i, ...] = graph2
            hdf5_file[set_name + '_labels'][i] = distance

            print(f"\rSet: {set_name}. GED examples generated: {i} / {num_examples}."
                  f"Time for example {time.time() - start_time}", end='')
    print('Dataset generated')
    hdf5_file.close()
    return


def generate_graph_class_list(num_nodes, num_graphs, directed=False, dtype=np.float64):
    """
    Generate a list of graphs where each graph is unique (no two graphs in the list are isomorphic)
    :param num_nodes: int
    :param num_graphs: Length of the list to be generated (num graphs in the list)
    :param directed: bool - whether graphs should be directed
    :param dtype: dtype of the adjacency matrix array
    :return: list of square numpy arrays representing the graphs
    """
    # Ensure the requested number of graphs is possible
    max_unique = {2: 2,
                  3: 4,
                  4: 11,
                  5: 34,
                  6: 156,
                  7: 1044,
                  8: 12346,
                  9: 274668,
                  }
    if num_nodes < 10:
        assert num_graphs < max_unique[num_nodes]

    # Create a dictionary with degree distribution as key to reduce the number of isomorphic tests (heuristic)
    graph_dict = {}

    i = 0
    while i < num_graphs:
        rand_graph = random_graph(num_nodes, directed=directed, dtype=dtype)
        degree_dist = np.sum(rand_graph, axis=1, keepdims=False)
        degree_dist.sort()
        degree_dist = tuple(degree_dist)  # Needs to be a tuple as np arrays not hashable

        # Check whether an automorphism has already been generated
        if degree_dist not in graph_dict.keys():
            graph_dict[degree_dist] = [rand_graph]
            i += 1
        else:
            automorphism_in_dict = False
            for graph in graph_dict[degree_dist]:
                if is_isomorphic_from_adj(rand_graph, graph):
                    automorphism_in_dict = True
                    break
            if not automorphism_in_dict:
                graph_dict[degree_dist].append(rand_graph)
                i += 1
        print(f"\rGenerated unique graphs: {i} / {num_graphs}", end='')

    print("\nUnique graphs generated.\n")
    # Construct the graph list from the dictionary
    graph_list = []
    for deg_graph_list in graph_dict.values():
        graph_list += deg_graph_list
    assert len(graph_list) == num_graphs
    return graph_list


def random_graph(num_nodes, directed=False, dtype=np.float64):
    adj_mat = np.random.randint(0, 2, size=[num_nodes] * 2, dtype=dtype)
    # Ensure the diagonal is zero
    adj_mat -= adj_mat * np.eye(num_nodes, dtype=adj_mat.dtype)
    if not directed:
        adj_mat[np.tril_indices(num_nodes)] = adj_mat.T[np.tril_indices(num_nodes)]
    return adj_mat


def generate_batch(batch_size, num_nodes, directed=False):
    # todo: rewrite and encapsulate pieces (using random graph for instance)
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

if __name__ == '__main__':
    ged_dataset_path = '../datasets/ged_dataset.h5'

    num_nodes = 8
    num_examples = 50000
    num_graphs = 2000

    generate_ged(ged_dataset_path, num_nodes, num_examples, num_graphs, graph_edit_distance_from_adj)
