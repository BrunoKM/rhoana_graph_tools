import numpy as np
import itertools


def rearrange_adj_matrix(matrix, ordering):
    assert matrix.ndim == 2
    # Check that matrix is square
    assert matrix.shape[0] == matrix.shape[1]
    num_nodes = matrix.shape[0]
    assert len(ordering) == num_nodes

    # Swap rows into correct ordering
    matrix = matrix[ordering, :]
    # Swap columns into correct ordering
    matrix = matrix[:, ordering]
    return matrix


def rand_permute_adj_matrix(matrix):
    """Randomly permute the order of vertices in the adjacency matrix, while maintaining the connectivity
    between them."""
    num_vertices = matrix.shape[0]
    rand_order = np.arange(num_vertices)
    np.random.shuffle(rand_order)
    matrix_permuted = rearrange_adj_matrix(matrix, rand_order)
    return matrix_permuted


def adj_matrix_to_edge_list(adj_matrix, directed=True, first_id=0, weighted=False):
    num_nodes = adj_matrix.shape[0]

    if directed:
        num_edges = np.sum(adj_matrix)
    else:
        num_edges = int(np.sum(adj_matrix) / 2)
    if weighted:
        edge_list = np.zeros([num_edges, 3], dtype=np.int32)
    else:
        edge_list = np.zeros([num_edges, 2], dtype=np.int32)

    i = 0
    for node_in in range(num_nodes):
        if directed:
            range_2 = range(num_nodes)
        else:
            range_2 = range(node_in + 1, num_nodes)
        for node_out in range_2:
            edge_val = adj_matrix[node_in, node_out]
            if edge_val > 0:
                # If there is a connection
                if weighted:
                    edge_list[i] = (node_in + first_id, node_out + first_id, edge_val)
                else:
                    edge_list[i] = (node_in + first_id, node_out + first_id)
                i += 1

    return edge_list


def edge_list_to_textfile(edge_list, filepath, weighted=False):
    with open(filepath, 'w') as file:
        if weighted:
            for i, j, weight in edge_list:
                file.write(f"{i} {j} {weight}\n")
        else:
            for i, j in edge_list:
                file.write(f"{i} {j}\n")
    return
