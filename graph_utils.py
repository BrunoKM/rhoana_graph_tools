import numpy as np


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
