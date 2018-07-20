import networkx as nx
import numpy as np
import graph_utils


def random_with_min_motif_freq(motif, min_freq, num_nodes, connectivity=None, shuffle_on_connecting=True):
    """
    Generate a random directed graph with specified minimum frequency of occurrences of a certain motif.
    Edges are currently unweighted
    :param motif:
    :param min_freq:
    :param num_nodes:
    :param connectivity:
    :param shuffle_on_connecting:
    :return:
    """
    motif_size = motif.shape[0]
    # Assert min_freq is below the maximum minimum frequency this algorithm is applicable for
    max_freq_allowed = num_nodes - motif_size + 1
    if min_freq > max_freq_allowed:
        raise AttributeError(f'min_frequency {min_freq} too large. Maximum allowed for number of nodes {num_nodes}'
                             f' is {num_nodes - motif_size + 1}.')

    adj_matrix = np.zeros([num_nodes, num_nodes])
    # Create a mask to mark where the added motifs are in the overall network
    mask = np.zeros(adj_matrix.shape, dtype=np.bool)

    # Depending on min frequency, motif size and the number of nodes, overlap between motifs might be necessary.
    # Randomise the overlap size by randomly selecting the indices for the addition of motifs
    # on the overall adjacency matrix
    steps = get_random_steps(min_freq, num_nodes, motif_size)

    i = 0  # This keeps the index around the diagonal of where to input the next array
    adj_matrix[0:motif_size, 0:motif_size] += motif
    mask[0:motif_size, 0:motif_size] = 1

    # Keep the motif matrix as ordered last time.
    last_motif_mat = motif
    for step in steps:
        i += step
        if step >= motif_size:
            # There is no overlap, just insert the matrix
            adj_matrix[i:i + motif_size, i:i + motif_size] = motif
        else:
            # A subgraph of size (motif_size - step) is overlapping
            overlap = motif_size - step
            # Rearrange the motif adjacency matrix so that it 'lines up' with the overlapping part
            ordering = np.arange(0, motif_size)
            ordering = np.roll(ordering, overlap)
            motif = graph_utils.rearrange_adj_matrix(last_motif_mat, ordering)

            assert np.all(adj_matrix[i:i+overlap, i:i+overlap] == motif[0:overlap, 0:overlap])  # todo: remove after testing
            # Add the motif to the overall adjacency matrix
            adj_matrix[i:i + motif_size, i:i + motif_size] = motif

        # Alter the mask to show where the motif is located
        mask[i:i + motif_size, i:i + motif_size] = 1

        if shuffle_on_connecting:
            # Randomly shuffle the part of the generated graph adj. matrix where the motif was added
            new_ordering = np.arange(0, num_nodes)
            np.random.shuffle(new_ordering[i:i + motif_size])
            adj_matrix = graph_utils.rearrange_adj_matrix(adj_matrix, new_ordering)
            # Alter the mask accordingly as well to reflect the reordering
            mask = graph_utils.rearrange_adj_matrix(mask, new_ordering)
            # Store the new adj. matrix of the last added motif (as it has been rearranged)
            last_motif_mat = adj_matrix[i:i + motif_size, i:i + motif_size]
        else:
            last_motif_mat = motif
    # Fill in the values not filled in by the motifs with random connections with connectivity p
    if connectivity is None:
        # If connectivity not specified, use the average connectivity of the parts filled in with the motifs
        p = np.sum(adj_matrix * mask) / np.sum(mask)
    else:
        # Otherwise use the value given
        p = connectivity
    rand_connection_fill = np.random.binomial(1, p, size=adj_matrix.shape)
    # Zero-out the entries that coincide with the motifs assigned earlier
    mask_reverse = np.ones(adj_matrix.shape, dtype=np.int32) - mask
    rand_connection_fill *= mask_reverse
    # Finally combine the motif entries and the randomly generated connections
    adj_matrix += rand_connection_fill
    return adj_matrix


def get_random_steps(min_freq, num_nodes, motif_size):
    """
    Get random indices for random_with_min_motif_freq. The result should be a random array of size min_freq that
    holds the ordered random indices for diagonal positions where to add motifs. Each element must be smaller than
    or equal to (num_nodes - motif_size). Also, no two elements can be the same
    """
    num_steps = min_freq - 1
    steps_base = np.ones([num_steps], np.int32)

    # Generate random addition to steps base to make the sum equal to num_nodes - motif_size
    steps_add = np.random.uniform(0, 1, size=[num_steps])
    # Make sure the sum is equal to num_nodes - sum(steps_base) - motif_size
    steps_add *= (num_nodes - num_steps - motif_size) / np.sum(steps_add)
    # Round the numbers to the closest integer
    steps_add = steps_add.round().astype(np.int32)
    if np.sum(steps_add) != num_nodes - num_steps - motif_size:
        # If the there were any values with .50000 in decimal places, the rounding could have changed the sum
        # Run the function again to get new steps_add
        return get_random_steps(min_freq, num_nodes, motif_size)
    return steps_base + steps_add


def random_with_second_order(num_nodes, p, a_recip, a_conv, a_div, a_chain):
    """
    Generate a random graph with chosen second order statistics (second order motifs) as described in the
    "Synchornization from second order network connectivity" (Zhao et al, 2011) paper.
    :param num_nodes: Total number of nodes (vertices)
    :param p: average connectivity (number expected edges / total number of possible edges)
    Specify how probability of the reciprocal/convergent/divergent/chain motifs deviate from independence
    (positive => more of the motif then in Erdos-Renyi, negative => less):
    :param a_recip: statistical deviation of reciprocal motif
    :param a_conv: statistical deviation of convergent motif
    :param a_div: statistical deviation of divergent motif
    :param a_chain: statistical deviation of chain motif
    :return:
    """
    # todo
    pass


if __name__ == '__main__':
    # todo: remove
    motif = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 0]])
    g = random_with_min_motif_freq(motif, 3, 10)
    print(g)
    motif = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]])
    g = random_with_min_motif_freq(motif, 23, 30)
    print(g)