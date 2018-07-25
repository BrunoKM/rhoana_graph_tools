import networkx as nx
import numpy as np
import graph_utils
import utils
import itertools
import math
import time


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
    # Randomly allocate the motifs within the adjacency matrix, keeping track of where they have been allocated in mask
    adj_matrix, mask = rand_allocate_motifs(motif, min_freq, num_nodes, shuffle_on_connecting, dtype=np.int8)

    # Fill in the values not filled in by the motifs with random connections with connectivity p
    if connectivity is None:
        # If connectivity not specified, use the average connectivity of the parts filled in with the motifs
        p = np.sum(adj_matrix * mask) / np.sum(mask)
    else:
        # Otherwise use the value given
        p = connectivity
    rand_connection_fill = np.random.binomial(1, p, size=adj_matrix.shape)
    # Zero-out the entries that coincide with the motifs assigned earlier
    mask_inverse = np.logical_not(mask)
    # Zero-out the entries on the diagonal (no self-loops)
    mask_inverse = np.logical_and(mask_inverse, np.logical_not(np.eye(num_nodes, dtype=np.bool)))

    rand_connection_fill *= mask_inverse
    # Finally combine the motif entries and the randomly generated connections
    adj_matrix += rand_connection_fill
    return adj_matrix


def random_weighted_with_min_motif_freq(motif, min_freq, num_nodes, motif_edge_std=0, rand_edge_mean=1,
                                        rand_edge_std=0, connectivity=None, shuffle_on_connecting=True, discrete=False):
    """
    Generate a random weighted directed graph with specified minimum frequency of occurrences of a certain motif.
    :param motif:
    :param min_freq:
    :param num_nodes:
    :param connectivity:
    :param shuffle_on_connecting:
    :return:
    """
    # Randomly allocate the motifs within the adjacency matrix, keeping track of where they have been allocated in mask
    adj_matrix, mask = rand_allocate_motifs(motif, min_freq, num_nodes, shuffle_on_connecting, dtype=np.float32)

    # Generate the deviation for all edges
    # Generate only once for both motif edges and the rest of random edges to improve efficiency (as these two cases
    # are not overlapping)
    edge_var = np.random.normal(loc=0.0, scale=1.0, size=adj_matrix.shape)

    # Add variation to the motif edge weights
    if motif_edge_std != 0:
        motif_edge_var = edge_var * motif_edge_std
        # Want to add deviation only to existing edges todo: consider adding possibility of altering non-existing edges
        edge_var_mask = (adj_matrix != 0)
        adj_matrix += motif_edge_var * edge_var_mask

    # Fill in the values not filled in by the motifs with random connections with connectivity p
    if connectivity is None:
        # If connectivity not specified, use the average connectivity of the parts filled in with the motifs
        if discrete:
            # If the values will be rounded, the connections (or edges) are values larger or equal to 0.5
            num_edges = np.sum(adj_matrix >= 0.5)
        else:
            # If the values will be rounded, any positive value is a connection
            num_edges = np.sum(adj_matrix > 0)
        p = np.sum(num_edges) / np.sum(mask)
    else:
        # Otherwise use the value given
        p = connectivity
    rand_connection_fill = np.random.binomial(1, p, size=adj_matrix.shape)
    # Zero-out the entries that coincide with the motifs assigned earlier
    mask_inverse = np.logical_not(mask)
    # Zero-out the entries on the diagonal (no self-loops)
    mask_inverse = np.logical_and(mask_inverse, np.logical_not(np.eye(num_nodes, dtype=np.bool)))
    rand_connection_fill *= mask_inverse

    if rand_edge_std != 0:
        # Get the random weights for the connected edges
        rand_connection_weights = (edge_var * rand_edge_std) + rand_edge_mean
        rand_connection_weights *= rand_connection_fill
    else:
        rand_connection_weights = rand_connection_fill * rand_edge_mean

    adj_matrix += rand_connection_weights

    if discrete:
        adj_matrix = np.round(adj_matrix)

    # Finally, rectify the values so that all weights take on non-negative values
    adj_matrix[adj_matrix < 0] = 0
    return adj_matrix


def rand_allocate_motifs(motif, min_freq, num_nodes, shuffle_on_connecting, dtype=np.float64):
    """
    Handles the random allocation of motifs for the random_with_min_motif_freq and
    random_weighted_with_min_motif_freq function.
    :param dtype: Specifies the dtype of the adjacency matrix. For a normal directed graph bool or int will do. For
    a graph with weighted edges with weights taking on real values, or if motifs are to follow a random distribution,
    use float.
    :return: tuple of (the adjacency matrix with assigned motifs, the edge positions where motifs have been assigned
    """
    motif_size = motif.shape[0]
    # Assert min_freq is below the maximum minimum frequency this algorithm is applicable for
    max_freq_allowed = num_nodes - motif_size + 1
    if min_freq > max_freq_allowed:
        raise AttributeError(f'min_frequency {min_freq} too large. Maximum allowed for number of nodes {num_nodes}'
                             f' is {num_nodes - motif_size + 1}.')

    adj_matrix = np.zeros([num_nodes, num_nodes], dtype=dtype)
    # Create a mask to mark where the added motifs are in the overall network
    mask = np.zeros(adj_matrix.shape, dtype=np.bool)

    # Depending on min frequency, motif size and the number of nodes, overlap between motifs might be necessary.
    # Randomise the overlap size by randomly selecting the indices for the addition of motifs
    # on the overall adjacency matrix
    indices = get_random_indices(min_freq - 1, num_nodes, motif_size)

    adj_matrix[0:motif_size, 0:motif_size] += motif
    mask[0:motif_size, 0:motif_size] = 1
    last_idx = 0

    # Store the motif matrix as it was ordered last time.
    last_motif_mat = motif
    for i in indices:
        step = i - last_idx
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
        last_idx = i
    return adj_matrix, mask


def random_preferential_by_dist(num_nodes, conn_prob_function, function_diameter=1, adj_matrix_dtype=np.int8):
    """
    Generate a random graph where the connections have higher probabilities when the nodes are close. The nodes are
    randomly distributed in 3d space and distance between each node has to be computed (which is the computationally
    expensive process.
    :param num_nodes:
    :param conn_prob_function:
    :param function_diameter:
    :param adj_matrix_dtype:
    :return:
    """
    coord = np.random.uniform(size=[num_nodes, 3])

    # Get the number of divisions along each axis
    if function_diameter >= 1:
        num_divs = 1
    else:
        num_divs = int(math.ceil(1 / function_diameter))

    divs, coord_idxs = _get_point_divisions(coord, num_divs, function_diameter)

    # Calculate distances between points in neighbouring divisions
    distance_matrix = np.zeros([num_nodes] * 2, dtype=coord.dtype)  # Will store all the distances between nodes
    for x1, y1, z1 in itertools.product(range(num_divs), repeat=3):
        div1_idx_start, div1_idx_stop, div1_coords = divs[x1, y1, z1]
        for x2, y2, z2 in get_neighbouring_idxs(x1, y1, z1, num_divs):
            div2_idx_start, div2_idx_stop, div2_coords = divs[x2, y2, z2]
            distances = calc_distance_between_points(div1_coords, div2_coords)
            # Add the distances to the adjacency matrix (only have to fill in the upper triangular part as the
            # distance matrix is symmetric)
            assert div1_idx_start <= div2_idx_start
            distance_matrix[div1_idx_start:div1_idx_stop, div2_idx_start:div2_idx_stop] = distances
    # Make sure the distance matrix is symmetric by making it upper triangular and adding it's transpose to itself
    distance_matrix = np.triu(distance_matrix) + np.tril(distance_matrix.T)

    # Now that the distance_matrix contains all the distances, apply the connection probability function element-wise
    conn_prob_matrix = conn_prob_function(distance_matrix)

    # Finally, determine whether there is a connection by sampling values from from the uniform distribution.
    # If the value sampled is smaller than probability of connection, let there be a connection, and if it's larger
    # there won't be a connection. This way a connection will be generated with probability equal to that
    # in the conn_prob_matrix.
    adj_matrix = np.random.uniform(0, 1, size=[num_nodes, num_nodes]) < conn_prob_matrix
    adj_matrix = adj_matrix.astype(dtype=adj_matrix_dtype)

    # Lastly remove the self-connections (the diagonal has to be zero) by multiplying with za matrix of ones with
    # zeros along the diagonal.
    adj_matrix *= np.ones(adj_matrix.shape, dtype=adj_matrix_dtype) - np.eye(num_nodes, dtype=adj_matrix_dtype)

    return adj_matrix, coord, coord_idxs, distance_matrix


def _get_point_divisions(coord, num_divs, function_diameter):
    """
    Separate the points in space with coordinates specified in coord into neighbouring cubic subspaces (a grid).
    Each cube in the grid can be indexed into in array divs which stores the coordinates and ids of the points within
    each division (grid-cube).

    The ids work as follows:
    Since we do not care about the index of each particular point (node) in the generated adjacency matrix as long as
    each point has a unique index, we can just assign index while dividing the coordinates into divisions. For each
    division, we store the range of indices for that division [m, n) and assume that the coordinates within that
    division have incrementing indices (m, m + 1, m + 2, ...). We only have to store to values m and n for each
    division, and this allows for easy indexing into the adjacency matrix later.
    :param coord: np.ndarray of shape (num_nodes, 3) with 3d coordinates of the points. The point have to be confined
    to the space 0 <= x, y, z < 1
    :param function_diameter: the length of each division
    :return: a tuple with two elements:
    1. a 3d cubic tensor (np.ndarray) of shape [num_divs, num_divs, num_divs]. Each element stores a tuple of
    start_idx, stop_idx and coordinates of the points within the corresponding grid division; start_idx and stop_idx
    are integers such that all the indices of the coordinates for that particular division lie within the range
    idx_start <= index of any coordinate within division < idx_stop
    2. the array of indices for the coordinates of the points.
    """
    next_idx_available = 0  # Stores the next smallest index that can be assigned to the coordinates

    # Initialise an empty array to store the arrays of points within each division
    divs = np.empty([num_divs] * 3, dtype=tuple)

    coord_idxs = np.zeros(shape=coord.shape[0], dtype=np.int32)
    # Divide the point within the space into the divisions
    for z, y, x in itertools.product(range(num_divs), repeat=3):
        coord_in_range = _get_coord_in_range(coord, x * function_diameter, y * function_diameter,
                                             z * function_diameter, function_diameter)
        # Get the number of points in this division
        num_in_div = np.sum(coord_in_range)
        # Store the tuple containing indexes of the points in this division in the first entry
        # and their coordinates in the second entry
        idx_start = next_idx_available
        idx_stop = next_idx_available + num_in_div
        div = (idx_start, idx_stop, coord[coord_in_range, :])
        divs[x, y, z] = div

        # Save the assigned indices of the coordinates
        coord_idxs[coord_in_range] = np.arange(idx_start, idx_stop, dtype=np.int32)

        # Increment next_idx_available so that no indices overlap
        next_idx_available += num_in_div
    return divs, coord_idxs


def _get_coord_in_range(coord, min_x, min_y, min_z, interval_length):
    coord_in_range_x = np.logical_and(coord[:, 0] >= min_x, coord[:, 0] < min_x + interval_length)
    coord_in_range_y = np.logical_and(coord[:, 1] >= min_y, coord[:, 1] < min_y + interval_length)
    coord_in_range_z = np.logical_and(coord[:, 2] >= min_z, coord[:, 2] < min_z + interval_length)
    coord_in_range = np.logical_and(np.logical_and(coord_in_range_x, coord_in_range_y), coord_in_range_z)
    return coord_in_range


def get_neighbouring_idxs(x, y, z, num_divs):
    """
    Use the following 'elegant' manually combined list of indices in order to consider all the divisions
    neighbouring to (x1, y1, z1) that we haven't considered before.
    """
    if min([x, y, z]) < 0 or max([x, y, z]) >= num_divs:
        raise AttributeError(f"Current division index out of bounds: {(x, y, z)}")

    # Computer the limits for each index (so that the generated indices don't
    # step out of bounds (i.e. (x, y, z) < num_divs)
    x_max = min(x + 2, num_divs)  # x indices have to be smaller than this limit
    y_max = min(y + 2, num_divs)
    x_min = max(x - 1, 0)  # x indices have to be smaller than this limit
    y_min = max(y - 1, 0)
    neighbour_idxs = [(x, y, z)]
    # First consider all indices in range (x - 1, x + 1) (y - 1, y + 1) in the layer above in the z direction
    if z + 1 < num_divs:
        neighbour_idxs += list(itertools.product(range(x_min, x_max), range(y_min, y_max), [z + 1]))
    # Consider all indices in range (x - 1, x + 1) at y + 1 in the current z layer (z)
    if y + 1 < num_divs:
        neighbour_idxs += list(itertools.product(range(x_min, x_max), [y + 1], [z]))
    # Consider the next index in the current column (y, z).
    if x + 1 < num_divs:
        neighbour_idxs.append((x + 1, y, z))
    # Consider the pairwise distances between points in the division itself
    return neighbour_idxs


def calc_distance_between_points(coord1, coord2):
    """
    Calculate the Euclidean distance between all points in coord1 and coord2 and store them in a matrix. Works for
    any space with number of dimensions >= 2.
    :param coord1: np.ndarray with shape [num_points1, num_dimensions]
    :param coord2: np.ndarray with shape [num_points2, num_dimensions]
    :return: np.ndarray matrix with shape [num_points1, num_points2] with entries in position [i, j] being the
    Euclidean distance between i-th point of coord1 and j-th point of coord2
    """
    # Calculate the displacement vectors between each pair of points
    disp_vecs= np.expand_dims(coord1, axis=1) - np.expand_dims(coord2, axis=0)
    distance = np.sqrt(np.sum(np.square(disp_vecs), axis=-1))  # Compute the Euclidean distance: sqrt(x**2, y**2, ..)
    return distance


def random_gaussian_preferential_by_dist(num_nodes, gauss_std, gauss_height, max_error=0.0, const=0.0,
                                         adj_matrix_dtype=np.int8):
    """
    Generate a random graph where the connections have higher probabilities when the nodes are close. The nodes are
    randomly distributed in 3d space and distance between each node has to be computed (which is the computationally
    expensive process. The process can be made faster for larger network by truncating the Gaussian filter and only
    considering distances to the nodes that lie close.
    :return:
    """
    # Separate the two cases to optimise the runtime if const == 0 (which is the expected way of using this function)
    if const != 0.0:
        conn_prob_function = lambda x: utils.gaussian_function(x, std=gauss_std, peak_height=gauss_height) + const
    else:
        conn_prob_function = lambda x: utils.gaussian_function(x, std=gauss_std, peak_height=gauss_height)

    if max_error == 0:
        return random_preferential_by_dist(num_nodes, conn_prob_function, function_diameter=1,
                                           adj_matrix_dtype=adj_matrix_dtype)
    else:
        function_radius = utils.inverse_gaussian(max_error, std=gauss_std, peak_height=gauss_height)
        if function_radius is np.nan:
            function_diameter = 1.0
        else:
            function_diameter = 2 * function_radius
        return random_preferential_by_dist(num_nodes, conn_prob_function, function_diameter=function_diameter,
                                           adj_matrix_dtype=adj_matrix_dtype)


def random_preferential_grid_pos_2d():
    """
    Generate a random graph where the connections have higher probabilities when the nodes are close. Assumes the nodes
    are uniformly spaced on a 2d grid. Elegant solution using a 2d Gaussian kernel.
    :return:
    """
    # todo: complete
    pass


def random_preferential_grid_pos_3d():
    """
    Generate a random graph where the connections have higher probabilities when the nodes are close. Assumes the nodes
    are uniformly spaced on a 3d grid.
    :return:
    """
    # todo: complete
    pass


def get_random_indices(num_indices, num_nodes, motif_size):
    """
    Get random indices for random_with_min_motif_freq. The result should be a random array of size num_indices that
    holds the ordered random indices for diagonal positions where to add motifs. Each element must be smaller than
    or equal to (num_nodes - motif_size) and larger than 0. Also, no two elements can be the same
    """
    max_index = num_nodes - motif_size
    # Sample without replacement from range (1, max_index + 1)
    indices = np.random.choice(np.arange(1, max_index + 1), size=num_indices, replace=False)
    # Return the sorted indices
    return np.sort(indices)


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
    # motif = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 0]])
    # g = random_with_min_motif_freq(motif, 3, 10)
    # g2 = random_weighted_with_min_motif_freq(motif, 3, 10, shuffle_on_connecting=True, motif_edge_std=1.2,
    #                                          rand_edge_mean=30, rand_edge_std=1.4, discrete=False)
    # print(np.sum(np.eye(g.shape[0]) * g), np.sum(np.eye(g2.shape[0]) * g2))
    # print(g2)
    # motif = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]])
    # g = random_with_min_motif_freq(motif, 23, 30)
    # print(g)
    # for size in [1000, 3000, 5000, 8000, 10000]:
    #     start_time = time.time()
    #     g = random_weighted_with_min_motif_freq(motif, int(size/2), size, shuffle_on_connecting=False, discrete=True)
    #     print(f"Size: {size}, Time taken: {time.time() - start_time:.2f}s")
    #     del g
    g = random_gaussian_preferential_by_dist(10000, 0.4, 1)
    print(g)
