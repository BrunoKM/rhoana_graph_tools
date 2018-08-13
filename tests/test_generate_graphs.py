from context import *
from graph_generation import *


def test_random_with_min_motif_freq():
    motif = motif = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 0]])
    generated_graph = random_with_min_motif_freq(motif, 4, 10)
    # Check that all the diagonal entries are 0
    assert np.sum(np.eye(generated_graph.shape[0]) * generated_graph) == 0


def test_get_neighbouring_idxs():
    """
    Test the _get_neighbouring_idxs function against some known values.
    """
    try:
        # Case 1
        idxs = get_neighbouring_idxs(0, 0, 0, num_divs=3)
        expected_idxs = [(0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        assert len(idxs) == 8
        for idx in expected_idxs:
            assert idx in idxs
        # Case 2 (middle)
        x, y, z = (2, 2, 2)
        idxs = get_neighbouring_idxs(x, y, z, num_divs=5)
        expected_idxs = [(x, y, z), (x + 1, y, z), (x - 1, y + 1, z),  (x, y + 1, z), (x + 1, y + 1, z),
                         (x, y - 1, z + 1), (x - 1, y - 1, z + 1), (x + 1, y - 1, z + 1), (x - 1, y, z + 1),
                         (x, y, z + 1), (x + 1, y, z + 1), (x - 1, y + 1, z + 1), (x, y + 1, z + 1),
                         (x + 1, y + 1, z + 1)]
        assert len(idxs) == 14
        for idx in expected_idxs:
            assert idx in idxs
        # Case 3 (end)
        idxs = get_neighbouring_idxs(3, 3, 3, num_divs=4)
        expected_idxs = [(3, 3, 3)]
        assert len(idxs) == 1
        assert idxs[0] == expected_idxs[0]

        # Case 4
        idxs = get_neighbouring_idxs(4, 4, 3, num_divs=5)
        expected_idxs = [(4, 4, 3), (3, 3, 4), (4, 3, 4), (3, 4, 4), (4, 4, 4)]
        assert len(idxs) == 5
        for idx in expected_idxs:
            assert idx in idxs
    except AssertionError as e:
        exception_msg = f"The expected idxs were {expected_idxs} \n(Num entries: {len(expected_idxs)}\n" \
                        f"Actual idxs were {idxs} \n Num entries: {len(idxs)}"
        raise Exception(exception_msg) from e
    return


def test_calc_distance_between_points():
    # Check that all the values returned are positive and within expected bounds (for example, for points with each
    # parameter within [0, 1), the maximum distance is sqrt(3) in 3d.
    shape = [100, 3]
    allowed_err = 0.0001
    try:
        coords = np.random.uniform(0, 1, size=shape)
        distances = calc_distance_between_points(coords, coords)
        assert distances.shape[0] == shape[0]
        assert distances.shape[1] == shape[0]
        assert np.all(distances >= 0)
        assert np.all(distances < np.sqrt(3))
        # Make sure the matrix is symmetric
        assert np.all(distances.T == distances)
        # Check all the distances with a for loop:
        for i, j in itertools.product(range(shape[0]), repeat=2):
            difference_vec = coords[i] - coords[j]
            distance = np.sqrt(np.sum(difference_vec**2))
            assert abs(distance - distances[i, j]) < allowed_err
        # Calculate some expected distances:
        coords = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0]], dtype=np.float64)
        expected_distances = np.array([[0, 1, 1],
                                       [1, 0, np.sqrt(2)],
                                       [1, np.sqrt(2), 0]], dtype=np.float64)
        distances = calc_distance_between_points(coords, coords)
        err = np.abs(distances - expected_distances)
        assert np.all(err < allowed_err)
    except AssertionError as e:
        raise Exception(f"The shape of the distances matrix is: {distances.shape}\n"
                        f"The first 3x3 elements are:\n{distances[:3, :3]}") from e


if __name__ == '__main__':
    test_get_neighbouring_idxs()
