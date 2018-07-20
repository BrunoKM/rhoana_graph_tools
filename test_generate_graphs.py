from generate_graphs import *


def test_random_with_min_motif_freq():
    motif = motif = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 0]])
    generated_graph = random_with_min_motif_freq(motif, 4, 10)
    # Check that all the diagonal entries are 0
    assert np.sum(np.eye(generated_graph.shape[0]) * generated_graph) == 0
