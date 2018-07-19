import networkx as nx
import numpy as np


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

