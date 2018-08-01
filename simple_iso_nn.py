import collections
import math
import os
import sys
import argparse
import random
import itertools
from sympy.utilities.iterables import multiset_permutations
import networkx as nx
import numpy as np
import tensorflow as tf

from graph_utils import rand_permute_adj_matrix, is_isomorphic_from_adj


random_seed = 0
np.random.seed(random_seed)

aggregate_log_dir = "./log"


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


def initialise_log_dir(aggregate_log_dir='./log'):
    """Initialises a new directory for the run and returns its path"""
    dirs = os.listdir(aggregate_log_dir)
    previous_runs = list(filter(lambda d: d.startswith('run_'), dirs))
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

    log_dir = f"run_{run_number}"
    return os.path.join(aggregate_log_dir, log_dir)


def generate_batch(batch_size, num_nodes, directed=False):
    # todo: rewrite and encapsulate pieces
    assert batch_size % 4 == 0

    batch_input_1 = np.ndarray(shape=(batch_size, num_nodes, num_nodes), dtype=np.int32)
    batch_input_2 = np.ndarray(shape=(batch_size, num_nodes, num_nodes), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

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


def train():
    # Build the graph
    directed = False
    num_nodes = 2
    batch_size = 4
    learning_rate = 0.001

    graph = tf.Graph()

    with graph.as_default():
        #     # Ops and variables pinned to the CPU because of missing GPU implementation
        #     with tf.device('/cpu:0'):

        # Input data.
        with tf.name_scope('inputs'):
            train_input_1 = tf.placeholder(tf.float32, shape=(batch_size, num_nodes, num_nodes))
            train_input_2 = tf.placeholder(tf.float32, shape=(batch_size, num_nodes, num_nodes))
            train_labels = tf.placeholder(tf.int32, shape=(batch_size, 1))
            train_input_1_flat = tf.reshape(train_input_1, shape=(batch_size, num_nodes ** 2))
            train_input_2_flat = tf.reshape(train_input_2, shape=(batch_size, num_nodes ** 2))
            #         valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        train_input_1_flat = tf.Print(train_input_1_flat, [train_input_1_flat, train_input_2_flat, train_labels],
                                      "hmm\n", first_n=20, summarize=40)

        # The embedding network
        # Both networks share the weights
        with tf.name_scope('joint_embedding'):
            hidden_size = 30
            embedding_size = 10
            with tf.variable_scope('embed'):
                fc_1 = tf.layers.dense(train_input_1_flat, hidden_size, activation=tf.nn.relu, trainable=True,
                                       name='fc_1')
                fc_2 = tf.layers.dense(fc_1, hidden_size, activation=tf.nn.relu, trainable=True, name='fc_2')
                embed_1 = tf.layers.dense(fc_2, embedding_size, activation=tf.nn.tanh, trainable=True,
                                          name='fc_3')  # Try activation function here?

            with tf.variable_scope('embed', reuse=True):
                fc_1 = tf.layers.dense(train_input_2_flat, hidden_size, activation=tf.nn.relu, trainable=True,
                                       name='fc_1', reuse=True)
                fc_2 = tf.layers.dense(fc_1, hidden_size, activation=tf.nn.relu, trainable=True, name='fc_2')
                embed_2 = tf.layers.dense(fc_2, embedding_size, activation=tf.nn.tanh, trainable=True, name='fc_3',
                                          reuse=True)

        with tf.name_scope('discriminate'):
            combined_embedding = tf.concat([embed_1, embed_2], 1, name='concat')  # Check whether axis is right
            logits = tf.layers.dense(combined_embedding, 1, activation=None, trainable=True, name='fc_3')

            # Compute the loss
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(train_labels, logits))

        with tf.name_scope('accuracy'):
            prediction = tf.cast(tf.greater(logits, 0), tf.int32, name='prediction')
            accuracy = tf.reduce_mean(tf.cast(tf.equal(train_labels, prediction), tf.float32))
            print(loss, "\n", type(accuracy))

        # Add the loss and accuracy values as a scalar to summary.
        tf.summary.scalar('training_accuracy', accuracy)
        tf.summary.scalar('loss', loss)

        # Construct the SGD optimizer using a learning rate
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            #     norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            #     normalized_embeddings = embeddings / norm
            #     valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
            #                                               valid_dataset)
            #     similarity = tf.matmul(
            #         valid_embeddings, normalized_embeddings, transpose_b=True)

        # Merge all summaries.
        merged = tf.summary.merge_all()

        # Add variable initializer.
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

        # Create a saver.
        saver = tf.train.Saver()

    # Start training
    num_steps = 10000
    avg_loss_interval = 1000

    # Initialise new logging directory for each run
    log_dir = initialise_log_dir()

    with tf.Session(graph=graph) as session:
        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(log_dir, session.graph)

        # We must initialize all variables before we use them.
        session.run(init)
        print('Initialised')

        average_loss = 0
        for step in range(num_steps):
            batch_input_1, batch_input_2, batch_labels = generate_batch(batch_size, num_nodes, directed=directed)
            feed_dict = {train_input_1: batch_input_1,
                         train_input_2: batch_input_2,
                         train_labels: batch_labels}

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
            # Feed metadata variable to session for visualizing the graph in TensorBoard.
            _, summary, loss_val = session.run(
                [optimizer, merged, loss],
                feed_dict=feed_dict,
                run_metadata=run_metadata)
            average_loss += loss_val

            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
            # Add metadata to visualize the graph for the last run.
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % avg_loss_interval == 0:
                if step > 0:
                    average_loss /= avg_loss_interval
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                #         if step % 10000 == 0:
                #             sim = similarity.eval()
                #             for i in xrange(valid_size):
                #                 valid_word = reverse_dictionary[valid_examples[i]]
                #                 top_k = 8  # number of nearest neighbors
                #                 nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                #                 log_str = 'Nearest to %s:' % valid_word
                #                 for k in xrange(top_k):
                #                     close_word = reverse_dictionary[nearest[k]]
                #                     log_str = '%s %s,' % (log_str, close_word)
                #                 print(log_str)
                #     final_embeddings = normalized_embeddings.eval()

                # Write corresponding labels for the embeddings.
                #     with open(log_dir + '/metadata.tsv', 'w') as f:
                #         for i in range(vocabulary_size):
                #             f.write(reverse_dictionary[i] + '\n')

        # Save the model for checkpoints.
        saver.save(session, os.path.join(log_dir, 'model.ckpt'))

    # # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    #     config = projector.ProjectorConfig()
    #     embedding_conf = config.embeddings.add()
    #     embedding_conf.tensor_name = embeddings.name
    #     embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    #     projector.visualize_embeddings(writer, config)

    writer.close()
    return




