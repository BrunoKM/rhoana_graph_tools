import pandas as pd
import numpy as np
import networkx as nx
import struct
import math


def read_kasthuri_original(file_path):
    column_names = ['synapse_id', # Synapse No.
                    'psd_centroid_x',
                    'psd_centroid_y',
                    'psd_centroid_z',  # PSD centroid (microns from origin, pixel (1,1,1))
                    'psd_centroid_pixel',  # PSD centroid (pixel location column, row, section)
                    'in_cylinder_1',
                    'in_cylinder_2',
                    'in_cylinder_3',  # Synapse located within cylinder 1, 2, or 3
                    'axon_id',  # Axon No
                    'dendrite_id',  # Dendrite No
                    'axon_type',  # Axon type Excitatory = 0 Inhibitory = 1 Myelinated = 2 Unknown = -1
                    'bouton_id',  # Bouton No (Not in Cylinder 1 = -1 Not available = 0)
                    'axon_terminal',  # Axon terminal = 1 En-passant synapse = 0 (Not in cylinder 1 or 2 = -1)
                    'vesicle_count',  # Vesicle count (-1 = Not in Cylinder 1 or not available)
                    'mitochondria_count',  # Number of mitochondria in axon bouton (Not available = -1)
                    'multisynaptic_bouton',  # Multi synaptic axon bouton Yes = 1; No = 0 Unknown = -1
                    'axon_skeleton_length',  # Axon skeleton length within Cylinder 1 in micrometer (Not available = -1)
                    'spine_or_shaft',  # Spine synapse  = 1 Shaft synapse = 0
                    'dendrite_type',  # Dendrite type: Excitatory/Spiny = 0 Inhibitory/Smooth = 1
                    'psd_size',  # PSD size metric (pixels)
                    'spine_id',  # Spine ID No. (shaft synapse = 0 Unknown = -1)
                    'spine_vol',  # Spine Volume (in number of voxels, at 24x24x29 nm each) (Shaft synapse = -1
                    # not available = 0)
                    'spine_apparatus',  # NEW Spine Apparatus No = 0; Yes = 1 N/A = -1 Uncertain = -2
                    'num_synapses_on_spine',  # Nr of synapses on spine
                    ]
    # Dictionary from column name to initial index for convenience (as name cannot be used on creation of dataframe)
    column_idx_dict = {column_names[idx]: idx for idx in range(len(column_names))}

    # Dictionary to specify the desired types of each column when creating the dataframe
    column_dtypes = {column_idx_dict['synapse_id']: 'int64',
                     column_idx_dict['in_cylinder_1']: 'int8',
                     column_idx_dict['in_cylinder_2']: 'int8',
                     column_idx_dict['in_cylinder_3']: 'int8',  # Synapse located within cylinder 1, 2, or 3
                     column_idx_dict['axon_id']: 'int32',  # Axon No
                     column_idx_dict['dendrite_id']: 'int32',  # Dendrite No
                     column_idx_dict['axon_type']: 'int8',  # Axon type
                     column_idx_dict['bouton_id']: 'int32',  # Bouton No (Not in Cylinder 1 = -1 Not available = 0)
                     column_idx_dict['axon_terminal']: 'int8',  # Axon terminal = 1 En-passant synapse = 0
                     column_idx_dict['vesicle_count']: 'int32',  # Vesicle count
                     column_idx_dict['mitochondria_count']: 'int32',  # Number of mitochondria in axon bouton
                     column_idx_dict['multisynaptic_bouton']: 'int8',  # Multi synaptic axon bouton Yes = 1; No = 0
                     column_idx_dict['spine_or_shaft']: 'int8',  # Spine synapse  = 1 Shaft synapse = 0
                     column_idx_dict['dendrite_type']: 'int8',  # Dendrite type:
                     column_idx_dict['spine_apparatus']: 'int32',  # NEW Spine Apparatus
                     column_idx_dict['num_synapses_on_spine']: 'int32',  # Nr of synapses on spine
                     }

    df = pd.read_excel(file_path, header=None, skiprows=2, names=column_names, dtype=column_dtypes)

    # Drop redundant information
    df = df.drop('psd_centroid_pixel', axis=1)

    # # Replace three rows of cylinder information with one (binary into list representing in which cylinders
    # # the synapse is located)
    # in_cylinders = [[] for i in range(df.shape[0])]
    # for cylinder_id, in_this_cylinder_array in enumerate([df.in_cylinder_1, df.in_cylinder_2, df.in_cylinder_3]):
    #     for idx in np.where(in_this_cylinder_array != 0)[0]:
    #         in_cylinders[idx].append(cylinder_id + 1)
    #
    # df = df.drop(['in_cylinder_1', 'in_cylinder_2', 'in_cylinder_3'], axis=1)
    # df["in_cylinders"] = in_cylinders

    return df


def read_kasthuri_generated(filepath):
    column_names = ['synapse_id',  # Synapse No.
                    'psd_centroid_x',
                    'psd_centroid_y',
                    'psd_centroid_z',  # PSD centroid (microns from origin, pixel (1,1,1))
                    'in_cylinder_1',
                    'in_cylinder_2',
                    'in_cylinder_3',  # Synapse located within cylinder 1, 2, or 3
                    'axon_id',  # Axon No
                    'dendrite_id',  # Dendrite No
                    'axon_type',  # Axon type Excitatory = 0 Inhibitory = 1 Myelinated = 2 Unknown = -1
                    'bouton_id',  # Bouton No (Not in Cylinder 1 = -1 Not available = 0)
                    'axon_terminal',  # Axon terminal = 1 En-passant synapse = 0 (Not in cylinder 1 or 2 = -1)
                    'vesicle_count',  # Vesicle count (-1 = Not in Cylinder 1 or not available)
                    'mitochondria_count',  # Number of mitochondria in axon bouton (Not available = -1)
                    'multisynaptic_bouton',  # Multi synaptic axon bouton Yes = 1; No = 0 Unknown = -1
                    'axon_skeleton_length',  # Axon skeleton length within Cylinder 1 in micrometer (Not available = -1)
                    'spine_or_shaft',  # Spine synapse  = 1 Shaft synapse = 0
                    'dendrite_type',  # Dendrite type: Excitatory/Spiny = 0 Inhibitory/Smooth = 1
                    'psd_size',  # PSD size metric (pixels)
                    'spine_id',  # Spine ID No. (shaft synapse = 0 Unknown = -1)
                    'spine_vol',  # Spine Volume (in number of voxels, at 24x24x29 nm each) (Shaft synapse = -1
                    # not available = 0)
                    'spine_apparatus',  # NEW Spine Apparatus No = 0; Yes = 1 N/A = -1 Uncertain = -2
                    'num_synapses_on_spine',  # Nr of synapses on spine
                    'input_neuron_id',
                    'output_neuron_id',
                    ]
    column_idx_dict = {column_names[idx]: idx for idx in range(len(column_names))}
    # Dictionary to specify the desired types of each column when creating the dataframe
    column_dtypes = {column_idx_dict['synapse_id']: 'int64',
                     column_idx_dict['in_cylinder_1']: 'int8',
                     column_idx_dict['in_cylinder_2']: 'int8',
                     column_idx_dict['in_cylinder_3']: 'int8',  # Synapse located within cylinder 1, 2, or 3
                     column_idx_dict['axon_id']: 'int32',  # Axon No
                     column_idx_dict['dendrite_id']: 'int32',  # Dendrite No
                     column_idx_dict['axon_type']: 'int8',  # Axon type
                     column_idx_dict['bouton_id']: 'int32',  # Bouton No (Not in Cylinder 1 = -1 Not available = 0)
                     column_idx_dict['axon_terminal']: 'int8',  # Axon terminal = 1 En-passant synapse = 0
                     column_idx_dict['vesicle_count']: 'int32',  # Vesicle count
                     column_idx_dict['mitochondria_count']: 'int32',  # Number of mitochondria in axon bouton
                     column_idx_dict['multisynaptic_bouton']: 'int8',  # Multi synaptic axon bouton Yes = 1; No = 0
                     column_idx_dict['spine_or_shaft']: 'int8',  # Spine synapse  = 1 Shaft synapse = 0
                     column_idx_dict['dendrite_type']: 'int8',  # Dendrite type:
                     column_idx_dict['spine_apparatus']: 'int32',  # NEW Spine Apparatus
                     column_idx_dict['num_synapses_on_spine']: 'int32',  # Nr of synapses on spine
                     column_idx_dict['input_neuron_id']: 'int32',
                     column_idx_dict['output_neuron_id']: 'int32',
                     }
    df = pd.read_excel(filepath, dtype=column_dtypes)
    return df


def allocate_neurons(df, num_neurons, copy=False):
    """
    Take a dataframe of the original kasthuri dataset (no neuron information), and randomly add num_neurons neurons to
    the data while ensuring the correctness of structure, in particular:
        > Any synaptic connection with same axon/dendrite id will be connected to the same neuron on the axon/dendrite
        side
        > any dendrites or axons that belong to a particular neuron must all be of the same type: inhibitory or
        excitatory (unknown can be connected to either)

    :param df: kasthuri dataset pandas dataframe
    :param num_neurons: int
    :param copy: whether to copy the dataframe, or perform changes on the original
    :return: the modified dataframe
    """
    # Get all the unique ids for excitatory and inhibitory axons and dendrites
    inhibitory_axons = df[(df.axon_type == 1)].axon_id.values
    excitatory_axons = df[(df.axon_type == 0)].axon_id.values
    other_axons = df[np.logical_and((df.axon_type != 0), (df.axon_type != 1))].axon_id.values # myelinated or uncertain

    # For now, consider other to be excitatory (only three other axons in kasthuri anyways)
    excitatory_axons = np.concatenate((excitatory_axons, other_axons))

    inhibitory_dendrites = df[(df.dendrite_type == 1)].dendrite_id.values
    excitatory_dendrites = df[(df.dendrite_type == 0)].dendrite_id.values

    # Get all the unique ids
    unique_inh_axons = np.unique(inhibitory_axons)
    unique_exc_axons = np.unique(excitatory_axons)
    unique_inh_dendrites = np.unique(inhibitory_dendrites)
    unique_exc_dendrites = np.unique(excitatory_dendrites)

    # Check that the subsets of axon and dendrite ids are disjoint todo: remover later
    for id in unique_exc_axons:
        assert id not in unique_inh_axons
    for id in unique_exc_dendrites:
        assert id not in unique_inh_dendrites

    num_axons = unique_inh_axons.size + unique_exc_axons.size
    num_dendrites = unique_inh_dendrites.size + unique_exc_dendrites.size

    # Raise an error if number of neurons will result in a disconnected graph
    if num_neurons > num_axons + num_dendrites:
        raise ValueError(f"The number of neurons {num_neurons} exceeds the total"
                         f"number of axons and dendrites {num_axons + num_dendrites}")

    # Generate two sets of neurons (inhibitory and excitatory)
    fraction_inh_neurons = (unique_inh_dendrites.size + unique_inh_axons.size) / (num_axons + num_dendrites)
    num_inh = math.floor(num_neurons * fraction_inh_neurons)
    num_exc = num_neurons - num_inh

    exc_axon_neuron_assignment = assign_random_neurons(unique_exc_axons, num_exc)
    exc_dendrite_neuron_assignment = assign_random_neurons(unique_exc_dendrites, num_exc)
    inh_axon_neuron_assignment = assign_random_neurons(unique_inh_axons, num_inh)
    inh_dendrite_neuron_assignment = assign_random_neurons(unique_inh_dendrites, num_inh)

    # Increment the values for inhibitory neurons (as inhibitory neuron ids have to lie within
    # a different range from the excitatory ones
    inh_axon_neuron_assignment += num_exc
    inh_dendrite_neuron_assignment += num_exc

    axon_neuron_assignment = np.concatenate((exc_axon_neuron_assignment, inh_axon_neuron_assignment))
    dendrite_neuron_assignment = np.concatenate((exc_dendrite_neuron_assignment, inh_dendrite_neuron_assignment))

    unique_axons = np.concatenate((unique_exc_axons, unique_inh_axons))
    unique_dendrites = np.concatenate((unique_exc_dendrites, unique_inh_dendrites))

    # Finally create dictionaries that map axon/dendrite id to the neuron id of neuron they are attached to
    axon_neuron_dict = dict(zip(unique_axons, axon_neuron_assignment))
    dendrite_neuron_dict = dict(zip(unique_dendrites, dendrite_neuron_assignment))

    axon_neuron_map = np.vectorize(axon_neuron_dict.get)
    dendrite_neuron_map = np.vectorize(dendrite_neuron_dict.get)

    # Compute the two new columns for the dataframe using the dictionary
    input_neuron_id = axon_neuron_map(df.axon_id.values)
    output_neuron_id = dendrite_neuron_map(df.dendrite_id.values)

    # Add the two new columns to the dataframe
    if copy:
        df = df.copy()

    df["input_neuron_id"] = input_neuron_id
    df["output_neuron_id"] = output_neuron_id

    return df


def assign_random_neurons(unique_ids, num_neurons):
    """
    Return a random assignment of neurons to axon/dendrite ids, making sure each neuron is connected to at least one
    dendrite/axon if possible

    Randomly assign each axon and dendrite to a neuron of correct type
    To make sure that each neuron is present at least once:
    1. Create ordered array of size equal to number of neurons
    2. Randomly generate more neuron_ids to make the array match the size of unique axon/dendrite array
    3. Randomly permute (shuffle) the array
    :param unique_ids: Array of unique axon/dendrite ids
    :param num_neurons: Number of neurons to be assigned
    :return: ndarray
    """
    if unique_ids.size >= num_neurons:
        # If possible, make sure each neuron present at least once
        neurons_ordered = np.arange(1, num_neurons + 1)
        neuron_remainder = np.random.randint(1, num_neurons + 1, size=[unique_ids.size - num_neurons])
        neuron_assignment = np.concatenate((neurons_ordered, neuron_remainder))
        # Shuffle the assignments
        np.random.shuffle(neuron_assignment)
    else:
        neuron_assignment = np.random.randint(1, num_neurons + 1, size=[unique_ids.size])
    return neuron_assignment


def kasthuri_to_graph(df):
    # Specify which attributes will be stored as edge attributes
    edge_attributes = ['synapse_id',
                    'psd_centroid_x',
                    'psd_centroid_y',
                    'psd_centroid_z',  # PSD centroid (microns from origin, pixel (1,1,1))
                    'in_cylinder_1',
                    'in_cylinder_2',
                    'in_cylinder_3',  # Synapse located within cylinder 1, 2, or 3
                    'axon_id',  # Axon No
                    'bouton_id',  # Bouton No (Not in Cylinder 1 = -1 Not available = 0)
                    'axon_terminal',  # Axon terminal = 1 En-passant synapse = 0 (Not in cylinder 1 or 2 = -1)
                    'vesicle_count',  # Vesicle count (-1 = Not in Cylinder 1 or not available)
                    'mitochondria_count',  # Number of mitochondria in axon bouton (Not available = -1)
                    'multisynaptic_bouton',  # Multi synaptic axon bouton Yes = 1; No = 0 Unknown = -1
                    'axon_skeleton_length',  # Axon skeleton length within Cylinder 1 in micrometer (Not available = -1)
                    'spine_or_shaft',  # Spine synapse  = 1 Shaft synapse = 0
                    'psd_size',  # PSD size metric (pixels)
                    'spine_id',  # Spine ID No. (shaft synapse = 0 Unknown = -1)
                    'spine_vol',  # Spine Volume (in number of voxels, at 24x24x29 nm each) (Shaft synapse = -1
                    # not available = 0)
                    'spine_apparatus',  # NEW Spine Apparatus No = 0; Yes = 1 N/A = -1 Uncertain = -2
                    'num_synapses_on_spine'  # Nr of synapses on spine
                    ]

    G = nx.from_pandas_edgelist(df, 'input_neuron_id','output_neuron_id',
                                edge_attr=edge_attributes,
                                create_using=nx.MultiDiGraph())

    # Add the node attributes manually:
    for index, row in df.iterrows():
        # If axon_type is inhibitory or excitatory (not unknown)
        if row.axon_type == 1 or row.axon_type == 0:
            # Specify the node on the graph that is the input to the synapse represented by row
            input_node = G.nodes[int(row.input_neuron_id)]
            if 'neuron_type' in input_node.keys():
                assert input_node['neuron_type'] == int(row.axon_type)
            else:
                input_node['neuron_type'] = int(row.axon_type)

        # Repeat for the output node (for the dendrite)
        # If dendrite_type is inhibitory or excitatory (not unknown)
        if row.dendrite_type == 1 or row.dendrite_type == 0:
            output_node = G.nodes[int(row.output_neuron_id)]
            if 'neuron_type' in output_node.keys():
                assert output_node['neuron_type'] == int(row.dendrite_type)
            else:
                output_node['neuron_type'] = int(row.dendrite_type)
    return G


def convert_to_bipartite(G):
    """Take a semi-compact representation and convert to full bipartite"""
    pass


def write_graph_to_binary(G, filepath):
    """Save the graph to a binary file. The values are stored as follows:
    'input_neuron_id': unsigned long (L)
    'output_neuron_id': unsigned long (L)
    'synapse_id': unsigned long long (Q)
    'psd_centroid_x': float (f)
    'psd_centroid_y': float (f)
    'psd_centroid_z': float (f)
    'in_cylinder_1': bool (?)
    'in_cylinder_2': bool (?)
    'in_cylinder_3': bool (?)
    'axon_id': long long (q)
    'axon_skeleton_length': float (f)
    'axon_terminal': short (h)
    'bouton_id': long long (q)
    'mitochondria_count': short (h)
    'multisynaptic_bouton': short (h)
    'num_synapses_on_spine': short (h)
    'psd_size': int (i)
    'spine_apparatus': short (h)
    'spine_id': long (l)
    'spine_or_shaft': short (h)
    'spine_vol': int (i)
    'vesicle_count': int (i)
    """
    with open(filepath, 'wb') as out_file:
        for u, v, d in G.edges(data=True):
            edge_bytes = struct.pack('=LLQfff???qfhqhhhihlhii', int(u), int(v),
                        int(d['synapse_id']),
                        d['psd_centroid_x'],
                        d['psd_centroid_y'],
                        d['psd_centroid_z'],
                        int(d['in_cylinder_1']),
                        int(d['in_cylinder_2']),
                        int(d['in_cylinder_3']),
                        int(d['axon_id']),
                        d['axon_skeleton_length'],
                        int(d['axon_terminal']),
                        int(d['bouton_id']),
                        int(d['mitochondria_count']),
                        int(d['multisynaptic_bouton']),
                        int(d['num_synapses_on_spine']),
                        int(d['psd_size']),
                        int(d['spine_apparatus']),
                        int(d['spine_id']),
                        int(d['spine_or_shaft']),
                        int(d['spine_vol']),
                        int(d['vesicle_count']))
            out_file.write(edge_bytes)
    return


def read_binary_to_graph(filepath):
    G = nx.MultiDiGraph()

    with open(filepath, 'rb') as in_file:
        while True:
            edge_bytes = in_file.read(79)  # todo: Replace with class variable
            if not edge_bytes: break
            assert len(edge_bytes) == 79

            edge_data = struct.unpack('=LLQfff???qfhqhhhihlhii', edge_bytes)
            input_neuron_id = edge_data[0]
            output_neuron_id = edge_data[1]
            attribute_dict = dict()
            attribute_dict['synapse_id'] = edge_data[2]
            attribute_dict['psd_centroid_x'] = edge_data[3]
            attribute_dict['psd_centroid_y'] = edge_data[4]
            attribute_dict['psd_centroid_z'] = edge_data[5]
            attribute_dict['in_cylinder_1'] = edge_data[6]
            attribute_dict['in_cylinder_2'] = edge_data[7]
            attribute_dict['in_cylinder_3'] = edge_data[8]
            attribute_dict['axon_id'] = edge_data[9]
            attribute_dict['axon_skeleton_length'] = edge_data[10]
            attribute_dict['axon_terminal'] = edge_data[11]
            attribute_dict['bouton_id'] = edge_data[12]
            attribute_dict['mitochondria_count'] = edge_data[13]
            attribute_dict['multisynaptic_bouton'] = edge_data[14]
            attribute_dict['num_synapses_on_spine'] = edge_data[15]
            attribute_dict['psd_size'] = edge_data[16]
            attribute_dict['spine_apparatus'] = edge_data[17]
            attribute_dict['spine_id'] = edge_data[18]
            attribute_dict['spine_or_shaft'] = edge_data[19]
            attribute_dict['spine_vol'] = edge_data[20]
            attribute_dict['vesicle_count'] = edge_data[21]

            G.add_edges_from([(input_neuron_id, output_neuron_id, attribute_dict)])
    return G


def convert_to_compact(G):
    """Take a semi-compact representation (directed multigraph), and turn it into compact representation (digraph)"""
    # Create a DiGraph from the MultiDiGraph representation
    G_compact = nx.DiGraph(G)

    for edge in G_compact.edges.keys():
        # Clear all the previous attributes (they are only relevant to edges in MultiGraphs)
        G_compact.edges[edge].clear()
        num_synapses = G.get_edge_data(*edge)
        G_compact.egdes[edge]['num_synapses'] = num_synapses
    return G_compact


def test_save_to_binary():
    df = read_kasthuri_generated('datasets/kasthuri_generated200.xls')
    G = kasthuri_to_graph(df)
    # Write graph to a test binary file
    write_graph_to_binary(G, 'datasets/kasthuri_binary_test.bin')
    G_from_bin = read_binary_to_graph('datasets/kasthuri_binary_test.bin')
    for edge in G.edges:
        attr_original = G.edges[edge]
        attr_from_bin = G_from_bin.edges[edge]
        for key in attr_original.keys():
            try:
                assert attr_original[key] == attr_from_bin[key]
            except AssertionError as error:
                # If the numbers aren't equal, check that they are equal to a significant number of
                # significant figures (they could differ due to floating point rounding errors)
                if not math.isclose(attr_original[key], attr_from_bin[key], abs_tol=0.0001):
                    # If the numbers differ by more than the error margin, re-raise the exception with additional
                    # information
                    error.args += (f"The attribute {key} differs.\n"
                                   f"Original: {attr_original[key]}, rounded: {round(attr_original[key], 2)}\n"
                                   f"From binary: {attr_from_bin[key]}, rouned: {round(attr_from_bin[key], 2):}", )
                    raise
    return


if __name__ == '__main__':
    # df = read_kasthuri_original('datasets/kasthuri_original.xls')
    # df_generated = allocate_neurons(df, 200, copy=True)
    # # Save to xls file
    # df_generated.to_excel('datasets/kasthuri_generated200.xls')

    df = read_kasthuri_generated('datasets/kasthuri_generated200.xls')
    G = kasthuri_to_graph(df)

    write_graph_to_binary(G, 'datasets/kasthuri_graph.bin')

    G = read_binary_to_graph('datasets/kasthuri_graph.bin')


# Synapse No.
# PSD centroid (microns from origin, pixel (1,1,1))
# PSD centroid (pixel location column, row, section)
# Synapse located within cylinder 1, 2, or 3
# Axon No
# Dendrite No
# Axon type Excitatory = 0 Inhibitory = 1 Myelinated = 2 Unknown = -1
# Bouton No (Not in Cylinder 1 = -1 Not available = 0)
# Axon terminal = 1 En-passant synapse = 0 (Not in cylinder 1 or 2 = -1)
# Vesicle count (-1 = Not in Cylinder 1 or not available)
# Number of mitochondria in axon bouton (Not available = -1)
# Multi synaptic axon bouton Yes = 1; No = 0 Unknown = -1
# Axon skeleton length within Cylinder 1 in micrometer (Not available = -1)
# Spine synapse  = 1 Shaft synapse = 0
# Dendrite type: Excitatory/Spiny = 0 Inhibitory/Smooth = 1
# PSD size metric (pixels)
# Spine ID No. (shaft synapse = 0 Unknown = -1)
# Spine Volume (in number of voxels, at 24x24x29 nm each) (Shaft synapse = -1 not available = 0)
# NEW Spine Apparatus No = 0; Yes = 1 N/A = -1 Uncertain = -2
# Nr of synapses on spine

