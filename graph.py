import pandas as pd
import numpy as np
import networkx as nx
import random
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
                    'num_synapses_on_spine'  # Nr of synapses on spine
                    ]
    # Dictionary from column name to initial index for convenience (as name cannot be used on creation of dataframe)
    column_idx_dict = {column_names[idx]: idx for idx in range(len(column_names))}

    # Dictionary to specify the desired types of each column when creating the dataframe
    column_dtypes = {column_idx_dict['synapse_id']: 'int32',
                     column_idx_dict['in_cylinder_1']: 'int32',
                     column_idx_dict['in_cylinder_2']: 'int32',
                     column_idx_dict['in_cylinder_3']: 'int32',  # Synapse located within cylinder 1, 2, or 3
                     column_idx_dict['axon_id']: 'int32',  # Axon No
                     column_idx_dict['dendrite_id']: 'int32',  # Dendrite No
                     column_idx_dict['axon_type']: 'int32',  # Axon type
                     column_idx_dict['bouton_id']: 'int32',  # Bouton No (Not in Cylinder 1 = -1 Not available = 0)
                     column_idx_dict['axon_terminal']: 'int32',  # Axon terminal = 1 En-passant synapse = 0
                     column_idx_dict['vesicle_count']: 'int32',  # Vesicle count
                     column_idx_dict['mitochondria_count']: 'int32',  # Number of mitochondria in axon bouton
                     column_idx_dict['multisynaptic_bouton']: 'int32',  # Multi synaptic axon bouton Yes = 1; No = 0
                     column_idx_dict['spine_or_shaft']: 'int32',  # Spine synapse  = 1 Shaft synapse = 0
                     column_idx_dict['dendrite_type']: 'int32',  # Dendrite type:
                     column_idx_dict['spine_apparatus']: 'int32',  # NEW Spine Apparatus
                     column_idx_dict['num_synapses_on_spine']: 'int32'  # Nr of synapses on spine
                     }

    df = pd.read_excel(file_path, header=None, skiprows=2, names=column_names, dtype=column_dtypes)

    # Drop redundant information
    df = df.drop('psd_centroid_pixel', axis=1)
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
    inhibitory_axons = df[(df.axon_type == 1)].axon_id
    excitatory_axons = df[(df.axon_type == 0)].axon_id
    other_axons = df[np.logical_and((df.axon_type != 0), (df.axon_type != 1))].axon_id # myelinated or uncertain

    unique_inh_axons = np.unique(inhibitory_axons.values)
    unique_exc_axons = np.unique(excitatory_axons.values)
    unique_other_axons = np.unique(other_axons.values)

    # For now, consider other to be excitatory (only three other axons in kasthuri anyways)
    unique_exc_axons = np.concatenate((unique_exc_axons, unique_other_axons))

    inhibitory_dendrites = df[(df.dendrite_type == 1)].dendrite_id
    excitatory_dendrites = df[(df.dendrite_type == 0)].dendrite_id

    unique_inh_dendrites = np.unique(inhibitory_dendrites.values)
    unique_exc_dendrites = np.unique(excitatory_dendrites.values)

    num_axons = unique_inh_axons.size + unique_exc_axons.size
    num_dendrites = unique_inh_dendrites.size + unique_exc_dendrites.size

    if num_neurons > num_axons + num_dendrites:
        raise AttributeError("The number of neurons {} exceeds the total number of axons and dendrites {}".format(
            num_neurons, num_axons + num_dendrites
        ))

    # Generate two sets of neurons (inhibitory and excitatory)
    fraction_inh_neurons = (unique_inh_dendrites.size + unique_inh_axons.size) / (num_axons + num_dendrites)
    num_inh = math.floor(num_neurons * fraction_inh_neurons)
    num_exc = num_neurons - num_inh

    # Randomly assign each axon and dendrite to a neuron of correct type
    # To make sure that each neuron is present at least once:
    # 1. Create ordered array of size equal to number of neurons
    # 2. Randomly generate more neuron_ids to make the array match the size of unique axon/dendrite array
    # 3. Randomly permute (shuffle) the array

    exc_axon_neuron_assignment = assign_random_neurons(unique_exc_axons, num_exc)
    exc_dendrite_neuron_assignment = assign_random_neurons(unique_exc_dendrites, num_exc)
    inh_axon_neuron_assignment = assign_random_neurons(unique_inh_axons, num_inh)
    inh_dendrite_neuron_assignment = assign_random_neurons(unique_inh_dendrites, num_inh)

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


# def test_correctness(df):
#     pass

if __name__ == '__main__':
    df = read_kasthuri_original('datasets/kasthuri_original.xls')
    df_generated = allocate_neurons(df, 200, copy=True)
    # Save to xls file
    df_generated.to_excel('datasets/kasthuri_generated200.xls')


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

