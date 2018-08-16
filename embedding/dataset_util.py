import torch
import torch.nn as nn
import h5py
import numpy as np


from torch.utils.data import DataLoader, Dataset


class GEDDataset(Dataset):
    """Graph Edit Distance dataset class."""

    def __init__(self, h5_file, which_set='train', adj_dtype=np.float32, transform=None, load_into_memory=True):
        # todo: finish docstring
        """

        """
        self.h5_file = h5py.File(h5_file, mode="r")
        assert which_set.lower() in ['train', 'val', 'test']
        self.which_set = which_set.lower()
        self.adj_dtype = adj_dtype
        self.transform = transform
        self.load_into_memory = load_into_memory
        if load_into_memory:
            # Load the entire dataset features into memory
            self.graph1_set = self.h5_file[self.which_set + "_graph1"][()]
            self.graph2_set = self.h5_file[self.which_set + "_graph2"][()]
            self.labels_set = self.h5_file[self.which_set + "_labels"][()]
            print('Data loaded into memory.')
        else:
            # Just load pointers to the dataset, while dataset remains on disk
            self.graph1_set = self.h5_file[self.which_set + "_graph1"]
            self.graph2_set = self.h5_file[self.which_set + "_graph2"]
            self.labels_set = self.h5_file[self.which_set + "_labels"]
            print('Dataset being accessed from disc (single worker only).')

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def __len__(self):
        return len(self.h5_file[self.which_set + '_labels'])

    def __getitem__(self, idx):
        graph1 = self.graph1_set[idx]
        graph2 = self.graph2_set[idx]
        label = self.labels_set[idx]

        graph1 = graph1.astype(self.adj_dtype)
        graph2 = graph2.astype(self.adj_dtype)
        label = label.astype(np.float32)

        sample = {'graph1': graph1, 'graph2': graph2, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        graph1, graph2, label = sample['graph1'], sample['graph2'], sample['label']

        return {'graph1': torch.from_numpy(graph1),
                'graph2': torch.from_numpy(graph2),
                'label': torch.from_numpy(label)}

