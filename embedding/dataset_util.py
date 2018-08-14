import torch
import torch.nn as nn
import h5py


from torch.utils.data import DataLoader, Dataset


class GEDDataset(Dataset):
    """Graph Edit Distance dataset class."""

    def __init__(self, h5_file, which_set='train', adj_dtype=np.float32, transform=None):
        # todo: finish docstring
        """

        """
        self.h5_file = h5py.File(h5_file, mode="r")
        assert which_set.lower() in ['train', 'val', 'test']
        self.which_set = which_set.lower()
        self.adj_dtype = adj_dtype

    def __len__(self):
        return len(self.h5_file[self.which_set + '_labels'])

    def __getitem__(self, idx):
        graph1 = self.h5_file[self.which_set + "_graph1"][idx]
        graph2 = self.h5_file[self.which_set + "_graph2"][idx]
        label = self.h5_file[self.which_set + "_labels"][idx]

        graph1 = graph1.astype(self.adj_dtype)
        graph2 = graph2.astype(self.adj_dtype)

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

