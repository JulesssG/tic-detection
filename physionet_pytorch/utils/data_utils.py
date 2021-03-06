from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn.utils import weight_norm
import scipy.io as sio
import numpy as np
from os import path
from .get_data import get_data





class TransformSubset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole dataset.
        indices (sequence): Indices in the whole set selected for subset.
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __getitem__(self, idx):
        if self.dataset.transform != self.transform:
            self.dataset.transform = self.transform
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class PhysionetMMMI(torch.utils.data.Dataset):

    def __init__(self, datapath, num_classes=4, transform=None):
        self.datapath = datapath
        self.transform = transform
        self.num_classes = num_classes
        if not path.isfile(path.join(datapath, f'{num_classes}class.npz')):
            print("npz file not existing. Load .edf and save data in npz files for faster loading of data next time.")
            X, y = get_data(datapath, n_classes=num_classes)
            np.savez(path.join(datapath,f'{num_classes}class'), X = X, y = y)
        npzfile = np.load(path.join(datapath, f'{num_classes}class.npz'))
        X, y = npzfile['X'], npzfile['y']
        self.samples = torch.Tensor(X).to(dtype=torch.float)
        self.labels = torch.Tensor(y).to(dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx, :, :]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
