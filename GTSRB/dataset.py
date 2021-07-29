from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py

from utils import *

class GTSRBDataset(Dataset):
    """GTSRB Landmarks dataset."""

    def __init__(self, training_images , training_labels , transform=None):

        self.images = training_images
        self.labels = training_labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if self.transform:
            image = self.transform( self.images[idx] )
            label = self.labels[idx]
            # label = self.transform( self.labels[idx] )
        else:
            image = self.images[idx]
            label = self.labels[idx]

        return ( image.T , label )

def get_loaders(batch_size=128):

    df =  h5py.File('training_new.h5', "r")

    train_images = df['train_images'][:]

    train_labels = df['train_labels'][:]

    labels = np.array(train_labels,'O')

    images = np.array(train_images)

    dataset = GTSRBDataset( images , labels )

    split = DataSplit(dataset, shuffle=True)

    train_loader, val_loader, test_loader = split.get_split(batch_size=batch_size, num_workers=8)

    return ( train_loader , val_loader , test_loader )

if __name__ == '__main__':

    df =  h5py.File('training_new.h5', "r")

    train_images = df['train_images'][:]

    train_labels = df['train_labels'][:]

    labels = np.array(train_labels,'O')

    images = np.array(train_images)

    dataset = GTSRBDataset( images , labels )

    split = DataSplit(dataset, shuffle=True)

    train_loader, val_loader, test_loader = split.get_split(batch_size=128, num_workers=8)

    print('Training : ' , len(train_loader) )

    print('Validation : ' , len(val_loader) )

    print('Testing : ' , len(test_loader) )
