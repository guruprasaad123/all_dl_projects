import numpy as np
import pandas as pd
import os
# for reading and displaying images
from skimage.io import imread
from skimage.transform import resize
# for reading the csv
import csv
# for creating datasets
import h5py
from matplotlib import pyplot as plt

import logging
from functools import lru_cache

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *
# inceptionv-1
# from pytorch.inception.inceptionv_1.model import *
# inceptionv-3
from pytorch.inception.inceptionv_3.model import *
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# import training dataset
train_dataset_path = os.path.join( 'training.h5')

# reading the contents from the train.h5 file
train_dataset = h5py.File(train_dataset_path,'r')

# layers
layers = 18

# training data features
train_x = train_dataset['train_x'][:]
train_y = train_dataset['train_y'][:]
list_class = train_dataset['classes'][:]

# wrap the values with numpy array
train_x = np.array(train_x)
train_y = np.array(train_y)
list_class = np.array(list_class)

train_y_hot = convert_to_one_hot( train_y , len(list_class) ).T

print('train_x ' , train_x.shape)
print('train_y ' , train_y_hot.shape)
print('list_class ',list_class)
print('train_y_hot : ',train_y_hot[0:10])
print('train_y : ' , train_y[0:10] )
# import testing dataset
test_dataset_path = os.path.join( 'testing.h5')

# reading the contents from the test.h5 file
test_dataset = h5py.File(test_dataset_path,'r')

# training data features
test_x = test_dataset['test_x'][:]
list_class = test_dataset['classes'][:]
test_array = test_dataset['test_array'][:]

test_array = list( map( lambda x : x.decode('utf-8')  , test_array ) )

# wrap the values with numpy array
test_x = np.array(test_x)
list_class = np.array(list_class)
array_test = np.array(test_array)

array_list = list_class.tolist()

print('test_x ',test_x.shape)
print('list_class ',array_list)

img_size = train_x.shape[1]
c_dim = train_x.shape[3]
label_dim = train_y_hot.shape[1]

print( "img_size : {} ".format(img_size) )
print( "c_dim : {} ".format(c_dim) )
print( "label_dim : {} ".format(label_dim) )


if __name__ == '__main__':

    dataset = ImagesDataset( train_x , train_y )

    batch_size = 128
    num_workers = 2

    train_loader = DataLoader( dataset, batch_size=batch_size,shuffle=False, num_workers=num_workers)

    net = inception_v3( aux_logits=True , num_classes=label_dim ).cuda()

    train_v2( train_loader , net )

