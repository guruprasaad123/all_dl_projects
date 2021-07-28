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
import numpy as np

from pytorch.inception.inceptionv_3.model import *
from utils import *

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

# (3489, 28, 28, 3)

# test_x = np.transpose(  test_x , ( 0 , 3 , 1 , 2 ) )

print('transpose => ',test_x.shape)

if __name__ == '__main__':

    net = load_model_v2( len(array_list) )

    net.eval()

    # [N, C, W, H]

    test_dataset = TestDataset( test_x )

    labels = test_v2( test_dataset , list_class , net )

    # print('labels : ' , labels.shape)    

    obj = dict({ 'Image' : test_array , 'Class' : labels })

    df = pd.DataFrame(obj)

    print('images => ',df)

    test_dir = os.path.join( 'test_2.csv' )

    df.to_csv( test_dir , index=False)


