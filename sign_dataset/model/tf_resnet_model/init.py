import math
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
# from utils import *
from model import *

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# import training dataset
train_dataset_path = os.path.join('../','signs_train','train_signs.h5')
# import testing dataset
test_dataset_path = os.path.join('../','signs_test','test_signs.h5')

train_dataset = h5py.File(train_dataset_path,'r')
test_dataset = h5py.File(test_dataset_path,'r')

# training data features
train_x = train_dataset['train_set_x'][:]
# wrap the values with numpy array
train_x = np.array(train_x)

# training data labels
train_y = train_dataset['train_set_y'][:]
# wrap the values with numpy array
train_y = np.array(train_y)
# reshape y to 1 x D
train_y = train_y.reshape((1,-1))

# training data features
test_x = test_dataset['test_set_x'][:]
# wrap the values with numpy array
test_x = np.array(test_x)

# training data labels
test_y = test_dataset['test_set_y'][:]
# wrap the values with numpy array
test_y = np.array(test_y)
# reshape y to 1 x D
test_y = test_y.reshape((1,-1))

# list of classes
classes = test_dataset['list_classes'][:]
# wrap yhe values with numpy array
classes = np.array(classes)


print(' train_x : {} \n train_y : {} \n test_x : {} \n test_y : {} \n classes : {} '.format( train_x.shape,train_y.shape,test_x.shape , test_y.shape,classes.shape ))

# values of X_train could be Standardized/Regularized by dividing all the values by 255 ( Maxinum [0-255] )
X_train = train_x/255.
# values of X_test  could be Standardized/Regularized by dividing all the values by 255 ( Maxinum [0-255] )
X_test = test_x/255.
Y_train = convert_to_one_hot(train_y, classes.shape[0]).T
Y_test = convert_to_one_hot(test_y, classes.shape[0]).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

img_size = X_train.shape[1]
c_dim = X_train.shape[3]
label_dim = Y_test.shape[1]

print( "img_size : {} ".format(img_size) )
print( "c_dim : {} ".format(c_dim) )
print( "label_dim : {} ".format(label_dim) )

# build_model( batch_size=64 , img_size=28 , c_dim=3 , label_dim=6 , test_x=X_train  , test_y=Y_train )

train( X_train , Y_train , X_test , Y_test )

conv_layers = {}

# _, _, parameters = model(X_train, Y_train, X_test, Y_test)