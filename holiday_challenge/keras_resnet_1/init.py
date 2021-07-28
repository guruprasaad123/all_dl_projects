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

# from utils import *
from model import *

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# import training dataset
train_dataset_path = os.path.join( '../','train.h5')

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


# import testing dataset
test_dataset_path = os.path.join( '../','test.h5')

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

input_shape = ( img_size , img_size , c_dim )

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt/tf__keras_resnet_{}".format(layers)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Prepare a directory to store all the logs for tensorboard.
log_dir = "./logs/tf_keras_resnet_{}".format(layers)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def create_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints , key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return create_model( layers=layers, input_shape=input_shape , classes=label_dim )


model = create_or_restore_model()

# 1e-3
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.46, staircase=True
)

optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)


model.compile(
    # Optimizer
    optimizer = keras.optimizers.Adam(0.0001) ,  
    # Loss function to minimize
    loss = 'categorical_crossentropy' ,
    # List of metrics to monitor
    metrics = [
        'accuracy'
        ] ,
)

filepath = checkpoint_dir + "/tf_keras_resnet_{}".format( layers ) + "_{epoch}"

callbacks = [

    keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath=filepath,
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="accuracy",
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least \42 epochs"
        patience=6,
        verbose=1,
    ),

    keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq="epoch",
    ),  # How often to write logs (default: once per epoch)


]


model.fit( train_x , train_y_hot , batch_size=64 , epochs=25 ,  callbacks=callbacks )


predictions = model.predict( test_x )

predictions_argmax = np.argmax( predictions , axis=1)

def convert(x):
    
    return list_class[int(x)].decode('utf-8')

convert_label = np.vectorize( convert )

labels = convert_label( predictions_argmax )

print('Predictions : ' , predictions_argmax.shape)

print('labels => ' , labels)


obj = dict({ 'Image' : test_array , 'Class' : labels })

df = pd.DataFrame(obj)

print('images => ',df)

test_dir = os.path.join( 'test_{}.csv'.format(layers) )

df.to_csv( test_dir , index=False)
