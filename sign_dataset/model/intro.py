import math
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from utils import *
import tensorflow as tf

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# import training dataset
train_dataset_path = os.path.join('signs_train','train_signs.h5')
# import testing dataset
test_dataset_path = os.path.join('signs_test','test_signs.h5')

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


height = X_train.shape[1]
width = X_train.shape[2]
channels = X_train.shape[3]
n_inputs = height * width * channels

conv1_fmaps = 64
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 128
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 10

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 16 * 16]) # 64 * 7 * 7 = 3136 , / 1769472

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):

    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

X_train = X_train.astype(np.float32).reshape(-1, height*width*channels) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, height*width*channels) / 255.0
train_y = train_y.reshape((-1))
Y_train = train_y.astype(np.int32)
Y_test = test_y.astype(np.int32).reshape((-1))

def shuffle_batch(X, y, batch_size):
    # print('length of X',len(X) , X.shape , len(y) ,y.shape)
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

n_epochs = 10
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, Y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        # cost_batch = cost.eval(feed_dict={X: X_batch , y : y_batch})
        # acc_test = accuracy.eval(feed_dict={X: X_test, y: Y_test})
        
        acc_test , logits_val , Y_proba_val = sess.run( 
            [ accuracy , logits , Y_proba ] ,
            feed_dict={ X : X_test , y : Y_test } 
            )

        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)
        # print('logits_val',logits_val.reshape((-1)),logits_val.shape)
        # print('Y_proba',Y_proba_val.reshape((-1)),Y_proba_val.shape)

        save_path = saver.save(sess, "./my_signs_model")