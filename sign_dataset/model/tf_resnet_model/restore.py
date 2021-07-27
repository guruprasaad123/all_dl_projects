import tensorflow as tf
import numpy as np
from PIL import Image
import os
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

'''

When saving the model, you'll notice that it takes 4 types of files to save it:

".meta" files: containing the graph structure
".data" files: containing the values of variables
".index" files: identifying the checkpoint
"checkpoint" file: a protocol buffer with a list of recent checkpoints

'''

height = width = 64
channels = 3

#read the image
im = Image.open( os.path.join("../","model_test","hand_signs_1.jpg") )

#image size
size=(height,width)
#resize image
out = im.resize(size)

test_image =  np.array(out.getdata())

test_image = test_image.reshape((height,width,channels))

# print(  )

#save resized image
out.save(os.path.join("../","model_test","resize_hand_sign_1.jpg") )

print('Resized Successfully')

# test_image = mpimg.imread(os.path.join("model_test","resize_hand_sign_1.jpg"))[:, :, :channels]

print('Test Image Shape',test_image.shape)

#plt.imshow(test_image) 
# plt.axis("off")
# plt.show()

test_image = test_image.reshape((-1,height,width,channels))

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

model_name = 'ResNet.model'

os.path.join('checkpoint')

import_meta = tf.train.import_meta_graph('checkpoint/ResNet18_64_0.1/final/{}.ckpt.meta'.format(model_name))

with tf.Session() as sess:

    # tf.train.latest_checkpoint('checkpoint/ResNet18_64_0.1'final/) # also works

    import_meta.restore( sess ,'checkpoint/ResNet18_64_0.1/final/{}.ckpt'.format(model_name) )

    # W1_val = sess.graph.get_tensor_by_name('W1:0')
    
    # X_val = sess.graph.get_tensor_by_name('Placeholder:0')

    # ArgMax = sess.graph.get_tensor_by_name('ArgMax:0')

    # ArgMax_val = ArgMax.eval({ 'Placeholder:0' : test_image })

    test_inputs = sess.graph.get_tensor_by_name( 'test_inputs:0' )

    ArgMax_2 = sess.graph.get_tensor_by_name( 'ArgMax_2:0' )

    print('test_inputs => ',test_inputs)

    print('test_image => ',test_image.shape)

    print('ArgMax_2 => ',ArgMax_2)

    ArgMax_2_val = ArgMax_2.eval({ 'test_inputs:0' : test_image })

    print('ArgMax_2_val => ' , ArgMax_2_val )

    # biasAdd = sess.graph.get_tensor_by_name( 'network_1/logit/dense/BiasAdd:0' )

    # biasAdd_val = biasAdd.eval({ 'test_inputs:0' : test_image })

    # print('biasAdd_val => ',biasAdd_val)

    # graph = tf.get_default_graph()
    
    # for op in graph.get_operations():
    #     print(op.name)

    # print('W1_val',W1_val)
    # print('X_val',X_val)
    # print('ArgMax',ArgMax_val)