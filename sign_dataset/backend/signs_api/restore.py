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
im = Image.open(os.path.join("model_test","hand_sign_0.jpg") )

#image size
size=(height,width)
#resize image
out = im.resize(size)

test_image =  np.array(out.getdata())

test_image = test_image.reshape((height,width,channels))

# print(  )

#save resized image
out.save(os.path.join("model_test","resize_hand_sign_1.jpg") )

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

# import meta from directory

model_name = 'cnn_model'

import_meta = tf.train.import_meta_graph('{}.meta'.format(model_name))

with tf.Session() as sess:

    # tf.train.latest_checkpoint(<dir>) also works

    import_meta.restore(sess,'{}.ckpt'.format(model_name))

    W1_val = sess.graph.get_tensor_by_name('W1:0')
    
    X_val = sess.graph.get_tensor_by_name('Placeholder:0')

    ArgMax = sess.graph.get_tensor_by_name('ArgMax:0')

    ArgMax_val = ArgMax.eval({ 'Placeholder:0' : test_image })

    # graph = tf.get_default_graph()
    
    # for op in graph.get_operations():
    #     print(op.name)

    # print('W1_val',W1_val)
    # print('X_val',X_val)
    print('ArgMax',ArgMax_val)