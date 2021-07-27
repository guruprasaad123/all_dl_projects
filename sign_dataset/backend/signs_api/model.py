import tensorflow as tf
import numpy as np
from PIL import Image
import os
# from matplotlib import image as mpimg
# from matplotlib import pyplot as plt

class api():

    height=64
    width=64
    channels=3
    model_name = 'cnn_model'
    classes = { 0 : 'Zero' , 1 : 'One' , 2 : 'Two' , 3 : 'Three' , 4 : 'Four' , 5 : 'Five' }

    def reset_graph(self,seed=42):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)


    def __init__(self,upload_path='uploads'):

        self.upload_path = upload_path

        # self.model_name = 'cnn_model'
        print('print',os.path.join('signs_api','{}.meta'.format(self.model_name)))

        # self.import_meta = tf.train.import_meta_graph(os.path.join('signs_api','{}.meta'.format(self.model_name)))

    def predict(self,im):

        try :

            # im = Image.open( os.path.join(self.upload_path,filename) )

            #image size
            size=(self.height,self.width)
            #resize image
            out = im.resize(size)

            test_image =  np.array(out.getdata())

            test_image = test_image.reshape((-1,self.height,self.width,self.channels))

            # to make this notebook's output stable across runs
            self.reset_graph()

            # import meta from directory
            # import_meta = tf.train.import_meta_graph('{}.meta'.format(self.model_name))
            import_meta = tf.train.import_meta_graph(os.path.join('signs_api','{}.meta'.format(self.model_name)))

            with tf.Session() as sess:

                # tf.train.latest_checkpoint(<dir>) also works

                import_meta.restore(sess,'{}.ckpt'.format( os.path.join('signs_api',self.model_name) ) )

                # W1_val = sess.graph.get_tensor_by_name('W1:0')

                # X_val = sess.graph.get_tensor_by_name('Placeholder:0')

                ArgMax = sess.graph.get_tensor_by_name('ArgMax:0')

                ArgMax_val = ArgMax.eval({ 'Placeholder:0' : test_image })

                # graph = tf.get_default_graph()

                # for op in graph.get_operations():
                #     print(op.name)

                # print('W1_val',W1_val)
                # print('X_val',X_val)
                print('ArgMax',ArgMax_val)
                index = ArgMax_val.tolist()[0]
                class_val = self.classes[index]

                # os.remove(os.path.join(self.upload_path,filename))

                return { 'value' : index , 'class' : class_val  }

        except (OSError,IOError) as e:
            print('error',e)
            return { 'error' : True }
