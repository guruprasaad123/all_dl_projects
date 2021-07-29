# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import csv
import os
import h5py
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def pre_process_image(path,normalize=False,resize_img=False):

    # reading the image using path
    img = imread( path )

    if normalize == True:
        # normalize the pixel values
        img = img/255

    if resize_img == True:
        # resizing the image to (28,28,3)
        img = resize(img, output_shape=(32,32,3), mode='constant', anti_aliasing=True)

    # converting the type of pixel to float 32
    img = img.astype('float32')

    return img


def create_dataset( obj=dict({}) , name='data' ):

    filename = '{}.h5'.format(name)

    hf = h5py.File( filename , 'w' )

    for key in obj:
            hf.create_dataset( key , data=obj[key] )

    hf.close()

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    arr_images = np.array([
                    [],
                    [],
                    [],
                    []
                    ]) # numpy array
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file

        print('gtfile : ',gtFile)

        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        # gtReader.next() # skip header
        next(gtReader)
        # loop over all images in current annotations file
        for row in gtReader:
            img = pre_process_image( prefix + row[0] , normalize=True , resize_img=True )
            # print('shape : ',img.shape)
            images.append( img ) # the 1th column is the filename
            # value = np.array( plt.imread(prefix + row[0]) )
            # print('value : shape ',value.shape)
            # np.append( arr_images , value , axis=0  )
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels


root_path = os.path.join('GTSRB_Final_Training_Images','GTSRB','Final_Training','Images')
# root_path = 'GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'

images , labels = readTrafficSigns(root_path)

arr_images = np.array(images)

# arr_images.astype(np.float64)

print('Images : ', arr_images.dtype)

train_obj = dict( { 'train_images' : arr_images , 'train_labels' : np.array( labels , dtype='S' ) } )

create_dataset( train_obj , 'training_new' )
