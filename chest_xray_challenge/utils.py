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


def pre_process_image(path,normalize=True,resize_img=True,output_shape=256):

    # reading the image using path
    img = imread( path , pilmode="RGB")

    if normalize == True:
        # normalize the pixel values
        img = img/255

    if resize_img == True:
        # resizing the image to (28,28,3)
        img = resize(
                     img,
                     output_shape=(output_shape,output_shape,3),
                     mode='constant',
                     anti_aliasing=True
                     )

    # converting the type of pixel to float 32
    img = img.astype('float32')

    return img
