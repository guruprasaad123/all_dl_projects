from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import os

model = load_model('best_model.h5') 

img_path = os.path.join("../","model_test","hand_sign_9.jpg")

# img_path = 'my_image.jpg'

def load_image_test():
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

    print('Test Image Shape',test_image.shape)


    test_image = test_image.reshape((-1,height,width,channels))

    return test_image


def load_img(img_path):

    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    return x

x = load_img(img_path)

print( model.predict(x) )

