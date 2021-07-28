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


# def predict(path) :
#     img = tf.keras.preprocessing.image.load_img(path,target_size=(150,150))
#     img = tf.keras.preprocessing.image.img_to_array(img)
#     img = img / 255.0
#     img = np.array([img])
#     pred = labels[np.argmax(model.predict(img))]
#     plt.imshow(img.reshape(150,150,3))
#     plt.title(pred)

# fig, ax = plt.subplots(nrows=6, ncols=4,figsize = (15,20))

'''
For Plotting Images
'''
# for X_batch, y_batch in training_set:
#     i=0
#     for row in ax:
#         for col in row:
#             col.imshow(X_batch[i])
#             i+=1
#     break
# plt.show()   
    
def pre_process_image(path,normalize=True,output_shape=28):

    # reading the image using path
    img = imread( path )

    if normalize == True:
        # normalize the pixel values
        img = img/255

    # resizing the image to (28,28,3)
    img = resize(img, output_shape=(output_shape,output_shape,3), mode='constant', anti_aliasing=True)

    # converting the type of pixel to float 32
    img = img.astype('float32')

    return img

def get_mapping(filename):
    
    obj = dict({})

    with open( filename , mode='r') as csv_file:
        
        csv_reader = csv.DictReader(csv_file)

        for i, row in enumerate(csv_reader):

            if 1 == 0:
                print('Column names are : ' , row )
            else :
                
                image = row['Image'] 
                _class = row['Class']

                obj[image] = _class
        
        return obj           

def get_list(dir,mapping=None):
    
    images = os.listdir( os.path.join( dir ) )

    img_list , image_list = ( [] , [] )

    class_list = []

    for image in images:
        
        if mapping:
            # find the mapping and get the class
            _class = mapping[image]
            class_list.append(_class)

        # pre-process the image
        img = pre_process_image( os.path.join( dir , image ) , output_shape=32)

        # appending the image into the list
        img_list.append(img)

        # appending the image_name into the list
        image_list.append(image)

    # converting the list to numpy array
    array_list = np.array(img_list)
    array_image = np.array( image_list , dtype='S' )

    if mapping:
        
        unique_ones = np.unique(class_list).tolist()
        print('unique_ones' , unique_ones)
    
        class_map = list(map( lambda x : unique_ones.index(x) , class_list ))
        # print('class_map' , class_map)
        classes = np.array(class_map)

        return ( array_list , unique_ones , classes )
    
    else:
        return ( array_list , array_image )

def create_dataset( obj=dict({}) , name='data' ):

    filename = '{}.h5'.format(name)

    hf = h5py.File( filename , 'w-' )

    for key in obj:
            hf.create_dataset( key , data=obj[key] )
    
    hf.close()

def extract_values( dir , filename ):

    with open( filename , mode='r') as csv_file:
        
        csv_reader = csv.DictReader(csv_file)

        image_list , class_list , img_list = ( list() , list() , list() )

        for i, row in enumerate(csv_reader):

            if 1 == 0:
                print('Column names are : ' , row )
            else :
                
                image = row['Image']
                
                image_path = os.path.join( dir , image )

                img = pre_process_image( image_path , output_shape=32)

                class_name = row['Class']

                image_list.append(image)

                img_list.append(img)

                class_list.append(class_name)

        image_array , img_array , class_array = ( np.array( image_list , dtype='S' ) , np.array( img_list ) , np.array( class_list , dtype='S' ) )

        unique_ones = np.unique(class_list).tolist()
        print('unique_ones' , unique_ones)
    
        class_map = list(map( lambda x : unique_ones.index(x) , class_list ))
        # print('class_map' , class_map)
        class_array = np.array( class_map )

        classes = np.array( unique_ones , dtype='S' )

        return ( image_array , img_array , class_array , classes )

csv_path = os.path.join( 'train.csv' )

# mapping = get_mapping( csv_path )

train_dir = 'train'
test_dir = 'test'

train_array , train_x , train_y , classes = extract_values( train_dir , csv_path )

# ['Airplane', 'Candle', 'Christmas_Tree', 'Jacket', 'Miscellaneous', 'Snowman']

# train_x , classes , train_y = get_list( train_dir , mapping )

test_x , test_array = get_list( test_dir )

train_obj = dict( { 'train_array' : train_array , 'train_x' : train_x , 'classes' : classes , 'train_y' : train_y } )

create_dataset( train_obj , 'training' )

print( 'train_x => ',train_x.shape )

print( 'train_y => ',train_y.shape )

print( 'test => ',test_x.shape )

test_obj = dict({ 'test_x' : test_x , 'test_array' : test_array , 'classes' : classes })

create_dataset( test_obj , 'testing' )
