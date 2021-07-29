import numpy as np
import pandas as pd
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils

# from vgg import *
from model import *

def get_model():

    BATCH_SIZE = 128
    NUM_CLASSES = 43

    CHECKPOINT_PATH = os.path.join( 'checkpoint' , 'GTSRB_VGG_SE_11' )

    dirs = os.listdir( CHECKPOINT_PATH )

    latest_checkpoint = max(dirs)

    checkpoint = torch.load( os.path.join(CHECKPOINT_PATH,latest_checkpoint) )

    model_state = checkpoint['model_state_dict']
    opt_state = checkpoint['optimizer_state_dict']
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(' [*] Model Restored from {} Epoch \n'.format(last_epoch) )

    model = VGG11( num_classes=NUM_CLASSES ).cuda()

    model.load_state_dict(model_state)

    model.eval()

    return model

def predict( model , input_image ):

    image = Image.open(input_image)
    image = image.convert("RGB")
    image

    load_size = 32

    h = image.size[0]
    w = image.size[1]
    ratio = h * 1.0 / w
    if ratio > 1:
        h = load_size
        w = int(h*1.0 / ratio)
    else:
        w = load_size
        h = int(w * ratio)

    image = image.resize((28, 28), Image.BICUBIC)
    image

    image = np.asarray(image)
    gpu = torch.cuda.is_available()
    # convert PIL image from  RGB to BGR
    image = image[:, :, [2, 1, 0]]

    # print('image : before' , image.shape)
    image = transforms.ToTensor()(image).unsqueeze(0)
    # transform values to (-1, 1)
    # image = -1 + 2 * image
    if gpu:
        image = Variable(image).cuda()
    else:
        image = image.float()

    # style transformation
    with torch.no_grad():
        # print('image : after' , image.shape)
        output = model(image)
        # output = output[0]

    print(output.argmax(dim=1))

    return output.argmax(dim=1)

if __name__ == '__main__':
    
    input_image = os.path.join( 'test' , 'left-sign.jpg' )

    model = get_model()

    result = predict( model , input_image )
