from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch import nn , optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from torch.autograd import Variable
# from pytorch.inception.inceptionv_1.model import *
from pytorch.inception.inceptionv_3.model import *
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

class ImagesDataset(Dataset):
    """GTSRB Landmarks dataset."""

    def __init__(self, training_images , training_labels , transform=None):

        self.images = training_images
        self.labels = training_labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if self.transform:
            image = self.transform( self.images[idx] )
            label = self.labels[idx]
            # label = self.transform( self.labels[idx] )
        else:
            image = self.images[idx]
            label = self.labels[idx]

        return ( image.T , label )

class TestDataset(Dataset):
    """GTSRB Landmarks dataset."""

    def __init__(self, training_images , transform=None):

        self.images = training_images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if self.transform:
            image = self.transform( self.images[idx] )
            # label = self.transform( self.labels[idx] )
        else:
            image = self.images[idx]

        return ( image.T )



EPOCHS = 140
BATCH_SIZE = 128
LEARNING_RATE = 1e-1
WEIGHT_DECAY = 1e-4

CHECKPOINT_PATH = os.path.join( 'checkpoint' , 'inception_v3' )

def train_v2(train_loader,net):

    model_state = None
    opt_state = None
    last_epoch = None
    loss = None

    # if the checkpoint path does not exists , then create it
    if not os.path.exists( CHECKPOINT_PATH ):
        os.makedirs( CHECKPOINT_PATH )
    # if checkpoint path already exists
    else:
        dirs = os.listdir( CHECKPOINT_PATH )

        if len(dirs) > 0:

            latest_checkpoint = max(dirs)

            checkpoint = torch.load( os.path.join(CHECKPOINT_PATH,latest_checkpoint) )

            model_state = checkpoint['model_state_dict']
            opt_state = checkpoint['optimizer_state_dict']
            last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            print(' [*] Model Restored from {} Epoch \n'.format(last_epoch) )

            model_restored = True

    # net = InceptionV1( training=True , aux_logits=True , num_classes=10 ).cuda()

    if model_state:
        net.load_state_dict(model_state)

    ACE = nn.CrossEntropyLoss().cuda()

    opt = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=.9, nesterov=True)

    if opt_state:
        opt.load_state_dict(opt_state)

    for epoch in range( last_epoch+1 , EPOCHS + 1 ) if last_epoch else range(1, EPOCHS + 1):

        print('[Epoch %d]' % epoch)

        train_loss = 0
        train_correct, train_total = 0 , 0

        batch_iter = 0

        start_point = time.time()

        for inputs, labels in train_loader:

            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # print('inputs : shape' , inputs.shape )

            opt.zero_grad()

            preds, aux_ops_1 = net(inputs)

            # preds  = net(inputs)

            loss_1 = ACE(aux_ops_1, labels)
            real_loss = ACE(preds, labels)

            loss = real_loss + ( 0.4 * loss_1 )

            loss.backward()

            opt.step()

            train_loss += loss.item()

            train_correct += (preds.argmax(dim=1) == labels).sum().item()
            train_total += len(preds)

            batch_iter = batch_iter + 1

            printProgressBar( batch_iter , len(train_loader) , prefix = 'Training : ', suffix = 'Complete', length = 50)

        print('train-acc : %.4f%% train-loss : %.5f' % (100 * train_correct / train_total, train_loss / len(train_loader)))
        print('elapsed time: %ds' % (time.time() - start_point))

        test_loss = 0
        test_correct, test_total = 0 , 0

        batch_iter = 0

        torch.save( {
             'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
                    } , os.path.join( CHECKPOINT_PATH , 'cifar-10-checkpoint-{:04d}.bin'.format(epoch) ) )


def train(train_loader,net):

    model_state = None
    opt_state = None
    last_epoch = None
    loss = None

    # if the checkpoint path does not exists , then create it
    if not os.path.exists( CHECKPOINT_PATH ):
        os.makedirs( CHECKPOINT_PATH )
    # if checkpoint path already exists
    else:
        dirs = os.listdir( CHECKPOINT_PATH )

        if len(dirs) > 0:

            latest_checkpoint = max(dirs)

            checkpoint = torch.load( os.path.join(CHECKPOINT_PATH,latest_checkpoint) )

            model_state = checkpoint['model_state_dict']
            opt_state = checkpoint['optimizer_state_dict']
            last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            print(' [*] Model Restored from {} Epoch \n'.format(last_epoch) )

            model_restored = True

    # net = InceptionV1( training=True , aux_logits=True , num_classes=10 ).cuda()

    if model_state:
        net.load_state_dict(model_state)

    ACE = nn.CrossEntropyLoss().cuda()

    opt = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=.9, nesterov=True)

    if opt_state:
        opt.load_state_dict(opt_state)

    for epoch in range( last_epoch+1 , EPOCHS + 1 ) if last_epoch else range(1, EPOCHS + 1):

        print('[Epoch %d]' % epoch)

        train_loss = 0
        train_correct, train_total = 0 , 0

        batch_iter = 0

        start_point = time.time()

        for inputs, labels in train_loader:

            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # print('inputs : shape' , inputs.shape )

            opt.zero_grad()

            aux_ops_1 , aux_ops_2 , preds  = net(inputs)

            # preds  = net(inputs)

            loss_1 = ACE(aux_ops_1, labels)
            loss_2 = ACE(aux_ops_2, labels)
            real_loss = ACE(preds, labels)

            loss = real_loss + ( 0.3 * loss_1 ) + ( 0.3 * loss_2 )

            loss.backward()

            opt.step()

            train_loss += loss.item()

            train_correct += (preds.argmax(dim=1) == labels).sum().item()
            train_total += len(preds)

            batch_iter = batch_iter + 1

            printProgressBar( batch_iter , len(train_loader) , prefix = 'Training : ', suffix = 'Complete', length = 50)

        print('train-acc : %.4f%% train-loss : %.5f' % (100 * train_correct / train_total, train_loss / len(train_loader)))
        print('elapsed time: %ds' % (time.time() - start_point))

        test_loss = 0
        test_correct, test_total = 0 , 0

        batch_iter = 0

        torch.save( {
             'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
                    } , os.path.join( CHECKPOINT_PATH , 'cifar-10-checkpoint-{:04d}.bin'.format(epoch) ) )

def load_model(label_dim):

    model_state = None
    opt_state = None
    last_epoch = None
    loss = None

    # if the checkpoint path does not exists , then create it
    if not os.path.exists( CHECKPOINT_PATH ):
        os.makedirs( CHECKPOINT_PATH )
    # if checkpoint path already exists
    else:
        dirs = os.listdir( CHECKPOINT_PATH )

        if len(dirs) > 0:

            latest_checkpoint = max(dirs)

            checkpoint = torch.load( os.path.join(CHECKPOINT_PATH,latest_checkpoint) )

            model_state = checkpoint['model_state_dict']
            opt_state = checkpoint['optimizer_state_dict']
            last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            print(' [*] Model Restored from {} Epoch \n'.format(last_epoch) )

            model_restored = True

    net = InceptionV1( training=True , aux_logits=True , num_classes=label_dim ).cuda()

    if model_state:
        net.load_state_dict(model_state)

    return net


def load_model_v2(label_dim):

    model_state = None
    opt_state = None
    last_epoch = None
    loss = None

    # if the checkpoint path does not exists , then create it
    if not os.path.exists( CHECKPOINT_PATH ):
        os.makedirs( CHECKPOINT_PATH )
    # if checkpoint path already exists
    else:
        dirs = os.listdir( CHECKPOINT_PATH )

        if len(dirs) > 0:

            latest_checkpoint = max(dirs)

            checkpoint = torch.load( os.path.join(CHECKPOINT_PATH,latest_checkpoint) )

            model_state = checkpoint['model_state_dict']
            opt_state = checkpoint['optimizer_state_dict']
            last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            print(' [*] Model Restored from {} Epoch \n'.format(last_epoch) )

            model_restored = True

    net = inception_v3( aux_logits=True , num_classes=label_dim ).cuda()

    if model_state:
        net.load_state_dict(model_state)

    return net

def test_v2( dataset , list_class , net ):

    test_loader = DataLoader( dataset, batch_size=128,shuffle=False, num_workers=2 )

    labels_array = []

    for inputs in test_loader:

        inputs = Variable(inputs).cuda()


        preds = net(inputs)

        predictions_argmax = np.array( preds.argmax(dim=1).cpu() )

        # print( 'predictions_argmax ' ,predictions_argmax )

        def convert(x):

            return list_class[int(x)].decode('utf-8')

        convert_label = np.vectorize( convert )

        labels = convert_label( predictions_argmax )

        # print('labels : ' , labels )

        labels_array.extend( [ *labels.tolist() ]  )
    # print('labels_array' , labels_array)
    return np.array( labels_array )


def test( dataset , list_class , net ):

    test_loader = DataLoader( dataset, batch_size=128,shuffle=False, num_workers=2 )

    labels_array = []

    for inputs in test_loader:
        
        inputs = Variable(inputs).cuda()

        aux_output_0 , aux_output_1 , preds = net(inputs)

        predictions_argmax = np.array( preds.argmax(dim=1).cpu() ) 

        # print( 'predictions_argmax ' ,predictions_argmax )

        def convert(x):
        
            return list_class[int(x)].decode('utf-8')

        convert_label = np.vectorize( convert )

        labels = convert_label( predictions_argmax )

        # print('labels : ' , labels )

        labels_array.extend( [ *labels.tolist() ]  )
    # print('labels_array' , labels_array)
    return np.array( labels_array )
