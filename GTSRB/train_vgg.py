from matplotlib import pyplot as plt
import csv
import os
import h5py
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pandas as pd

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix , precision_score , recall_score , f1_score
# from resources.plotcm import plot_confusion_matrix

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# to import models
# from torchvision import models

from dataset import *
from model import *

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter( os.path.join( 'runs' , 'GTSRB_VGG_SE_11' ) )
device = torch.device( 'cuda' if torch.cuda.is_available() == True else 'cpu' )

def split_train_test_val( train_data , test_train_split=0.8 , val_train_split=0.2 , shuffle=True ):

    dataset_size = len( train_data )
    indices = list( range( dataset_size ) )
    test_split = int( np.floor( test_train_split * dataset_size ) )

    if shuffle == True:
        np.random.shuffle(indices)

    train_indices, test_indices = indices[:test_split] , indices[test_split:]
    train_size = len( train_indices )
    validation_split = int( np.floor( ( 1 - val_train_split ) * train_size ) )

    train_indices, val_indices = train_indices[ : validation_split ] , train_indices[ validation_split : ]

    test_data = train_data[ test_indices ]
    val_data = train_data[ val_indices ]
    train_data = train_data[ train_indices ]

    return ( train_data , test_data , val_data )

import time

EPOCHS = 44
BATCH_SIZE = 128
LEARNING_RATE = 1e-1
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 43
INPUT_IMAGE_SIZE = 28

CHECKPOINT_PATH = os.path.join( 'checkpoint' , 'GTSRB_VGG_SE_11' )

def main():

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



    train_loader , val_loader , test_loader = get_loaders( batch_size=BATCH_SIZE )

    print('Training : ' , len(train_loader) )

    print('Validation : ' , len(val_loader) )

    print('Testing : ' , len(test_loader) )

    net = VGG11( num_classes=NUM_CLASSES ).to(device)

    at_start = True

    if model_state:
        net.load_state_dict(model_state)

    ACE = nn.CrossEntropyLoss().to(device)

    opt = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=.9, nesterov=True)

    if opt_state:
        opt.load_state_dict(opt_state)

    for epoch in range( last_epoch+1 , EPOCHS + 1 ) if last_epoch else range(1, EPOCHS + 1):

        print('[Epoch %d]' % epoch)

        train_loss = 0
        train_correct, train_total = 0 , 0

        start_point = time.time()

        batch_iter = 0

        for inputs, labels in train_loader:

            labels = np.array(labels ,dtype=np.long)

            inputs = torch.as_tensor(inputs)
            labels = torch.as_tensor(labels)


            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            if at_start == True:
                # create grid of images
                img_grid = make_grid(inputs)

                # write to tensorboard
                writer.add_image('sample Images'.format(batch_iter), img_grid)

                writer.add_graph(net, inputs)

                at_start = False

            opt.zero_grad()

            preds = net(inputs)

            loss = ACE(preds, labels)
            loss.backward()

            opt.step()

            train_loss += loss.item()

            train_correct += (preds.argmax(dim=1) == labels).sum().item()
            train_total += len(preds)

            # cm = confusion_matrix( labels , preds.argmax(dim=1) )

            training_p_score , training_recall , training_f1_score = get_metrics( labels , preds )

            writer.add_scalar('training p-score',
                                training_p_score ,
                                epoch * len(train_loader) + batch_iter
                                )

            writer.add_scalar('training recall',
                                training_recall ,
                                epoch * len(train_loader) + batch_iter
                                )


            writer.add_scalar('training f1-score',
                                training_f1_score ,
                                epoch * len(train_loader) + batch_iter
                                )

            # log the training loss
            writer.add_scalar('training loss',
                             train_loss / len(train_loader),
                            epoch * len(train_loader) + batch_iter
                            )

            # log the training accuracy
            writer.add_scalar('training accuracy',
                            100 * train_correct / train_total,
                            epoch * len(train_loader) + batch_iter
                            )

            batch_iter = batch_iter + 1

            printProgressBar( batch_iter , len(train_loader) , prefix = 'Training : ', suffix = 'Complete', length = 50)

        print('train-acc : %.4f%% train-loss : %.5f' % (100 * train_correct / train_total, train_loss / len(train_loader) ) )
        print('elapsed time: %ds' % (time.time() - start_point))

        test_loss = 0
        test_correct, test_total = 0 , 0
        batch_iter = 0

        for inputs, labels in test_loader:

            with torch.no_grad():

                labels = np.array(labels ,dtype=np.long)

                inputs , labels = torch.as_tensor(inputs) , torch.as_tensor(labels)

                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

                preds = net(inputs)

                test_loss += ACE(preds, labels).item()

                test_correct += (preds.argmax(dim=1) == labels).sum().item()

                test_total += len(preds)

                testing_p_score , testing_recall , testing_f1_score = get_metrics( labels , preds )

                writer.add_scalar('testing p-score',
                                    testing_p_score ,
                                    epoch * len(test_loader) + batch_iter
                                    )

                writer.add_scalar('testing recall',
                                    testing_recall ,
                                    epoch * len(test_loader) + batch_iter
                                    )


                writer.add_scalar('testing f1-score',
                                    testing_f1_score ,
                                    epoch * len(test_loader) + batch_iter
                                    )

                # log the testing loss
                writer.add_scalar('testing loss',
                                test_loss / len(test_loader),
                                epoch * len(test_loader) + batch_iter
                                )

                # log the testing accuracy
                writer.add_scalar('testing accuracy',
                                100 * test_correct / test_total,
                                epoch * len(test_loader) + batch_iter
                                )

                batch_iter = batch_iter + 1

                printProgressBar( batch_iter , len(test_loader) , prefix = 'Testing : ', suffix = 'Complete', length = 50)

        print('test-acc : %.4f%% test-loss : %.5f' % (100 * test_correct / test_total, test_loss / len(test_loader) ) )

        val_loss = 0
        val_correct, val_total = 0 , 0
        batch_iter = 0

        for inputs, labels in val_loader:

            with torch.no_grad():

                labels = np.array(labels ,dtype=np.long)

                inputs , labels = torch.as_tensor(inputs) , torch.as_tensor(labels)

                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

                preds = net(inputs)

                val_loss += ACE(preds, labels).item()

                val_correct += (preds.argmax(dim=1) == labels).sum().item()

                val_total += len(preds)

                val_p_score , val_recall , val_f1_score = get_metrics( labels , preds )

                writer.add_scalar('validation p-score',
                                    val_p_score ,
                                    epoch * len(val_loader) + batch_iter
                                    )

                writer.add_scalar('validation recall',
                                    val_recall ,
                                    epoch * len(val_loader) + batch_iter
                                    )


                writer.add_scalar('validation f1-score',
                                    val_f1_score ,
                                    epoch * len(val_loader) + batch_iter
                                    )

                # log the validation loss
                writer.add_scalar('validation loss',
                                val_loss / len(val_loader) ,
                                epoch * len(val_loader) + batch_iter
                                )

                # log the validation accuracy
                writer.add_scalar('validation accuracy',
                                100 * val_correct / val_total,
                                epoch * len(val_loader) + batch_iter
                                )

                batch_iter = batch_iter + 1

                printProgressBar( batch_iter , len(val_loader) , prefix = 'Validating : ', suffix = 'Complete', length = 50)

        print('val-acc : %.4f%% val-loss : %.5f' % (100 * val_correct / val_total, val_loss / len(val_loader) ) )


        for name , weight in net.named_parameters():
            writer.add_histogram( name , weight , epoch )
            writer.add_histogram( '{}.grad'.format(name) , weight.grad , epoch )

        torch.save( {
             'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
                    } , os.path.join( CHECKPOINT_PATH , 'GTSRB-checkpoint-{:04d}.bin'.format(epoch) ) )

        writer.flush()

    writer.close()


if __name__ == '__main__':

    main()
