import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from inception import GoogLeNet

from utils import *

import time
import os

EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 1e-1
WEIGHT_DECAY = 1e-4

CHECKPOINT_PATH = os.path.join( 'checkpoint' , 'mnist' )

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

    train_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),

    ]))

    test_dataset = MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),

    ]))

    # print('train_dataset' , train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    net = GoogLeNet( in_channels=1 , aux_logits=True , num_classes=10 ).cuda()

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

        for inputs, labels in test_loader:

            with torch.no_grad():

                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

                aux_ops_1 , aux_ops_2 , preds  = net(inputs)

                # preds  = net(inputs)
                
                loss_1 = ACE(aux_ops_1, labels)
                loss_2 = ACE(aux_ops_2, labels)
                real_loss = ACE(preds, labels)

                loss = real_loss + ( 0.3 * loss_1 ) + ( 0.3 * loss_2 )
                
                test_loss += loss.item()

                test_correct += (preds.argmax(dim=1) == labels).sum().item()

                test_total += len(preds)
                    
                batch_iter = batch_iter + 1

                printProgressBar( batch_iter , len(test_loader) , prefix = 'Testing : ', suffix = 'Complete', length = 50)

        print('test-acc : %.4f%% test-loss : %.5f' % (100 * test_correct / test_total, test_loss / len(test_loader)))

        torch.save( {
             'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
                    } , os.path.join( CHECKPOINT_PATH , 'mnist-checkpoint-{:04d}.bin'.format(epoch) ) )

if __name__ == '__main__':
    main()
