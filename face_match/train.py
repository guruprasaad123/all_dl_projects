from model import *
from utils import *
import sys
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

import os

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Declare Siamese Network
net = TwinNetwork().to(device)

# Decalre Loss Function
criterion = ContrastiveLoss()

# Declare Optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

# epochs
epochs = 40

# batch_size
batch_size = 4

# transfprms
transforms = transforms.Compose([
    transforms.ToTensor()
    ])

# dataset 
dataset = FacepairDataset(transforms=transforms)

# training , testing , validations
# train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
split = DataSplit( dataset , shuffle=True , val_train_split=None )

train_loader, test_loader = split.get_split( batch_size=batch_size , num_workers=8 )

# check point directory for storing model's progress on each epoch
checkpoint_dir = os.path.join( 'checkpoint' , 'siemese_network_v2' )

# tensorboard for visualizing the model performance
writer = SummaryWriter( os.path.join( 'runs' , 'siemese_network_v2' ) )

# train the model
def train():

    model_state = None
    opt_state = None
    last_epoch = None
    loss = None
    best_loss = float(sys.maxsize)

    if os.path.exists( checkpoint_dir ) == False:

        os.makedirs( checkpoint_dir )

    else:

        dirs = os.listdir( checkpoint_dir )

        if len(dirs) > 0:
            
            latest_checkpoint = max(dirs)

            checkpoint = torch.load( os.path.join( checkpoint_dir , latest_checkpoint ) )

            model_state = checkpoint['model_state_dict']
            opt_state = checkpoint['optimizer_state_dict']
            last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            print(' [*] Model Restored from {} Epoch \n'.format(last_epoch) )

    if model_state:
        net.load_state_dict(model_state)
    
    if opt_state:
        optimizer.load_state_dict(opt_state)

    at_start = True
    
    # if starting as new then start from 0 , or else if resuming the previous state 

    for epoch in range( last_epoch+1 , epochs + 1 ) if last_epoch else range(1, epochs + 1):

        # training

        net.train()
        
        total_train_loss = 0
        batch_iter = 0

        print(" [*] Starting Epoch : {}".format(epoch) )
        
        for i, data in enumerate(train_loader,0):


            img0, img1 , label = data
            img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)

            if at_start == True:
                # create grid of images
                img_grid = make_grid( img0 )

                # write to tensorboard
                writer.add_image('sample Images'.format(batch_iter), img_grid)
                
                writer.add_graph(net, (img0 , img1) )

                at_start = False

            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            
            train_loss_contrastive = criterion( output1 , output2 , label )

            total_train_loss += train_loss_contrastive.item()

            train_loss_contrastive.backward()

            optimizer.step()

            # log the training loss
            writer.add_scalar('training loss',
                             total_train_loss / len(train_loader),
                            epoch * len(train_loader) + batch_iter
                            )
            
            batch_iter = batch_iter + 1

            printProgressBar( i+1 , len(train_loader) , prefix='Training' )
        
        print(" [*] training loss : {:.5f}".format( epoch , ( total_train_loss / len(train_loader) ) ) )

        net.eval()

        total_test_loss = 0
        batch_iter = 0

        print(' Total Test Loss : ' , total_test_loss )

        for i , data in enumerate( test_loader , 0 ):

            img0, img1 , label = data
            img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)
            
            output1,output2 = net(img0,img1)
            
            test_loss_contrastive = criterion(output1,output2,label)

            total_test_loss += test_loss_contrastive.item()
            
            # log the training loss
            writer.add_scalar('testing loss',
                             total_test_loss / len(test_loader),
                            epoch * len(test_loader) + batch_iter
                            )
            
            batch_iter = batch_iter + 1
            
            printProgressBar( i+1 , len(test_loader) , prefix='Testing' )
        
        print(" [*] testing loss : {:.5f}".format( epoch , ( total_test_loss / len(test_loader) ) ) )

        '''
        
        total_val_loss = 0
        batch_iter = 0

        for i , data in enumerate( val_loader , 0 ):

            img0, img1 , label = data
            img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)
            
            output1,output2 = net(img0,img1)
            
            val_loss_contrastive = criterion(output1,output2,label)

            total_val_loss += val_loss_contrastive.item()

            # log the training loss
            writer.add_scalar('validation loss',
                             total_val_loss / len(val_loader),
                            epoch * len(val_loader) + batch_iter
                            )
            
            batch_iter = batch_iter + 1
            
            printProgressBar( i+1 , len(val_loader) , prefix='Validating' )   
        
        print(" Vaidation loss : {:.5f}".format( epoch , ( total_val_loss / len(val_loader) ) ) )

        '''

        best_loss = min( best_loss , total_test_loss )

        if best_loss < total_test_loss:

            torch.save( {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_train_loss,
                } , os.path.join( checkpoint_dir , 'siemese-nn-checkpoint-{:04d}.bin'.format(epoch) ) )    

    # show_plot(counter, loss)   
    return net
#set the device to cuda
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = train()
# torch.save(model.state_dict(), "model.pt")
# print("Model Saved Successfully") 

if __name__ == '__main__':

    train()
