from model import *
from utils import *

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from utils import *
from model import *
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
'''
def load_model():


    net = TwinNetwork().to(device)

    checkpoint_dir = os.path.join( 'checkpoint' , 'siemese_network_v2' )

    model_state = None

    dirs = os.listdir( checkpoint_dir )

    if len(dirs) > 0:

        latest_checkpoint = max(dirs)

        checkpoint = torch.load( os.path.join( checkpoint_dir , latest_checkpoint ) )

        model_state = checkpoint['model_state_dict']
        # opt_state = checkpoint['optimizer_state_dict']
        last_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(' [*] Model Restored from {} Epoch \n'.format(last_epoch) )

    if model_state:
        net.load_state_dict(model_state)

    net.eval()

    return net

'''
'''
def confidence_score( net , x0 , x1 ):

    x0 , x1 = transforms.ToTensor()(x0).unsqueeze(0) , transforms.ToTensor()(x1).unsqueeze(0)

    # print( 'after : ' , x0.shape , x1.shape )

    # concat = torch.cat((x0,x1),0)

    output1,output2 = net( x0.to(device) , x1.to(device) )

    eucledian_distance = F.pairwise_distance(output1, output2)

    return eucledian_distance

test_obj = {
'src' :  'tests/jennifer-aniston.jpg',
'dest' : 'tests/tom-cruise.jpg'
}

net = load_model()

input0, input1 = auto_crop( test_obj['src'] ) , auto_crop( test_obj['dest'] )

eucledian_distance = confidence_score( net , input0 , input1 )

print( 'confidence score : ' , eucledian_distance.item() )
