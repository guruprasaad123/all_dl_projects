import torch
from torch import nn
from torch.nn import functional as F
import os

'''
This utility is for creating conv_layer with desired input , filter_size along with relu activate , batch normalization
'''
def conv_layer( out_channels , in_channels = 3 , kernels=3 , norm=True , max_pool=True , drop_out = None ):
    # conv_layer
    layers = list([])

    # print(' conv ' , in_channels , out_channels )

    layers.append( nn.Conv2d( in_channels , out_channels , kernel_size=kernels, padding=1 , stride=1 ) )

    if norm == True:
        layers.append( nn.LocalResponseNorm( 5 , alpha=0.0001 , beta=0.75 , k=2 ) )
    
    layers.append( nn.ReLU(inplace=True) )
    
    if max_pool == True:
        layers.append( nn.MaxPool2d( 3 , stride=2 ) )

    if drop_out:
        layers.append( nn.Dropout2d( p=drop_out ) )

    return layers


'''
This utility is for creating fully connected dense layers with desired input & output , with relu , drop_out
'''
def full_layer( in_channels , out_channels , drop_out=None ):
    # full layer

    layers = list([])

    layers.append( nn.Linear( in_channels , out_channels) )
    layers.append( nn.ReLU(inplace=True) )

    if drop_out:
        layers.append( nn.Dropout2d( p=drop_out ) )

    return layers

class TwinNetwork(nn.Module):
    def __init__( self , in_channels=3 ):
        super(TwinNetwork, self).__init__()

        self.conv_layers = [
            {
                'out_channels' : 96 ,
                'kernels' : 11 ,
            },
            {
                'out_channels' : 256 ,
                'kernels' : 5 ,
                'drop_out' : 0.3
            },
            {
                'out_channels' : 384 ,
                'kernels' : 3,
                'norm' : False,
                'max_pool' : False,
            },
            {
                'out_channels' : 256 ,
                'kernels' : 3,
                'norm' : False,
                'drop_out' : 0.5
            }
        ]

        self.full_layers = [
            {
                'out_channels' : 1024,
                'drop_out' : 0.5
            },
            {
                'out_channels' : 128
            }
        ]

        self.features = nn.Sequential( *self._make_conv_features( in_channels=in_channels ) )

        # print( self.features )
        # 30976 , 43264
        self.classifier = nn.Sequential( *self._make_full_layers( in_channels=43264 ) )

        # print( self.classifier )

    def forward_once( self , x ):
        # forward
        out = self.features(x)

        out = out.view( out.size()[0] , -1 )

        # print( 'shape : ' , out.shape)

        out = self.classifier(out)

        return out

    def forward( self , x0 , x1 ):

        # forward
        out0 = self.forward_once(x0)

        out1 = self.forward_once(x1)

        return out0 , out1

    def _make_conv_features(self , in_channels=3):

        layers = []

        for ob in self.conv_layers:

            ob['in_channels'] = in_channels

            layer = conv_layer( **ob )

            in_channels = ob['out_channels']

            layers.extend(layer)

        return layers

    def _make_full_layers(self , in_channels=3):

        layers = []

        for ob in self.full_layers:

            ob['in_channels'] = in_channels

            layer = full_layer( **ob )

            in_channels = ob['out_channels']

            layers.extend(layer)

        return layers

class SiemeseNetwork(nn.Module):

    def __init__(self,in_channels=3):
        super(SiemeseNetwork,self).__init__()

        self.twin_network0 = TwinNetwork(in_channels=in_channels)
        self.twin_network1 = TwinNetwork(in_channels=in_channels)

    def forward(self,x0,x1):

        out0 = self.twin_network0(x0)

        out1 = self.twin_network1(x1)

        return out0 , out1

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

if __name__ == '__main__':

    # SiemeseNetwork(in_channels=3)

    TwinNetwork()


