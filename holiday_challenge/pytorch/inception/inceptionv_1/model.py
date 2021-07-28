import torch
from torch import nn
import torch.nn.functional as F

## Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d( in_channels , out_chanels , **kwargs )
        self.bn = nn.BatchNorm2d( out_chanels )
        
    def forward(self, x):
        relu = F.relu( self.bn( self.conv(x) ) )

        return relu

## Inception Block
class InceptionBlock(nn.Module):
    def __init__(
        self, 
        x,
        filters_1x1,
        filters_3x3_reduce,
        filters_3x3,
        filters_5x5_reduce,
        filters_5x5,
        filters_pool_proj,
    ):

        super(InceptionBlock, self).__init__()
        # red_ = _reduce
        
        self.conv_1x1 = ConvBlock( x , filters_1x1 , kernel_size=1 )

        self.conv_3x3 = nn.Sequential(
            ConvBlock( x , filters_3x3_reduce , kernel_size=1,padding=0 ),
            ConvBlock( filters_3x3_reduce , filters_3x3 , kernel_size=3 , padding=1 ),
        )

        self.conv_5x5 = nn.Sequential(
            ConvBlock( x , filters_5x5_reduce , kernel_size=1 ),
            ConvBlock( filters_5x5_reduce , filters_5x5 , kernel_size=3 , padding=1 ),
        )

        self.pool_proj = nn.Sequential(
            nn.MaxPool2d( kernel_size=3 , padding=1 , stride=1 , ceil_mode=True ),
            ConvBlock( x , filters_pool_proj , kernel_size=1 ),
        )

    
    def forward(self, x):
        # branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        branches = ( self.conv_1x1 , self.conv_3x3 , self.conv_5x5 , self.pool_proj )
        concat = torch.cat( [ branch(x) for branch in branches ] , 1)
        return concat

## InceptionAux
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AdaptiveAvgPool2d( (4, 4) )
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class InceptionV1(nn.Module):
    def __init__(self, in_channels=3, training=True, aux_logits=True , init_weights=True , num_classes=10):
        super(InceptionV1, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = ConvBlock(
            in_channels=in_channels, 
            out_chanels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionBlock(
            x=192, 
            filters_1x1=64, 
            filters_3x3_reduce=96, 
            filters_3x3=128, 
            filters_5x5_reduce=16, 
            filters_5x5=32, 
            filters_pool_proj=32,
            )

        self.inception3b = InceptionBlock(
            x=256, 
            filters_1x1=128, 
            filters_3x3_reduce=128, 
            filters_3x3=192, 
            filters_5x5_reduce=32, 
            filters_5x5=96, 
            filters_pool_proj=64,
            )

        self.inception4a = InceptionBlock(
            x=480, 
            filters_1x1=192, 
            filters_3x3_reduce=96, 
            filters_3x3=208, 
            filters_5x5_reduce=16, 
            filters_5x5=48, 
            filters_pool_proj=64,
            )

        self.inception4b = InceptionBlock(
            x=512, 
            filters_1x1=160, 
            filters_3x3_reduce=112, 
            filters_3x3=224, 
            filters_5x5_reduce=24, 
            filters_5x5=64, 
            filters_pool_proj=64,
            )

        self.inception4c = InceptionBlock(
            x=512, 
            filters_1x1=128, 
            filters_3x3_reduce=128, 
            filters_3x3=256, 
            filters_5x5_reduce=24, 
            filters_5x5=64, 
            filters_pool_proj=64,
            )

        self.inception4d = InceptionBlock(
            x=512, 
            filters_1x1=112, 
            filters_3x3_reduce=144, 
            filters_3x3=288, 
            filters_5x5_reduce=32, 
            filters_5x5=64, 
            filters_pool_proj=64,
            )

        self.inception4e = InceptionBlock(
            x=528, 
            filters_1x1=256, 
            filters_3x3_reduce=160, 
            filters_3x3=320, 
            filters_5x5_reduce=32, 
            filters_5x5=128, 
            filters_pool_proj=128,
            )

        self.inception5a = InceptionBlock(
            x=832, 
            filters_1x1=256, 
            filters_3x3_reduce=160, 
            filters_3x3=320, 
            filters_5x5_reduce=32, 
            filters_5x5=128, 
            filters_pool_proj=128,
            )

        self.inception5b = InceptionBlock(
            x=832, 
            filters_1x1=384, 
            filters_3x3_reduce=192, 
            filters_3x3=384, 
            filters_5x5_reduce=48, 
            filters_5x5=128, 
            filters_pool_proj=128,
            )

        
        self.avgpool = nn.AdaptiveAvgPool2d( (1, 1) )
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None
        
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)
        
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.aux_logits and self.training:
            return aux1, aux2, x
        return x

model = InceptionV1()
