from PIL import Image
from tqdm import tqdm
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPBlock, self).__init__()
        """
        Initialize a basic multi-layer perceptron module components.
        Illustration: https://docs.google.com/drawings/d/1gTPLeK0H5ooMcn7CNPysqwr9_07fTqkHE4-T3ZqyhPo/edit?usp=sharing

        Instructions:
            1. Implement an algorithm that initializes necessary components as illustrated in the above link. 
            2. Initialized network components will be referred in `forward` method 
               for constructing the dynamic computational graph.
        
        Args:
            1. in_channels (int): Number of channels in input.
            2. out_channels (int): Number of channels to be produced.
        """
        #######################################
        ## This section is an example.       ##        
        self.fc1 = nn.Linear(in_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        #######################################

    def forward(self, x):
        """
        Feed-forward data 'x' through the module.
                
        Instructions:
            1. Construct the feed-forward computational graph as illustrated in the link 
               using the initialized components in __init__ method.

        Args:
            1. x (torch.FloatTensor): A tensor of shape (B, in_channels)
            .
        Returns:
            1. output (torch.FloatTensor): An output tensor of shape (B, out_channels). 
        """
        #######################################
        ## This section is an example.       ##
        output = self.act(self.bn1(self.fc1(x)))
        output = self.act(self.bn2(self.fc2(output)))
        output = self.act(self.bn3(self.fc3(output)))
        #######################################
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1):
        super(ConvBlock, self).__init__()
        """
        Initialize a basic convolutional layer module components.
        Illustration: https://docs.google.com/drawings/d/1MRYBywpuazlldwC11UTa-kuWMWEDsFewDnirKiFX5us/edit?usp=sharing

        Args:
            1. in_channels (int): Number of channels in the input. 
            2. out_channels (int): Number of channels produced.
            3. kernel_size (int) : Size of the kernel used in conv layer (Default:3)
            4. stride (int) : Stride of the convolution (Default:1)
            5. padding (int) : Zero-padding added to both sides of the input (Default:1)
        """
        #################################
        ## P1.1. Write your code here  ##
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding = padding, bias =False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()




        #################################

    def forward(self, x):
        """
        Feed-forward the data 'x' through the module.
        Instructions:
            1. Construct the feed-forward computational graph as illustrated in the link 
               using the initialized components in __init__ method.

        Args:
            1. x (torch.FloatTensor): A tensor of shape (B, in_channels, H, W).
            
        Returns:
            1. output (torch.FloatTensor): An output tensor of shape (B, out_channels, H, W). 
        """
        #################################
        ## P1.2. Write your code here  ##       
        output = self.act(self.bn1(self.conv1(x)))



        #################################
        return output


class ResBlockPlain(nn.Module):
    def __init__(self, in_channels):
        super(ResBlockPlain, self).__init__()
        """Initialize a residual block module components.

        Illustration: https://docs.google.com/drawings/d/19FS5w7anbTAF6UrMPdM4fs8nk9x3Lm5KRIODawC4duQ/edit?usp=sharing

        Instructions:
            1. Implement an algorithm that initializes necessary components as illustrated in the above link. 
            2. Initialized network components will be referred in `forward` method 
               for constructing the dynamic computational graph.

        Args:
            1. in_channels (int): Number of channels in the input.
        """
        #################################
        ## P2.1. Write your code here ##
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        #################################

    def forward(self, x):
        """Feed-forward the data `x` through the network.

        Instructions:
            1. Construct the feed-forward computational graph as illustrated in the link 
               using the initialized components in __init__ method.

        Args:
            1. x (torch.FloatTensor): An tensor of shape (B, in_channels, H, W).

        Returns:
            1. output (torch.FloatTensor): An output tensor of shape (B, in_channels, H, W). 
        """
        ################################
        ## P2.2. Write your code here ## 
        output = self.act(x + self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))))

        ################################
        return output 


class ResBlockBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ResBlockBottleneck, self).__init__()
        """Initialize a residual block module components.

        Illustration: https://docs.google.com/drawings/d/1n2E0TwiWhf1IGdD16-MeQjzUcys_V7ETTzn33j_bEy0/edit?usp=sharing

        Instructions:
            1. Implement an algorithm that initializes necessary components as illustrated in the above link. 
            2. Initialized network components will be referred in `forward` method 
               for constructing the dynamic computational graph.

        Args:
            1. in_channels (int): Number of channels in the input. 
            2. hidden_channels (int): Number of hidden channels produced by the first ConvLayer module.
        """
        #################################
        ## P3.1. Write your code here  ##
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=hidden_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(in_channels=hidden_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()




        #################################

    def forward(self, x):
        """Feed-forward the data `x` through the network.

        Instructions:
            1. Construct the feed-forward computational graph as illustrated in the link 
               using the initialized components in __init__ method.

        Args:
            1. x (torch.FloatTensor): An tensor of shape (B, in_channels, H, W).

        Returns:
            1. output (torch.FloatTensor): An output tensor of shape (B, in_channels, H, W). 
        """
        ################################
        ## P3.2. Write your code here ##
        output = self.act(x + self.bn3(self.conv3(self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))))) )



        ################################
        return output


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        """Initialize a basic InpcetionBlock module components.

        Illustration: https://docs.google.com/drawings/d/1I020R1YqVAr8LWKHgm7N5J5fzFpHvx1fqXuAs6z8qyE/edit?usp=sharing

        Instructions:
            1. Implement an algorithm that initializes necessary components as illustrated in the above link. 
            2. Initialized network components will be referred in `forward` method 
               for constructing the dynamic computational graph.

        Args:
            1. in_channels (int): Number of channels in the input. 
            2. out_channels (int): Number of channels in the final output.
        """
        assert out_channels%8==0, 'out channel should be mutiplier of 8'

        ################################
        ## P4.1. Write your code here ##
        
        self.conv11_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU()
        )

        self.conv33_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(),
        )

        self.conv55_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels//8),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels//8, out_channels=out_channels//8, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channels//8),
            nn.ReLU(),
        )

        self.maxpool_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels//8),
            nn.ReLU(),
        )

        ################################

    def forward(self, x):
        """Feed-forward the data `x` through the module.

        Instructions:
            1. Construct the feed-forward computational graph as illustrated in the link 
               using the initialized components in the __init__ method.

        Args:
            1. x (torch.FloatTensor): A tensor of shape (B, in_channels, H, W).

        Returns:
            1. output (torch.FloatTensor): An output tensor of shape (B, out_channels, H, W). 

        """
        ################################
        ## P4.2. Write your code here ##
        x1 = self.conv11_branch(x)
        x2 = self.conv33_branch(x)
        x3 = self.conv55_branch(x)
        x4 = self.maxpool_branch(x)

        output = torch.cat([x1, x2, x3, x4], dim=1)

        ################################
        return output


#--------------Network Class -------------#


class MyNetworkExample(nn.Module):
    def __init__(self, nf, block_type='mlp'):
        super(MyNetworkExample, self).__init__()
        """Initialize an entire network module components.

        Instructions:
            1. Implement an algorithm that initializes necessary components. 
            2. Initialized network components will be referred in `forward` method 
               for constructing the dynamic computational graph.

        Args:        
            1. nf (int): Number of input channels for the first nn.Linear Module. An abbreviation for num_filter.
            2. block_type (str, optional): Type of blocks to use. ('mlp'. default: 'mlp')
        """
        #######################################
        ## This section is an example.       ##
        if block_type == 'mlp':
            block = MLPBlock
            # Since shape of input image is 3 x 32 x 32, the size of flattened input is 3*32*32. 
            self.mlp = block(3*32*32, nf)
            self.fc = nn.Linear(nf, 10)
        else:
            raise Exception(f"Wrong type of block: {block_type}.Expected : mlp")
        #######################################

    def forward(self, x):
        """Feed-forward the data `x` through the network.

        Instructions:
            1. Construct the feed-forward computational graph as illustrated in the link 
               using the initialized network components in __init__ method.
        Args:
            1. x (torch.FloatTensor): An image tensor of shape (B, 3, 32, 32).

        Returns:
            1. output (torch.FloatTensor): An output tensor of shape (B, 10). 
        """
        #######################################
        ## This section is an example.       ##
        output = self.mlp(x.view(x.size()[0], -1))
        output = self.fc(output)
        return output
        #######################################


class MyNetwork(nn.Module):
    def __init__(self, nf, block_type='conv', num_blocks=[1, 1, 1]):
        super(MyNetwork, self).__init__()
        """Initialize an entire network module components.

        Illustration: https://docs.google.com/drawings/d/1L8PYO8A1EL4BN4bzTWH4ygr-WiS7NDeFz7P1PkhBZwE/edit?usp=sharing

        Instructions:
            1. Implement an algorithm that initializes necessary components as illustrated in the above link. 
            2. Initialized network components will be referred in `forward` method 
               for constructing the dynamic computational graph.

        Args:
            1. nf (int): Number of output channels for the first nn.Conv2d Module. An abbreviation for num_filter.
            2. block_type (str, optional): Type of blocks to use. ('conv' | 'resPlain' | 'resBottleneck' | 'inception'. default: 'conv')
            3. num_blocks (list or tuple, optional): A list or tuple of length 3. 
               Each item at i-th index indicates the number of blocks at i-th Layer.  
               (default: [1, 1, 1])
        """
        
        self.block_type = block_type

        # Define blocks according to block_type
        if self.block_type == 'conv':
            block = ConvBlock
            block_args = lambda x: (x, x, 3, 1, 1)
        elif self.block_type == 'resPlain':
            block = ResBlockPlain
            block_args = lambda x: (x,)
        elif self.block_type == 'resBottleneck':
            block = ResBlockBottleneck
            block_args = lambda x: (x, x//2)
        elif self.block_type == 'inception':
            block = InceptionBlock
            block_args = lambda x: (x, x)
        else:
            raise Exception(f"Wrong type of block: {block_type}")

        # Define block layer by stacking multiple blocks. 
        # You don't need to modify it. Just use these block layers in forward function.  
        self.block1 = nn.Sequential(*[block(*block_args(nf)) for _ in range(num_blocks[0])])
        self.block2 = nn.Sequential(*[block(*block_args(nf*2)) for _ in range(num_blocks[1])])
        self.block3 = nn.Sequential(*[block(*block_args(nf*4)) for _ in range(num_blocks[2])])

        ################################
        ## P5.1. Write your code here ##
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=nf, out_channels=nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=nf*2, out_channels=nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.AdaAvgPool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(nf*4, 10)

        ################################

    def forward(self, x):
        """Feed-forward the data `x` through the network.

        Instructions:
            1. Construct the feed-forward computational graph as illustrated in the link 
               using the initialized network components in __init__ method.
        Args:
            1. x (torch.FloatTensor): An image tensor of shape (B, 3, 32, 32).

        Returns:
            1. output (torch.FloatTensor): An output tensor of shape (B, 10). 
        """

        #######################################################################
        ## P5.2. Write your code here                                        ##
        ## Hint : use self.block1, self.block2, self.block3 for block layers ##
        x1 = self.conv_layer1(x)
        x2 = self.block1(x1)
        x3 = self.conv_layer2(x2)
        x4 = self.block2(x3)
        x5 = self.conv_layer3(x4)
        x6 = self.block3(x5)
        x7 = self.AdaAvgPool(x6)
        x8 = self.flatten(x7)
        output = self.fc(x8)
        
        #######################################################################
        return output