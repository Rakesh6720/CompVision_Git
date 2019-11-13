## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        # after one pool layer, output becomes (32, 110, 110)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.drop1 = nn.Dropout(p=0.1)
        
        # second conv layer: 32 inputs
        # output size = (W-F)S + 1 = (110-5)/1 + 1 = 106
        # why do the number of filters increase 2x per convolution layer?
        # the output tensor will have size (64, 106, 106)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        self.drop2 = nn.Dropout(p=0.2)
        
        # after another maxpool layer, output becomes (64, 53, 53)
        
        
        # flattening the vector = 64*53*53 = 179776
        self.fc1 = nn.Linear(179776, 1797)
        
        self.drop3 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(1797, 136)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # one linear layer
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        
        return x
