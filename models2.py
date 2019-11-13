## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class KeshNet(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 + 1 = 220
        # batch size = 10
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        self.pool = nn.MaxPool2d(2, 2)
                  
        # second convolution layer: 32 inputs, 64 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output Tensor will have dimensions: (64, 106, 106)
        # after another pool layer this becomes (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # batch norm
        #self.conv2_bn = nn.BatchNorm2d(64)
        
        # 64 outputs * the 5x5 filtered/pooled map size
        # 68 output channels (for the 68 keypoint pairs)
        self.fc1 = nn.Linear(179776, 272)
        
        # batch norm
        #self.fc1_bn = nn.BatchNorm1d(272)
        
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(272, 136)    

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        # adding batch norm 2d here
        #x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
              
        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
         
        # one linear layer
        # adding batch norm 1d here
        #x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
