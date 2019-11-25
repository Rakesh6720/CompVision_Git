import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-3)/1 + 1 = 222

        #CONVOLUTION LAYER 1
        self.conv1 = nn.Conv2d(1, 32, 3)
        # output volume spatial size: (32, 222, 222)

        #maxpool layer
        # pool kernel=2, pool stride=2        
        self.pool = nn.MaxPool2d(2, 2)
        # after one pool layer, output becomes (32, 111, 111)

        self.drop1 = nn.Dropout(p=0.1)

        #CONVOLUTION LAYER 2
        # input volume spatial size: (32, 111, 111)
            ## 32 feature maps outputted from ConvLayer_1
            ## square dimensions are 111 x 111 because divided by 2 by pool layer
            ## kernel size remains 3x3
        self.conv2 = nn.Conv2d(32, 64, 3)
        #output size = (W-F)/S+1 = (111-3)/1 + 1 = 109
        # output volume spatial size: (64, 109, 109)
        # output volume spatial size after pooling: (64, 54, 54)
            ## will i need to add zero padding to this layer to make sure the output pooling layer is integer?
            ## 1 epoch ran with 2 layers so, apparently not...
        self.drop2 = nn.Dropout(p=0.2)
        #after another pool it becomes (64, 54, 54)

        #CONVOLUTION LAYER 3
        # input volume spatial size: (64, 54, 54)
        self.conv3 = nn.Conv2d(64, 128, 3)
        #output dimensions = (W-F)/S+1 = (54-3)/1+1 = 52
        #output volume spatial size: (128, 52, 52)
        #output volume spatial size after pooling: (128, 26, 26)
        self.drop3 = nn.Dropout(p=0.3)

        #CONVOLUTION LAYER 4
        # input size = (128, 26, 26)
        # output size = (W-F)/S+1 = (26-3)1+1 = 24
        self.conv4 = nn.Conv2d(128, 256, 3)
        # output volume spatial size: (256, 24, 24)
        # output volume spatial size after pooling: (256, 12, 12)
        self.drop4 = nn.Dropout(p=0.4)

        #CONVOLUTION LAYER 5
        # input size = (256, 12, 12)
        # output size = (W-F)/S+1 = (12-3)/1+1 = 10
        self.conv5 = nn.Conv2d(256, 512, 3)
        # output volume spatial size: (512, 10, 10)
        # output volume spatial size after pooling: (512, 5, 5)

        # Flattening the vector = 512*5*5 = 12800
        self.fc1 = nn.Linear(12800, 512)
        self.dropFC = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 136)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        #print(x.shape)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        #print(x.shape)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        #print(x.shape)

        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop4(x)
        #print(x.shape)

        x = self.pool(F.relu(self.conv5(x)))
        #print(x.shape)

        # Flatten inputs for Linear Layer
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropFC(x)
        x = self.fc2(x)

        return x
        