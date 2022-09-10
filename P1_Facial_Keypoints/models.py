## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import pdb

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # (W-F)/s = (224-3)/1 +1= 222
        self.conv1 = nn.Conv2d(1, 32, 5)  # output dim = (32, 222, 222)
        self.pool = nn.MaxPool2d(2, 2) # (32, 111, 111)
        # (111-5)/1 + 1 = 97
        self.conv2 = nn.Conv2d(32, 20, 5)  # (20, 106, 106)-> pool (20, 53, 53)
        self.fc1 = nn.Linear(20*53*53, 500)
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(500, 136)

        
        ## Note that among the layers to add, consider including:
            # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batchnormalization) to avoid overfitting


        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
      
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)

        # two linear layers with dropout in between
        #pdb.set_trace()

        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
