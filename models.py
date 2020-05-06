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
        ## output size = (W-F)/S +1 = (224-4)/1 +1 = 221
        # the output Tensor for one image, will have the dimensions: (32, 221, 221)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 4)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output Tensor for one image, will have the dimensions: (64, 108, 108)
        # after one pool layer, this becomes (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # third conv layer
        ## output size = (W-F)/S +1 = (54-2)/1 +1 = 53
        # the output Tensor for one image, will have the dimensions: (128, 53, 53)
        # after one pool layer, this becomes (128, 26, 26) # round down
        self.conv3 = nn.Conv2d(64, 128, 2)
        
        # 128 outputs * the 25 x 25 filtered / pooled map size
        self.fc1 = nn.Linear(128*26*26, 1000)
        
        # dropout with p=0.5
        self.fc1_drop = nn.Dropout(p=0.5)
        
        # 128 outputs * the 25 x 25 filtered / pooled map size
        self.fc2 = nn.Linear(1000, 1000)
        
        # dropout with p=0.5
        self.fc2_drop = nn.Dropout(p=0.5)
        
        # create the last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        self.fc3 = nn.Linear(1000, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # 3 conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        #print('shape after conv3 {}'.format(x.shape))
        
        # flattend for linear layer
        x = x.view(x.size(0), -1)
        
        #print('shape after flatten {}'.format(x.shape))
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
