import torch.nn as nn
import torch
import torch.nn.functional as F


class DQN_Deep2(nn.Module):

    def __init__(self, w, h, frame_stacks, outputs):
        super(DQN_Deep2, self).__init__()

        self.conv1=nn.Conv2d(3*frame_stacks, 16, kernel_size=5, stride=2)
        self.bn1=nn.BatchNorm2d(16)
        self.max1= nn.MaxPool2d(2)
        self.relu1=nn.ReLU()
        self.conv2= nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2=nn.BatchNorm2d(32)
        self.max2= nn.MaxPool2d(2)
        self.relu2= nn.ReLU()
        self.conv3= nn.Conv2d(32, 32, kernel_size=1, stride=2)
        self.bn3= nn.BatchNorm2d(32)
        self.avg1= nn.AvgPool2d(1)
        self.relu2= nn.ReLU()

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32

        
        self.head = nn.Linear(192,outputs)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x= self.bn1(x)
        #print(x.shape)
        x=self.max1(x)
        #print(x.shape)
        x=self.relu1(x)
        #print(x.shape)
        x=self.conv2(x)
        #print(x.shape)
        x=self.bn2(x)
        #print(x.shape)
        #x=self.max2(x)
        #print(x.shape)
        x=self.relu2(x)
        #print(x.shape)
        x=self.conv3(x)
        #print(x.shape)
        x=self.bn3(x)
        #print(x.shape)
        x=self.avg1(x)
        #print(x.shape)
        x=self.relu2(x)
        #print(x.shape)
#
        x=x.view(x.size(0), -1)
        #print(x.shape)

        
        return  self.head(x)
