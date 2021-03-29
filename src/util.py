
from collections import namedtuple
import torch

import gc
import numpy as np
import torch.nn as nn
import gym
import gc

from itertools import count
import random
import math
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import imageio
import base64
import psutil
import os
import util as util

def remove_outliers(x, constant):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * constant
    quartile_set = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []

    for y in a.tolist():
        if y >= quartile_set[0] and y <= quartile_set[1]:
            resultList.append(y)
        else:
            resultList.append(0)
    return resultList

class DQN(nn.Module):
    
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        try:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            return self.head(x.view(x.size(0), -1))
        except:
            print(x)
            return self.head(x.view(x.size(0), -1))


Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # if len(self.memory) > self.capacity:
        #    self.clear()

        # self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        # print("CLEARING MEMORY")
        self.position = 0
        for m in self.memory:
            del m
        del self.memory
        self.memory = []

    def __len__(self):
        return len(self.memory)
