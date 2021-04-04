#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from utils.general import get_logger
from utils.test_env import EnvTest
from q2_schedule import LinearExploration, LinearSchedule
from q3_linear_torch import Linear
import copy

from configs.q4_nature import config


# In[2]:


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history
        MAKE SURE YOU USE THESE VARIABLES FOR INPUT SIZE

        Each network has the following architecture (see th nature paper for more details):
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
            - Conv2d with 32 8x8 filters and stride 4 + ReLU activation
            - Conv2d with 64 4x4 filters and stride 2 + ReLU activation
            - Conv2d with 64 3x3 filters and stride 1 + ReLU activation
            - Flatten
            - Linear with output 512. What is the size of the input?
                you need to calculate this img_height, img_width, and number of filter.
            - Relu
            - Linear with 512 input and num_actions outputs

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        To simplify, we specify the paddings as:
            ((stride - 1) * img_height - stride + filter_size) // 2
        Make sure you follow use this padding for every layer

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
            3. If you use OrderedDict, make sure the keys for the the layers are:
                - "0", "2", "4" for three Conv2d layers
                - "7" for the first Linear layer
                - "9" for the final Linear layer
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = len(self.env.action_space)
        strides = np.array([4, 2, 1])  # The stride size for every conv2d layer
        filter_sizes = np.array([8, 4, 3])  # The filter size for every conv2d layer
        numb_filters = np.array([32, 64, 64])  # number of filters for every conv2d layer
        ##############################################################
        ################ YOUR CODE HERE - 25-30 lines lines ################
                ################ HELPER FUNCTIONS ################
        def calc_padding(size, stride, filter_size):
            """
                returns the padding for a conv2d layer
                args:
                    size: (int) image height or width
                    stride: (int) stride value in the conv2d layer
                    filter_size: (int) filter size of the conv2d layer
                returns:
                    (int) padding value
            """
            return ((stride-1) * size - stride + filter_size) // 2

        def calc_output_shape(h_in, w_in, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1)):
            """
                returns the output shape of an image from a conv2d layer
                args:
                    h_in: (int) image height
                    2_in: (int) image width
                    kernel_size: (int, int) filter size of the conv2d layer
                    stride: (int, int) stride value of the conv2d layer
                    padding: (int, int) padding value of the conv2d layer
                    dilation: (int, int) dilation value of the conv2d layer
                returns:
                    a tuple of resulting height and width feeded to a conv2d layer
            """
            h_out = np.floor((h_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
            w_out = np.floor((w_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
            return (h_out, w_out)
        
        def calc_linear_input_size(last_conv2d_ch_out, img_height, img_width, layers_info):
            """
                returns the flattened value of an image when passed through a few conv2d layers
                args:
                    last_conv2d_ch_out: (int) the output channel of the last conv2d layer
                    img_height: (int) image height
                    img_width: (int) image width
                    layers_info: (OrderedDict) with the following members
                        - paddings: (list of tuples) padding used in each conv2d layer
                        - dilations: (list of tuples) dilations used in each conv2d layer
                        - kernel_sizes: (list of tuples) filter sizes used in each conv2d layer
                        - strides: (list of tuples) strides used in each conv2d layer
                        - num_layers: (int) number of conv2d layers used in the network
                returns:
                    (int) the input size that passes through the first linear layer
            """
            heights = [img_height] + [0]*layers_info['num_layers']
            widths = [img_width] + [0]*layers_info['num_layers']
            for i in range(layers_info['num_layers']):
                (heights[i+1], widths[i+1]) = calc_output_shape(heights[i], widths[i], layers_info['kernel_sizes'][i],                                layers_info['strides'][i], layers_info['paddings'][i], layers_info['dilations'][i])
            return np.uint32(last_conv2d_ch_out * heights[len(heights)-1] * widths[len(widths)-1])
        
        def create_dqn_model(input_channel, out_channels, filter_sizes, strides, paddings):
            """
                returns a sequential DQN model
                Args:
                    input_channel: (int) the input channel of the first conv2d layer
                    out_channels: (list of int) output channel of each layer
                    strides: (int list) a sequential list of the strides used in conv2d layers
                    filter_sizes: (int list) a sequential list of filter sizes of conv2d layers
                    num_filters: (int list) a sequential list of number of filters used in conv2d layers
                    paddings: (list of tuples) paddings of each conv2d layer
                returns:
                    Deep-Q-Network
            """
            model = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(input_channel, out_channels[0], filter_sizes[0], stride=strides[0], padding=paddings[0])),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(out_channels[0], out_channels[1], filter_sizes[1], stride=strides[1], padding=paddings[1])),
                ('relu2', nn.ReLU()),
                ('conv3', nn.Conv2d(out_channels[1], out_channels[2], filter_sizes[2], stride=strides[2], padding=paddings[2])),
                ('relu3', nn.ReLU()),
                ('flatten', nn.Flatten()),
                ('linear', nn.Linear(out_channels[3], out_channels[4])),
                ('relu4', nn.ReLU()),
                ('output', nn.Linear(out_channels[4], out_channels[5]))
            ]))
            print(model)
            return model
                ################ HELPER FUNCTIONS ENDS ################
            
        # Setting up the variables
        padding1 = calc_padding(img_height, strides[0], filter_sizes[0])
        padding2 = calc_padding(img_height, strides[1], filter_sizes[1])
        padding3 = calc_padding(img_height, strides[2], filter_sizes[2])
        input_channel = n_channels * self.config.state_history
        input_size = img_height * img_width * input_channel
        layers_info = OrderedDict([
            ('paddings',  [(padding1,padding1), (padding2,padding2), (padding3,padding3)]),
            ('dilations', [(1,1)]*len(numb_filters)),
            ('kernel_sizes', [(filter_sizes[0],filter_sizes[0]), (filter_sizes[1],filter_sizes[1]), (filter_sizes[2],filter_sizes[2])]),
            ('strides', [(strides[0],strides[0]), (strides[1],strides[1]), (strides[2],strides[2])]),
            ('num_layers', len(numb_filters))
        ])
        linear_input_size = calc_linear_input_size(numb_filters[2], img_height, img_width, layers_info)
        out_channels = np.concatenate((numb_filters, [linear_input_size, 512, num_actions]))
        
        # initializing the networks
        self.q_network = create_dqn_model(input_channel, out_channels, filter_sizes, strides, layers_info['paddings'])
        self.target_network = create_dqn_model(input_channel, out_channels, filter_sizes, strides, layers_info['paddings'])

        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None
        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ################
        state = state.transpose(1,3)
        if network == 'q_network':
            out = self.q_network(state)
        else:
            out = self.target_network(state)
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


# In[3]:


if __name__ == '__main__':
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)


# In[ ]:




