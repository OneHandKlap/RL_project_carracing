from agents.dqn_agent import DQN_Agent
#from config.adam_config import config
from config.baseline import config 


from models.dqn_deep import DQN_Deep
from models.dqn_deep2 import DQN_Deep2
from models.dqn_basic import DQN_Basic

import matplotlib.pyplot as plt
import torch
import psutil
import os
import numpy as np

from util.generate_video import generate_policy_video

from wrappers.memory_wrapper import MemoryWrapper

import gym



env = MemoryWrapper(lambda: gym.make("CarRacing-v0"))
agent1=DQN_agent(env, DQN_basic, config1)
agent1.load("/models/pretrained/basic1/rl_model_weights_10_100.pth")
generate_policy_video(agent,env,filename=f"test")