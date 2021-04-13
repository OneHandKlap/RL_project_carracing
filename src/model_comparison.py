from agents.dqn_agent import DQN_Agent
from config.adam_config import config as config1
from config.baseline import config as config2


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
agent1=DQN_Agent(env, DQN_Basic, config1)
agent1.load("C:\\Users\\pabou\\Documents\\GitHub\\CPS824-RL\\CPS824-Project\\src\\models\\pretrained\\basic1\\rl_model_weights_10_100.pth")
agent2=DQN_Agent(env, DQN_Basic, config2)
agent2.load("C:\\Users\\pabou\\Documents\\GitHub\\CPS824-RL\\CPS824-Project\\src\\models\\pretrained\\basic2\\rl_model_weights_20_100.pth")
agent3=DQN_Agent(env, DQN_Deep, config2)
agent3.load("C:\\Users\\pabou\\Documents\\GitHub\\CPS824-RL\\CPS824-Project\\src\\models\\pretrained\\deeper_maxpool_2conv\\rl_model_weights_10_100.pth")
agent4=DQN_Agent(env, DQN_Deep2,config2)
agent4.load("C:\\Users\\pabou\\Documents\\GitHub\\CPS824-RL\\CPS824-Project\\src\\models\\pretrained\\deepesst_maxpool_avgpool_3conv\\rl_model_weights_16_100.pth")

fig, axs= plt.subplots(2,2)
x=[x for x in range(30)]
# agent1_rewards, agent1_steps= agent1.test(env)
# agent2_rewards, agent2_steps= agent2.test(env)
# agent3_rewards, agent3_steps= agent3.test(env)
# agent4_rewards, agent4_steps= agent4.test(env)
# np.savetxt("rewards.txt",([agent1_rewards,agent2_rewards,agent3_rewards,agent4_rewards]))
# np.savetxt("steps.txt",([agent1_steps,agent2_steps,agent3_steps,agent4_steps]))

agent1_rewards,agent2_rewards,agent3_rewards,agent4_rewards = np.loadtxt("rewards.txt")
agent1_steps,agent2_steps,agent3_steps,agent4_steps=np.loadtxt("steps.txt")
# print(np.mean(agent4_rewards))


axs[0,0].plot(x,agent1_rewards/agent1_steps, label="basic")
#m,b=np.polyfit(x,agent1_rewards,1)
axs[0,0].axhline(np.mean(agent1_rewards/agent1_steps), color='red')
axs[0,0].set_title('basic dqn')
axs[0,1].plot(x,agent2_rewards/agent2_steps, label="basic with LR")
#m,b=np.polyfit(x,agent2_rewards,1)
axs[0,1].axhline(np.mean(agent2_rewards/agent2_steps), color='red')
axs[0,1].set_title('basic dqn with LR scheduler')
axs[1,0].plot(x,agent3_rewards/agent3_steps, label="maxpool 2 conv")
#m,b=np.polyfit(x,agent3_rewards,1)
axs[1,0].axhline(np.mean(agent3_rewards/agent3_steps), color='red')
axs[1,0].set_title('maxpool 2 conv layers')
axs[1,1].plot(x,agent4_rewards/agent4_steps, label = "max/avg 3 conv")
#m,b=np.polyfit(x,agent4_rewards,1)
axs[1,1].axhline(np.mean(agent4_rewards/agent4_steps), color='red')
axs[1,1].set_title('max/avg pool 3 conv layers')

#ax2.set_title("Steps#")
#ax2.plot(x,agent1_steps)
#ax2.plot(x,agent2_steps)
#ax2.plot(x,agent3_steps)
#ax2.plot(x,agent4_steps)

for ax in axs.flat:
    ax.set(xlabel='Episode', ylabel='Accuracy')
    ax.set_ylim(-1,1)

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
#plt.legend()

plt.savefig("accuracy.png")