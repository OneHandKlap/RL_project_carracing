# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch

import gc
import numpy as np
import torch.nn as nn
import gym
import gc
from collections import namedtuple
from itertools import count
import random
import math
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
import torch.optim as optim
import imageio
import base64
import psutil
import os

from torchsummary import summary ######## pip install torch-summary ######## DELETE ########


def memory_used():
    return psutil.Process(os.getpid()).memory_info().rss * 1e-6  # To megabyte

'''
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
'''

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


# %%
# Hyperparameters

BATCH_SIZE = 128
MEMORY_CAPACITY = 7000
NUM_TRAINING_EPISODES = 50
MAX_EPISODE_TIME = 1000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
ENV_CLEAR = 5


# %%
######### RESNET #########
# SOURCE: https://medium.com/the-owl/extracting-features-from-an-intermediate-layer-of-a-pretrained-model-in-pytorch-c00589bda32b

#rn18 = models.resnet18(pretrained=True) ##### Expects 4-dimensional input for 4-dimensional weight [64, 3, 7, 7]
#children_counter = 0
#for n,c in rn18.named_children():
#    print("Children Counter: ",children_counter," Layer Name: ",n,)
#    children_counter+=1

class rn18_feature_extractor(nn.Module):
    def __init__(self,output_layer = None):
        super().__init__()
        self.pretrained = models.resnet18(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.dummy_var= None ##### freeing up some space
        
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None ##### freeing up some space

    def forward(self,x):
        x = self.net(x)
        return x


rn18_fe = rn18_feature_extractor(output_layer='maxpool')

img = torch.Tensor(1, 3, 60, 40) ##### returns torch.Size([1, 64, 15, 10])
features = rn18_fe(img)
print(features.shape)
#rn18._modules


# %%
# summary(rn18_fe, input_size=(1, 3, 60, 40))
# print('*'*80)
# rn18_fe._modules


# %%
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        """
            :param h is the height of the image (in this case 60)
            :param w is the height of the image (in this case 40)
        """
        super(DQN, self).__init__()
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        #self.bn1 = nn.BatchNorm2d(16)
        #self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        #self.bn2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        #self.bn3 = nn.BatchNorm2d(32)
        
        
        ##### DELETE #####
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        #--------------- USING BASIC ARCHITECTURE ---------------
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2)
        #self.bn1 = nn.BatchNorm2d(64)
        #self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        #self.bn2 = nn.BatchNorm2d(128)
        #self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        #self.bn3 = nn.BatchNorm2d(256)
        #linear_input_size = conv2d_size_out(conv2d_size_out(conv2d_size_out(w))) * conv2d_size_out(conv2d_size_out(conv2d_size_out(h))) * 256
        #print(linear_input_size)
        #self.dense1 = nn.Linear(linear_input_size, 1024)
        #self.dense2 = nn.Linear(1024, 512)
        #self.head = nn.Linear(512, outputs)
        
        
        #--------------- USING PRE-TRAINED RESNET18 ---------------
        self.rn18_fe = rn18_feature_extractor(output_layer='maxpool')
        self.conv1 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        #self.conv3 = nn.Conv2d(256, 512, kernel_size=5, stride=1)
        linear_input_size = conv2d_size_out(conv2d_size_out(conv2d_size_out(w))) * conv2d_size_out(conv2d_size_out(conv2d_size_out(h))) * 64
        #print(linear_input_size)
        self.dense1 = nn.Linear(linear_input_size, 512)
        self.dense2 = nn.Linear(512, 256)
        self.head = nn.Linear(256, outputs)
        
        ##### DELETE #####

        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        #def conv2d_size_out(size, kernel_size=5, stride=2):
        #    return (size - (kernel_size - 1) - 1) // stride + 1
        #convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        #convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        #linear_input_size = convw * convh * 32 ##### linear_input_size = 256
        #self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        try:
            #--------------- USING BASIC ARCHITECTURE ---------------
            #x = F.relu(self.bn1(self.conv1(x)))
            #print('conv1.shape = ', x.shape)
            #x = F.relu(self.bn2(self.conv2(x)))
            #print('conv2.shape = ', x.shape)
            #x = F.relu(self.bn3(self.conv3(x)))
            #print('conv3.shape = ', x.shape)
            
            ##### DELETE #####
            #x = x.view(x.size(0), -1) ##### Flattening ##### x = torch.flatten(x, 1)
            #print('view = ', x.shape)
            #x = F.relu(self.dense1(x))
            #print('dense1.shape = ', x.shape)
            #x = F.relu(self.dense2(x))
            #print('dense2.shape = ', x.shape)
            #output = self.head(x)
            #print('output = ', output)
            #return output
            
            
            #--------------- USING PRE-TRAINED RESNET18 ---------------
            #print('init.shape =',x.shape)
            x = self.rn18_fe.forward(x) ##### output shape is torch.Size([1, 64, 15, 10])
            #print('resnet18.shape =',x.shape)
            x = F.relu(self.bn1(self.conv1(x)))
            #print('conv1.shape = ', x.shape)
            x = F.relu(self.bn2(self.conv2(x)))
            
            #print('conv2.shape = ', x.shape)
            #x = F.relu(self.bn3(self.conv3(x)))
            #print('conv3.shape = ', x.shape)
            
            x = x.view(x.size(0), -1)
            #print('view = ', x.shape)
            x = F.relu(self.dense1(x))
            #print('dense1.shape = ', x.shape)
            x = F.relu(self.dense2(x))
            #print('dense2.shape = ', x.shape)
            output = self.head(x)
            #print('output = ', output)
            return output
            
            ##### DELETE #####
            
            #return self.head(x.view(x.size(0), -1)) ####### [batch_size, Ch*H*W] i.e. FLATENNING
        except:
            print(x)
            print('--------------- EXCEPTION ---------------')
            #return self.head(x.view(x.size(0), -1))


# %%
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# %%
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


# %%
# RL Model Class for Training and Testing


class RL_Model():

    # Creates a new RL Model, given a Gym Environment,
    # NeuralNetwork Class and optional Action Space
    def __init__(self, env, nn, action_space, env_string=None):
        # set env
        self.env = env
        if env_string:
            self.env_string = env_string
            self.env = gym.make(env_string).unwrapped

        # if gpu is to be used
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        print(self.device)
        

        # initialize screen and nn
        self.env.reset()
        _, _, screen_height, screen_width = self.get_screen().shape

        # set action space
        if(action_space):
            self.action_space = action_space
        else:
            self.action_space = env.action_space

        # policy net
        self.policy = nn(screen_height, screen_width,
                         len(self.action_space)).to(self.device) ##### Height = 60, Width = 40
        summary(self.policy, input_size=(3, screen_height, screen_width)) ###### DELETE ######

        # target net
        self.target = nn(screen_height, screen_width,
                         len(self.action_space)).to(self.device)
        summary(self.target, input_size=(3, screen_height, screen_width)) ###### DELETE ######
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        # optimizer
        self.optimizer = optim.RMSprop(self.policy.parameters())

        # memory
        self.memory = ReplayMemory(MEMORY_CAPACITY)

        # variables for training
        self.steps_taken = 0

        self.episode_durations = []

    # load policy-net weights
    def load(self, path="rl_model_weights.pth"):
        checkpoint = torch.load(path)
        # policy net
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy.eval()

        # target net
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        # optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # save policy-net weights
    def save(self, path="rl_model_weights"):
        torch.save({
            # 'epoch': epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'loss': loss,
        },
            path + ".pth")

    def get_screen(self):
        screen = self.env.render(mode='rgb_array')
        screen = screen[np.ix_([x for x in range(100, 400)], [
                               x for x in range(200, 400)])]
        screen = screen.transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255 ######## screen.shape = (3x300x200)
        screen = torch.from_numpy(screen) ######## torch.Size([3, 300, 200])
        return resize(screen).unsqueeze(0).to(self.device) ####### torch.Size([1, 3, 60, 40])

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) *             math.exp(-1. * self.steps_taken / EPS_DECAY)
        self.steps_taken += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(len(self.action_space))]], device=self.device, dtype=torch.long)

    def select_deterministic_action(self, state):
        with torch.no_grad():
            da = self.policy(state).max(1)[1].view(1, 1) ###### DELETE ######
            #print('action =', da) ###### DELETE ######
            return da
            #return self.policy(state).max(1)[1].view(1, 1)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(
            lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target(
            non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # del non_final_mask
        # del non_final_next_states
        # del state_batch
        # del action_batch
        # del reward_batch
        # del loss
        # gc.collect()

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.draw()
        plt.pause(0.001)  # pause a bit so that plots are updated

    def train(self, num_episodes=NUM_TRAINING_EPISODES, epoch=1, render=False):
        self.steps_taken = 0
        acc_rewards = np.zeros(num_episodes)
        plt_color = [(random.random(), random.random(), random.random())]
        for i_ep in range(1, num_episodes+1):
            #print("EPOCH: " + str(epoch) + " EPISODE: " + str(i_ep))
            #print("MEM ALLOCATED: " + str(torch.cuda.memory_allocated()))
            #print("MEM CACHE: " + str(torch.cuda.memory_reserved()))
            #print('Ram Used: %f' % memory_used())

            # clear env
            if i_ep % ENV_CLEAR == 0 and self.env_string:
                self.env.close()
                del self.env
                self.env = gym.make(self.env_string).unwrapped

            # reset env and state
            self.env.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen - last_screen
            
            ### MEASUREMENTS ###
            cntr = 0

            for t in count():
                if render:
                    plt.imshow(self.get_screen().cpu().squeeze(
                        0).permute(1, 2, 0).numpy(), interpolation='none')
                    plt.draw()
                    plt.pause(1e-3)

                # select an action from the state
                action = self.select_action(state)
                _, reward, done, _ = self.env.step(
                    self.action_space[action.item()])
                reward = torch.tensor([reward], device=self.device)

                # increase acc rewards
                acc_rewards[i_ep-1] += reward

                # observe new state
                last_screen = current_screen
                current_screen = self.get_screen()
                # if not done:
                next_state = current_screen - last_screen
                # else:
                #    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one optimization step on the target network
                self.optimize_model()
                
                cntr += 1 #### 

                # Check if past step limit
                if t > MAX_EPISODE_TIME:
                    break

            ### PRINTING PERFORMANCE ###
            print('Episode:', i_ep, ' '*3,'Total_reward: %.2f' % acc_rewards[i_ep-1], ' '*3,                  'Average_RPE: %.2f' % (acc_rewards[i_ep-1] / cntr))
            
            
            # Update target network, copy all weights and biases
            if i_ep % TARGET_UPDATE == 0:
                self.target.load_state_dict(self.policy.state_dict())

            # plot
            #plt.title('Rewards Over Episode')
            #plt.xlabel('Episode')
            #plt.ylabel('Rewards')
            #plt.xticks(range(1, epoch * (num_episodes+1)))
            #plt.scatter(torch.tensor([i_ep + (epoch - 1) * num_episodes], dtype=torch.float), torch.tensor(
            #    [max(acc_rewards[i_ep-1], -1000)], dtype=torch.float), c=plt_color, label="Epoch " + str(epoch) if i_ep == 1 else '')
            #plt.legend()
            #plt.draw()
            #plt.pause(1e-3)

            torch.cuda.empty_cache()
            gc.collect()
        
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

        return (range(1, num_episodes+1), remove_outliers(acc_rewards,1.3))

    def generate_policy_video(self, filename="rl_model", num_episodes=1, fps=30, max_episode_time=MAX_EPISODE_TIME):
        filename = filename + ".mp4"

        


        with imageio.get_writer(filename, fps=fps) as video:

            for episode in range(num_episodes):
                time_step = self.env.reset()
                done = False
                video.append_data(self.env.render(mode="rgb_array"))
                last_screen = self.get_screen()
                current_screen = self.get_screen()
                state = current_screen - last_screen
                

                for i in range(max_episode_time):
                    action = self.select_deterministic_action(state)
                    #action = self.select_action(state)
                    _, reward, done, _ = self.env.step(
                        self.action_space[action.item()])
                    video.append_data(self.env.render(mode="rgb_array"))
                    last_screen = current_screen
                    current_screen = self.get_screen()
                    state = current_screen-last_screen


                    if(done):
                        break


        return True


# %%
# env = gym.make('CarRacing-v0').unwrapped

discrete_action_space = {"turn_left": [-1, 0, 0], "turn_right": [1, 0, 0], "go": [0, 1, 0], "go_left": [-1, 1, 0], "go_right": [1, 1, 0], "brake": [0, 0, 1], "brake_left": [-1, 0, 1], "brake_right": [1, 0, 1],"slight_turn_left": [-.3, 0, 0], "slight_turn_right": [.3, 0, 0], "slight_go": [0, .3, 0], "slight_go_left": [-.3, .3, 0], "slight_go_right": [.3, .3, 0], "slight_brake": [0, 0, .3], "slight_brake_left": [-.3, 0, .3], "slight_brake_right": [.3, 0, .3]}
d_actions = list(discrete_action_space.values())

model = RL_Model(gym.make('CarRacing-v0').unwrapped,
                 DQN, d_actions, 'CarRacing-v0')

# model.generate_policy_video("rl_progress_ep_" + str(0))


# %%
for i in range(1, 20):
    ep, rewards = model.train(
        5, render=False, epoch=i)
    model.save("rl_progress_ep_" + str(i * 5))
    model.generate_policy_video("rl_progress_ep_" + str(i*5))
    plt.title('Rewards Over Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.scatter([x for x in range(len(rewards))], rewards)
    plt.legend()
    plt.savefig("rl_progress_ep_"+str(i*5))



# %%



