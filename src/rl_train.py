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
import torch.optim as optim
import imageio
import base64
import psutil
import os
import util as util


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

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))



# Hyperparameters
# BATCH_SIZE = 128
# MEMORY_CAPACITY = 7000
# NUM_TRAINING_EPISODES = 50
# MAX_EPISODE_TIME = 10
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
# TARGET_UPDATE = 10
# ENV_CLEAR = 5

# RL Model Class for Training and Testing


class RL_Model():

    # Creates a new RL Model, given a Gym Environment,
    # NeuralNetwork Class and optional Action Space
    def __init__(self, env, nn, action_space, env_string=None, batch_size=128,memory_capacity=7000,num_training_episodes=50,max_episode_time=100,gamma=0.999,eps_start=0.9,eps_end=0.05,eps_decay=200,target_update=10,env_clear=5):
        # set env
        self.env = env
        self.batch_size=batch_size
        self.memory_capacity=memory_capacity
        self.num_training_episodes=num_training_episodes
        self.max_episode_time=max_episode_time
        self.gamma=gamma
        self.eps_start= eps_start
        self.eps_end=eps_end
        self.eps_decay=eps_decay
        self.target_update=target_update
        self.env_clear=env_clear
        if env_string:
            self.env_string = env_string
            self.env = gym.make(env_string).unwrapped

        # if gpu is to be used
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

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
                         len(self.action_space)).to(self.device)

        # target net
        self.target = nn(screen_height, screen_width,
                         len(self.action_space)).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        # optimizer
        self.optimizer = optim.RMSprop(self.policy.parameters())

        # memory
        self.memory = util.ReplayMemory(self.memory_capacity)

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
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return resize(screen).unsqueeze(0).to(self.device)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_taken / self.eps_decay)
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
            return self.policy(state).max(1)[1].view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

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
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target(
            non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch

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

    def train(self, epoch=1, render=False):
        self.steps_taken = 0
        acc_rewards = np.zeros(self.num_training_episodes)
        plt_color = [(random.random(), random.random(), random.random())]
        for i_ep in range(1, self.num_training_episodes+1):
            print("EPOCH: " + str(epoch) + " EPISODE: " + str(i_ep))
            print("MEM ALLOCATED: " + str(torch.cuda.memory_allocated()))
            print("MEM CACHE: " + str(torch.cuda.memory_reserved()))
            print('Ram Used: %f' % memory_used())

            # clear env
            if i_ep % self.env_clear == 0 and self.env_string:
                self.env.close()
                del self.env
                self.env = gym.make(self.env_string).unwrapped

            # reset env and state
            self.env.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen - last_screen

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

                # Check if past step limit
                if t > self.max_episode_time:
                    break

            # Update target network, copy all weights and biases
            if i_ep % self.target_update == 0:
                self.target.load_state_dict(self.policy.state_dict())

            torch.cuda.empty_cache()
            gc.collect()

        return (range(1, self.num_training_episodes+1), util.remove_outliers(acc_rewards, 1.3))

    def generate_policy_video(self, filename="rl_model", num_episodes=1, fps=30):
        filename = filename + ".mp4"

        with imageio.get_writer(filename, fps=fps) as video:

            for episode in range(num_episodes):
                time_step = self.env.reset()
                done = False
                video.append_data(self.env.render(mode="rgb_array"))
                last_screen = self.get_screen()
                current_screen = self.get_screen()
                state = current_screen - last_screen

                for i in range(self.max_episode_time):
                    action = self.select_deterministic_action(state)
                    # action = self.select_action(state)
                    _, reward, done, _ = self.env.step(
                        self.action_space[action.item()])
                    video.append_data(self.env.render(mode="rgb_array"))
                    last_screen = current_screen
                    current_screen = self.get_screen()
                    state = current_screen-last_screen

                    if(done):
                        break

        return True


# env = gym.make('CarRacing-v0').unwrapped

discrete_action_space = {"turn_left": [-1, 0, 0], "turn_right": [1, 0, 0], "go": [0, 1, 0], "go_left": [-1, 1, 0], "go_right": [1, 1, 0], "brake": [0, 0, 1], "brake_left": [-1, 0, 1], "brake_right": [1, 0, 1], "slight_turn_left": [-.3,0, 0], "slight_turn_right": [.3, 0, 0], "slight_go": [0, .3, 0], "slight_go_left": [-.3, .3, 0], "slight_go_right": [.3, .3, 0], "slight_brake": [0, 0, .3], "slight_brake_left": [-.3, 0, .3], "slight_brake_right": [.3, 0, .3]}

d_actions = list(discrete_action_space.values())

model = RL_Model(gym.make('CarRacing-v0').unwrapped,
                 util.DQN, d_actions, 'CarRacing-v0',)

# model.generate_policy_video("rl_progress_ep_" + str(0))


for i in range(1, 20):
    ep, rewards = model.train(render=False, epoch=i)
    model.save("rl_progress_ep_" + str(i * 5))
    model.generate_policy_video("rl_progress_ep_" + str(i*5))
    plt.title('Rewards Over Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.scatter([x for x in range(len(rewards))], rewards)
    plt.legend()
    plt.savefig("rl_progress_ep_"+str(i*5))
