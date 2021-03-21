import torch

import numpy as np
import torch.nn as nn
import gym
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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
        #    self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.position = 0
        for m in self.memory:
            del m
        del self.memory

    def __len__(self):
        return len(self.memory)


# Hyperparameters
BATCH_SIZE = 128
MEMORY_CAPACITY = 1000
NUM_TRAINING_EPISODES = 50
MAX_EPISODE_TIME = 1000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# RL Model Class for Training and Testing


class RL_Model():

    # Creates a new RL Model, given a Gym Environment,
    # NeuralNetwork Class and optional Action Space
    def __init__(self, env, nn, action_space):
        # set env
        self.env = env

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
        self.memory = ReplayMemory(MEMORY_CAPACITY)

        # variables for training
        self.steps_taken = 0

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
        screen = screen.transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return resize(screen).unsqueeze(0).to(self.device)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_taken / EPS_DECAY)
        self.steps_taken += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(len(self.action_space))]], device=self.device, dtype=torch.long)

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

    def train(self, num_episodes=NUM_TRAINING_EPISODES):

        for i_ep in range(num_episodes):
            print("TRAINING ON EPISODE: " + str(i_ep))

            # reset env and state
            self.env.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen - last_screen

            for t in count():
                # print(t)
                # plt.imshow(self.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation = 'none')
                # plt.draw()
                # plt.pause(1e-3)
                # select an action from the state
                action = self.select_action(state)
                _, reward, done, _ = self.env.step(
                    self.action_space[action.item()])
                reward = torch.tensor([reward], device=self.device)

                # observe new state
                last_screen = current_screen
                current_screen = self.get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one optimization step on the target network
                self.optimize_model()

                # Check if past step limit
                if t > MAX_EPISODE_TIME:
                    break

            # Update target network, copy all weights and biases
            if i_ep % TARGET_UPDATE == 0:
                self.target.load_state_dict(self.policy.state_dict())

    def generate_policy_video(self, filename="rl_model", num_episodes=1, fps=30, max_episode_time=MAX_EPISODE_TIME):
        filename = filename + ".mp4"
        with imageio.get_writer(filename, fps=fps) as video:
            for _ in range(num_episodes):
                time_step = self.env.reset()
                done = False
                video.append_data(self.env.render(mode="rgb_array"))

                for i in range(max_episode_time):
                    last_screen = self.get_screen()
                    current_screen = self.get_screen()
                    state = current_screen - last_screen

                    action = self.select_action(state)
                    _, _, done, _ = self.env.step(
                        self.action_space[action.item()])
                    video.append_data(self.env.render(mode="rgb_array"))

                    if(done):
                        break
        return True


env = gym.make('CarRacing-v0').unwrapped

discrete_action_space = {"turn_left": [-1, 0, 0], "turn_right": [1, 0, 0], "go": [0, 1, 0], "go_left": [-1,
                                                                                                        1, 0], "go_right": [1, 1, 0], "brake": [0, 0, 1], "brake_left": [-1, 0, 1], "brake_right": [1, 0, 1]}
d_actions = list(discrete_action_space.values())

model = RL_Model(env, DQN, d_actions)

model.generate_policy_video("rl_progress_ep_" + str(0))

for i in range(1, 11):
    model.train(40)
    model.generate_policy_video("rl_progress_ep_" + str(i*40))
