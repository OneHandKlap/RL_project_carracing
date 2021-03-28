from Box2D.b2 import (polygonShape, circleShape)
from Box2D.b2 import fixtureDef
import Box2D
from Box2D.b2 import contactListener
import pyglet
from pyglet import gl
import torch
import time

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


# Hyperparameters
BATCH_SIZE = 128
MEMORY_CAPACITY = 7000
NUM_TRAINING_EPISODES = 50
MAX_EPISODE_TIME = 2000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
ENV_CLEAR = 5

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
        screen = screen[np.ix_([x for x in range(100, 360)], [
                               x for x in range(200, 400)])]
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

    def select_deterministic_action(self, state):
        with torch.no_grad():
            return self.policy(state).max(1)[1].view(1, 1)

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
            print("EPOCH: " + str(epoch) + " EPISODE: " + str(i_ep))
            print("MEM ALLOCATED: " + str(torch.cuda.memory_allocated()))
            print("MEM CACHE: " + str(torch.cuda.memory_reserved()))
            print('Ram Used: %f' % memory_used())

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
                    plt.pause(1e-6)

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
                if t > MAX_EPISODE_TIME or done:
                    break

            # Update target network, copy all weights and biases
            if i_ep % TARGET_UPDATE == 0:
                self.target.load_state_dict(self.policy.state_dict())

            torch.cuda.empty_cache()
            gc.collect()

        return (range(1, num_episodes+1), remove_outliers(acc_rewards, 1.3))

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

'''
Wrapper class that takes care of the memory fix
Pass in nuke_intervals -> which recreates a new environment fully
every nuke_intervals
'''


class MemoryWrapper(gym.Wrapper):
    def __init__(self, make_env, nuke_intervals=5):
        env = make_env()
        super().__init__(env)
        self.make_env = make_env

        self.num_resets = 0
        self.nuke_intervals = nuke_intervals

    def reset(self):
        if self.num_resets % self.nuke_intervals == 0:
            self.nuke()
            self.num_resets = 0

        self.env.reset()
        self.num_resets += 1

    def nuke(self):
        self.env.close()
        del self.env
        self.env = self.make_env()


ROAD_COLOR = [0.4, 0.4, 0.4]


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        self.env.on_track = True

        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData

        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        self.env.collisions[tile.tile_type] = 1

        if(tile.tile_type == "ROAD"):
            if not obj or "tiles" not in obj.__dict__:
                return
            if begin:
                obj.tiles.add(tile)
                if not tile.road_visited:
                    tile.road_visited = True
                    self.env.reward += 1000.0 / len(self.env.track)
                    self.env.tile_visited_count += 1
            else:
                obj.tiles.remove(tile)


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = False  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]


def create_track(self):
    CHECKPOINTS = 12

    # Create checkpoints
    checkpoints = []
    for c in range(CHECKPOINTS):
        noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
        alpha = 2 * math.pi * c / CHECKPOINTS + noise
        rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

        if c == 0:
            alpha = 0
            rad = 1.5 * TRACK_RAD
        if c == CHECKPOINTS - 1:
            alpha = 2 * math.pi * c / CHECKPOINTS
            self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
            rad = 1.5 * TRACK_RAD

        checkpoints.append(
            (alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
    self.road = []

    # Go from one checkpoint to another to create track
    x, y, beta = 1.5 * TRACK_RAD, 0, 0
    dest_i = 0
    laps = 0
    track = []
    no_freeze = 2500
    visited_other_side = False
    while True:
        alpha = math.atan2(y, x)
        if visited_other_side and alpha > 0:
            laps += 1
            visited_other_side = False
        if alpha < 0:
            visited_other_side = True
            alpha += 2 * math.pi

        while True:  # Find destination from checkpoints
            failed = True

            while True:
                dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(
                    checkpoints)]
                if alpha <= dest_alpha:
                    failed = False
                    break
                dest_i += 1
                if dest_i % len(checkpoints) == 0:
                    break

            if not failed:
                break

            alpha -= 2 * math.pi
            continue

        r1x = math.cos(beta)
        r1y = math.sin(beta)
        p1x = -r1y
        p1y = r1x
        dest_dx = dest_x - x  # vector towards destination
        dest_dy = dest_y - y
        # destination vector projected on rad:
        proj = r1x * dest_dx + r1y * dest_dy
        while beta - alpha > 1.5 * math.pi:
            beta -= 2 * math.pi
        while beta - alpha < -1.5 * math.pi:
            beta += 2 * math.pi
        prev_beta = beta
        proj *= SCALE
        if proj > 0.3:
            beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
        if proj < -0.3:
            beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
        x += p1x * TRACK_DETAIL_STEP
        y += p1y * TRACK_DETAIL_STEP
        track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
        if laps > 4:
            break
        no_freeze -= 1
        if no_freeze == 0:
            break

    # Find closed loop range i1..i2, first loop should be ignored, second is OK
    i1, i2 = -1, -1
    i = len(track)
    while True:
        i -= 1
        if i == 0:
            return False  # Failed
        pass_through_start = (
            track[i][0] > self.start_alpha and track[i -
                                                     1][0] <= self.start_alpha
        )
        if pass_through_start and i2 == -1:
            i2 = i
        elif pass_through_start and i1 == -1:
            i1 = i
            break
    if self.verbose == 1:
        print("BITCH Track generation: %i..%i -> %i-tiles track" %
              (i1, i2, i2 - i1))
    assert i1 != -1
    assert i2 != -1

    track = track[i1: i2 - 1]

    first_beta = track[0][1]
    first_perp_x = math.cos(first_beta)
    first_perp_y = math.sin(first_beta)
    # Length of perpendicular jump to put together head and tail
    well_glued_together = np.sqrt(
        np.square(first_perp_x * (track[0][2] - track[-1][2]))
        + np.square(first_perp_y * (track[0][3] - track[-1][3]))
    )
    if well_glued_together > TRACK_DETAIL_STEP:
        return False

    # Red-white border on hard turns
    border = [False] * len(track)
    for i in range(len(track)):
        good = True
        oneside = 0
        for neg in range(BORDER_MIN_COUNT):
            beta1 = track[i - neg - 0][1]
            beta2 = track[i - neg - 1][1]
            good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
            oneside += np.sign(beta1 - beta2)
        good &= abs(oneside) == BORDER_MIN_COUNT
        border[i] = good
    for i in range(len(track)):
        for neg in range(BORDER_MIN_COUNT):
            border[i - neg] |= border[i]

    # Create tiles
    for i in range(len(track)):
        alpha1, beta1, x1, y1 = track[i]
        alpha2, beta2, x2, y2 = track[i - 1]
        road1_l = (
            x1 - TRACK_WIDTH * math.cos(beta1),
            y1 - TRACK_WIDTH * math.sin(beta1),
        )
        road1_r = (
            x1 + TRACK_WIDTH * math.cos(beta1),
            y1 + TRACK_WIDTH * math.sin(beta1),
        )
        road2_l = (
            x2 - TRACK_WIDTH * math.cos(beta2),
            y2 - TRACK_WIDTH * math.sin(beta2),
        )
        road2_r = (
            x2 + TRACK_WIDTH * math.cos(beta2),
            y2 + TRACK_WIDTH * math.sin(beta2),
        )
        vertices = [road1_l, road1_r, road2_r, road2_l]
        self.fd_tile.shape.vertices = vertices
        t = self.world.CreateStaticBody(fixtures=self.fd_tile)
        t.userData = t
        c = 0.01 * (i % 3)
        t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
        t.road_visited = False
        t.road_friction = 1.0
        t.tile_type = "ROAD"
        t.fixtures[0].sensor = True
        self.road_poly.append(
            ([road1_l, road1_r, road2_r, road2_l], t.color))
        self.road.append(t)
        if border[i]:
            side = np.sign(beta2 - beta1)
            b1_l = (
                x1 + side * TRACK_WIDTH * math.cos(beta1),
                y1 + side * TRACK_WIDTH * math.sin(beta1),
            )
            b1_r = (
                x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
            )
            b2_l = (
                x2 + side * TRACK_WIDTH * math.cos(beta2),
                y2 + side * TRACK_WIDTH * math.sin(beta2),
            )
            b2_r = (
                x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
            )
            self.road_poly.append(
                ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i %
                 2 == 0 else (1, 0, 0))
            )

    self.track = track
    return True


def render_road(self):
    # [0, 0, 0, 1]*4  # [0.235, 0.052, 0.198, 1.0] * 4
    colors = [0.4, 0.9, 0.4, 1.0] * 4
    polygons_ = [
        +PLAYFIELD,
        +PLAYFIELD,
        0,
        +PLAYFIELD,
        -PLAYFIELD,
        0,
        -PLAYFIELD,
        -PLAYFIELD,
        0,
        -PLAYFIELD,
        +PLAYFIELD,
        0,
    ]

    # create grass
    grass_fd = fixtureDef(
        shape=polygonShape(vertices=[(+PLAYFIELD, +PLAYFIELD), (+PLAYFIELD, -PLAYFIELD),
                                     (-PLAYFIELD, -PLAYFIELD), (-PLAYFIELD, +PLAYFIELD)])
    )
    grass = self.world.CreateStaticBody(fixtures=grass_fd)
    grass.userData = grass
    grass.color = [1, 1, 1, 1.0]
    grass.tile_type = "GRASS"
    grass.road_friction = 1.0
    grass.fixtures[0].sensor = True

    self.road_poly.insert(0, ([(+PLAYFIELD, +PLAYFIELD), (+PLAYFIELD, -PLAYFIELD),
                               (-PLAYFIELD, -PLAYFIELD), (-PLAYFIELD, +PLAYFIELD)], [1, 1, 1, 1.0]))

    # Road generation
    for poly, color in self.road_poly:
        colors.extend([color[0], color[1], color[2], 1] * len(poly))
        for p in poly:
            polygons_.extend([p[0], p[1], 0])

    # draw grass + road
    vl = pyglet.graphics.vertex_list(
        len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors)  # gl.GL_QUADS,
    )
    vl.draw(gl.GL_QUADS)
    vl.delete()


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # print(env)
        env._create_track = lambda: create_track(env)
        env.on_track = False
        env.on_grass_count = 0
        self.init_time = 0
        env.contactListener_keepref = FrictionDetector(env)
        env.world = Box2D.b2World(
            (0, 0), contactListener=env.contactListener_keepref)
        env.render_road = lambda: render_road(env)
        env.collisions = {}
        self.env = env

    def reset(self):
        self.env.reset()
        self.init_time = 0

    def step(self, action):
        self.env.collisions = {}
        self.env.on_track = False
        # time.sleep(0.5)
        res = self.env.step(action)

        if self.init_time > 25:
            if self.env.collisions.get("ROAD", None) == 1:
                #print("ON ROAD")
                self.env.on_grass_count = 0
            else:
                #print("ON GRASS")
                self.env.on_grass_count += 1

                if (self.env.on_grass_count > 3):
                    self.env.reward -= 5

                if self.env.on_grass_count > 20:
                    obs, reward, _, info = res
                    #print("YOU ARE DONE!")
                    return obs, reward, True, info

        else:
            self.init_time += 1
           # print(self.init_time)

        return res


discrete_action_space = {"turn_left": [-1, 0, 0], "turn_right": [1, 0, 0], "go": [0, 1, 0], "go_left": [-1, 1, 0], "go_right": [1, 1, 0], "brake": [0, 0, 1], "brake_left": [-1, 0, 1], "brake_right": [1, 0, 1], "slight_turn_left": [-.3,
                                                                                                                                                                                                                                       0, 0], "slight_turn_right": [.3, 0, 0], "slight_go": [0, .3, 0], "slight_go_left": [-.3, .3, 0], "slight_go_right": [.3, .3, 0], "slight_brake": [0, 0, .3], "slight_brake_left": [-.3, 0, .3], "slight_brake_right": [.3, 0, .3]}
# discrete_action_space.values())
# list([discrete_action_space["go"],
d_actions = list([discrete_action_space["go"], discrete_action_space["go"]])
# d_actions = list(discrete_action_space.values())
# discrete_action_space["go_left"], discrete_action_space["go_right"]])

env = MemoryWrapper(lambda: RewardWrapper(gym.make('CarRacing-v0').unwrapped))
model = RL_Model(env, DQN, d_actions)

# model.generate_policy_video("rl_progress_ep_" + str(0))


for i in range(1, 10):
    ep, rewards = model.train(
        100, render=True, epoch=i)
    model.save("results/rl_progress_ep_" + str(i * 100))
    model.generate_policy_video("results/rl_progress_ep_" + str(i*100))
    plt.title('Rewards Over Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.scatter([x for x in range(len(rewards))], rewards)
    plt.legend()
    plt.savefig("results/rl_progress_ep_"+str(i*100))
    plt.show()
    plt.pause(1e-5)
    plt.draw()
