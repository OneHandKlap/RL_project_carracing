from util.replay_memory import ReplayMemory, Transition

from collections import deque

import numpy as np
import random
import torchvision.transforms as T
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import torch
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt


'''
DQN_Agent Class Providing Vanilla DQN Learning Capabilities
'''


class DQN_Agent():
    # create a new agent, passing in an environment, dqn_architecture and config
    def __init__(self, env, dqn, config):

        # set cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set env
        self.env = env
        self.env.reset()
        _, _, screen_height, screen_width = self.get_screen().shape

        # set dqn
        self.dqn = dqn

        # set config
        self.config = config

        # set various vars
        self.epsilon = self.config['EPSILON']

        # set replay memory
        self.replay_memory = ReplayMemory(self.config['MEMORY_CAPACITY'])

        # model net
        self.model = dqn(screen_width, screen_height, self.config["FRAME_STACK"],
                         len(self.config['ACTION_SPACE'])).to(self.device)

        # target net
        self.target = dqn(screen_width, screen_height, self.config["FRAME_STACK"],
                          len(self.config['ACTION_SPACE'])).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        # optimizer
        parameters_to_update = []
        for p in self.model.parameters():
            if p.requires_grad == True:
                parameters_to_update.append(p)

        self.optimizer = optim.Adam(parameters_to_update, lr=0.1)  # optim.RMSprop(parameters_to_update)
        self.scheduler = ExponentialLR(self.optimizer, gamma=.99)
        #

    def load(self, path="rl_model_weights.pth"):
        checkpoint = torch.load(path)
        # reinit
        self.__init__(self.env, self.dqn, checkpoint['config'])

        # model net
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # target net
        self.target.load_state_dict(checkpoint['target_state_dict'])
        self.target.eval()

        # optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # save model-net weights
    def save(self, path="rl_model_weights.pth"):
        torch.save({
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    # return the transformed state of the environment
    def get_screen(self):
        screen = self.env.render(mode='rgb_array')
        # screen = screen[np.ix_([x for x in range(100, 400)], [
        #                       x for x in range(200, 400)])]
        #screen = screen.transpose((2, 0, 1))
        # print(screen.shape)
        #screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        #screen = torch.from_numpy(screen)
        screen = T.Compose([T.ToPILImage(),
                            T.Resize(256, interpolation=Image.CUBIC),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])(screen).unsqueeze(0).to(self.device)

        return screen

    # select and taken an action on the environment, returning reward
    def act(self, state, deterministic=False):
        if deterministic == True:
            with torch.no_grad():
                return self.target(state).max(1)[1].view(1, 1)

        if np.random.rand() > self.epsilon:
            with torch.no_grad():
                return self.model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(len(self.config["ACTION_SPACE"]))]], device=self.device, dtype=torch.long)

    # optimize model based on replay memory, update weights of dqn and decay epsilon, return loss
    def learn(self):
        running_loss = 0
        if len(self.replay_memory) < self.config['BATCH_SIZE']:
            return running_loss

        # sample from replaymemory
        batch = self.replay_memory.sample(self.config['BATCH_SIZE'])

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
        # for each batch state according to model_net
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.config["BATCH_SIZE"], device=self.device)
        next_state_values[non_final_mask] = self.target(
            non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.config["GAMMA"]) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        running_loss += loss.item()

        # decay epsilon
        if self.epsilon > self.config['EPSILON_MIN']:
            self.epsilon *= self.config['EPSILON_DECAY']

        return running_loss

    # main training loop, resets epsilon each time you run it, so only run it once
    def train(self, episodes_per_epoch=None, number_of_epochs=None, callbacks=[], render=False):
        # reset epsilon
        self.epsilon = self.config['EPSILON']

        # set training time
        if episodes_per_epoch == None:
            episodes_per_epoch = self.config['EPISODES_PER_EPOCH']
        if number_of_epochs == None:
            number_of_epochs = self.config['NUMBER_OF_EPOCHS']

        # train
        for epoch in range(1, number_of_epochs + 1):
            for episode in range(1, episodes_per_epoch + 1):
                # reset env
                self.env.reset()
                init_state = self.get_screen()
                screen_stack = deque([init_state] * self.config['FRAME_STACK'], maxlen=self.config['FRAME_STACK'])
                state = torch.cat(list(screen_stack), dim=1)

                ep_reward = 0
                num_steps = 0
                ep_loss = 0

                negative_reward_count = 0

                while True:
                    if render == True:
                        plt.imshow(self.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
                        plt.draw()
                        plt.pause(1e-3)

                    # pick an action
                    action = self.act(state)

                    step_reward = 0

                    # do the action
                    for _ in range(self.config["FRAME_SKIP"]):
                        num_steps += 1
                        next_state, reward, done, _ = self.env.step(self.config["ACTION_SPACE"][action.item()])
                        step_reward += reward
                        if done:
                            break

                    ep_reward += step_reward
                    step_reward = torch.tensor([reward], device=self.device)

                    # generate next state stack
                    screen_stack.append(self.get_screen())
                    next_state = torch.cat(list(screen_stack), dim=1) if not done else None

                    # append to replay memory
                    self.replay_memory.append(state, action, next_state, step_reward)
                    state = next_state

                    # step reward counter
                    if step_reward < 0 and num_steps > 200:
                        negative_reward_count += 1

                        if negative_reward_count > 30:
                            break
                    else:
                        negative_reward_count = 0

                    # learn
                    ep_loss += self.learn()

                    if done:
                        break

                # run callbacks
                for c in callbacks:
                    c(self, epoch, episode, ep_reward, ep_loss/num_steps, self.epsilon, num_steps, self.scheduler.get_last_lr())

                if episode % self.config["TARGET_UPDATE_INTERVAL"] == 0:
                    self.target.load_state_dict(self.model.state_dict())

            # iterate scheduler after each epoch, comment out this step to prevent scheduler interference with optimizer
            self.scheduler.step()
            # print("\n\n\n"+str(self.scheduler.get_last_lr()))
            # for param_group in self.optimizer.param_groups:
            #     print(param_group['lr'])
