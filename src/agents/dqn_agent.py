from util.replay_memory import ReplayMemory

from collections import deque

import numpy as np
import random
import torchvision.transforms as T
import torch
import torch.optim as optim
from PIL import Image

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
        self.model = dqn(screen_width, screen_height,
                         len(self.config['ACTION_SPACE'])).to(self.device)

        # target net
        self.target = dqn(screen_width, screen_height,
                          len(self.config['ACTION_SPACE'])).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        # optimizer
        parameters_to_update = []
        for p in self.model.parameters():
            if p.requires_grad == True:
                parameters_to_update.append(p)

        self.optimizer = optim.RMSprop(parameters_to_update)

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

    # save policy-net weights
    def save(self, path="rl_model_weights"):
        torch.save({
            'config': self.config,
            'model_state_dict': self.policy.state_dict(),
            'target_state_dict': self.target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        },
            path + ".pth")

    # return the transformed state of the environment
    def get_screen(self):
        screen = self.env.render(mode='rgb_array')
        screen = screen[np.ix_([x for x in range(100, 400)], [
                               x for x in range(200, 400)])]
        screen = screen.transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return T.Compose([T.ToPILImage(),
                          T.Resize(40, interpolation=Image.CUBIC),
                          T.ToTensor()])(screen).unsqueeze(0).to(self.device)

    # select and taken an action on the environment, returning reward
    def act(self, state, deterministic=False):
        if np.random.rand() > self.epsilon:
            with torch.no_grad():
                return self.config["ACTION_SPACE"][self.model(state).max(1)[1].view(1, 1).item()]
        else:
            return self.config["ACTION_SPACE"][torch.tensor([[random.randrange(len(self.config["ACTION_SPACE"]))]], device=self.device, dtype=torch.long).item()]

    # optimize model based on replay memory, update weights of dqn and decay epsilon, return loss
    def learn(self):
        if len(self.replay_memory) < self.config['BATCH_SIZE']:
            return 0

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
        for param in self.model.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # decay epsilon
        if self.epsilon > self.config['EPSILON_MIN']:
            self.epsilon *= self.config['EPSILON_DECAY']

        return loss.item()

    # main training loop, resets epsilon each time you run it, so only run it once
    def train(self, episodes_per_epoch=None, number_of_epochs=None, callbacks=[]):
        # reset epsilon
        self.epsilon = self.config['EPSILON']

        # set training time
        if episodes_per_epoch == None:
            episodes_per_epoch = self.config['EPISODES_PER_EPOCH']
        if number_of_epochs == None:
            number_of_epochs = self.config['NUMBER_OF_EPOCHS']

        # train
        for epoch in range(number_of_epochs):
            for episode in range(episodes_per_epoch):
                # reset env
                self.env.reset()
                init_state = self.get_screen()
                state_stack = deque([init_state] * self.config['FRAME_STACK'], maxlen=self.config['FRAME_STACK'])
                frame_count = 0

                ep_reward = 0
                num_steps = 0
                ep_loss = 0

                negative_reward_count = 0

                for i in range(100):
                    num_steps += 1

                    # get current/last state stack
                    last_state_stack = np.array(state_stack)

                    # pick an action
                    action = self.act(last_state_stack)

                    step_reward = 0

                    # do the action
                    for _ in range(self.config["FRAME_SKIP"]):
                        next_state, reward, done, _ = self.env.step(action)
                        step_reward += reward
                        if done:
                            break

                    ep_reward += step_reward

                    # generate next state stack
                    state_stack.append(next_state)
                    next_stack_stack = np.array(state_stack)

                    # append to replay memory
                    self.replay_memory.append(last_state_stack, action, next_stack_stack, step_reward)

                    # step reward counter
                    if step_reward <= 0:
                        negative_reward_count += 1
                        if negative_reward_count > 10:
                            break
                    else:
                        negative_reward_count = 0

                    # learn
                    ep_loss += self.learn()

                # run callbacks
                for c in callbacks:
                    c(epoch, episode, ep_reward, ep_loss, self.epsilon, num_steps)
