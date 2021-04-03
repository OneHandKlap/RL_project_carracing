from agents.dqn_agent import DQN_Agent
from config.baseline import config

from models.dqn_basic import DQN_Basic

import matplotlib.pyplot as plt

from util.generate_video import generate_policy_video

import gym

HEADLESS = False

if HEADLESS:
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    is_ipython = 'inline' in plt.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

env = gym.make("CarRacing-v0")
agent = DQN_Agent(env, DQN_Basic, config)


def log(agent, epoch, episode, ep_reward, ep_loss, epsilon, num_steps):
    print(f"EPOCH: {epoch} - EPISODE: {episode} - REWARD: {ep_reward} - LOSS: {ep_loss} - EPSILON: {epsilon} - NUM_STEPS: {num_steps}")


def plot(agent, epoch, episode, ep_reward, ep_loss, epsilon, num_steps):
    plt.title('Rewards Over Episodes')
    plt.xlabel('Epochs')
    plt.ylabel('Rewards')
    plt.scatter((epoch+1) * (episode+1), ep_reward, color="blue")
    plt.savefig("results/plt.png")


def save(agent, epoch, episode, ep_reward, ep_loss, epsilon, num_steps):
    if episode % 50 == 0:
        print("SAVING AGENT")
        agent.save(f"results/rl_model_weights_{epoch}_{episode}.pth")
        generate_policy_video(agent, env, filename=f"results/video_{epoch}_{episode}")


agent.train(render=False, callbacks=[log, plot, save])
# agent.load("results/rl_model_weights_0_4.pth")
#generate_policy_video(agent, env, filename=f"tester")
