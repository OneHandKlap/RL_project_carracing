from agents.dqn_agent import DQN_Agent
from config.baseline import config

from models.dqn_basic import DQN_Basic

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


def f(epoch, episode, ep_reward, ep_loss, epsilon, num_steps):
    print(f"EPOCH: {epoch} - EPISODE: {episode} - REWARD: {ep_reward} - LOSS: {ep_loss} - EPSILON: {epsilon} - NUM_STEPS: {num_steps}")


agent.train(render=True, callbacks=[f])
