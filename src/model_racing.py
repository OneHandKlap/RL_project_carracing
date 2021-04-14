from agents.dqn_agent import DQN_Agent
from util.generate_video import generate_policy_video
from wrappers.track_wrapper import TrackWrapper, TrackMini
from wrappers.memory_wrapper import MemoryWrapper
from config.baseline import config

from models.dqn_deep import DQN_Deep
from models.dqn_deep2 import DQN_Deep2
from models.dqn_basic import DQN_Basic

import gym

import matplotlib.pyplot as plt

HEADLESS = True

if HEADLESS:
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    is_ipython = 'inline' in plt.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

seed = 42

env = MemoryWrapper(lambda: TrackMini(lambda: gym.make("CarRacing-v0").unwrapped, seed))

models = [("basic1", DQN_Basic, "10_100"), ("basic2", DQN_Basic, "20_100"), ("deep", DQN_Deep, "10_100"), ("deep2", DQN_Deep2, "16_100")]

for name, model_class, weights in models:
    agent = DQN_Agent(env, model_class, config)
    agent.load(f"./src/models/pretrained/{name}/rl_model_weights_{weights}.pth")
    generate_policy_video(agent, env, filename=f"results/race_{seed}_{name}", minimap=True, max_episode_time=5000)
