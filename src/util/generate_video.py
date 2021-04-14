import imageio
import torch
from PIL import Image
from collections import deque
import numpy as np


def generate_policy_video(agent, env, filename="rl_model", num_episodes=1, max_episode_time=2000, fps=30, minimap=False):
    filename = filename + ".mp4"

    with imageio.get_writer(filename, fps=fps) as video:
        for episode in range(num_episodes):
            # reset env
            env.reset()
            init_state = agent.get_screen()
            screen_stack = deque([init_state] * agent.config['FRAME_STACK'], maxlen=agent.config['FRAME_STACK'])
            state = torch.cat(list(screen_stack), dim=1)

            num_steps = 0

            while True:
                done = False
                # pick an action
                action = agent.act(state, deterministic=True)

                # do the action
                for _ in range(agent.config["FRAME_SKIP"]):
                    num_steps += 1
                    next_state, reward, done, _ = env.step(agent.config["ACTION_SPACE"][action.item()])
                    if minimap:
                        mini = Image.fromarray(env.mini.render(mode="rgb_array"))
                        mini = mini.resize((150, 150))
                        full = Image.fromarray(env.render(mode="rgb_array"))
                        full.paste(mini, (0, 0, 150, 150))
                        video.append_data(np.array(full))
                    else:
                        video.append_data(env.render(mode="rgb_array"))
                    if done:
                        break

                # generate next state stack
                screen_stack.append(agent.get_screen())
                if minimap:
                    mini = Image.fromarray(env.mini.render(mode="rgb_array"))
                    mini = mini.resize((150, 150))
                    full = Image.fromarray(env.render(mode="rgb_array"))
                    full.paste(mini, (0, 0, 150, 150))
                    video.append_data(np.array(full))
                else:
                    video.append_data(env.render(mode="rgb_array"))
                next_state = torch.cat(list(screen_stack), dim=1) if not done else None
                state = next_state

                if done or num_steps > max_episode_time:
                    break
