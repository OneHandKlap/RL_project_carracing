import imageio
import torch
from collections import deque


def generate_policy_video(agent, env, filename="rl_model", num_episodes=1, max_episode_time=2000, fps=30):
    filename = filename + ".mp4"

    with imageio.get_writer(filename, fps=fps) as video:
        for episode in range(num_episodes):
            # reset env
            env.reset()
            init_state = agent.get_screen()
            screen_stack = deque([init_state] * agent.config['FRAME_STACK'], maxlen=agent.config['FRAME_STACK'])
            state = torch.cat(list(screen_stack), dim=1)

            done = False

            num_steps = 0

            while True:
                # pick an action
                action = agent.act(state)

                # do the action
                for _ in range(agent.config["FRAME_SKIP"]):
                    num_steps += 1
                    next_state, reward, done, _ = env.step(agent.config["ACTION_SPACE"][action.item()])
                    video.append_data(agent.env.render(mode="rgb_array"))
                    if done:
                        break

                # generate next state stack
                screen_stack.append(agent.get_screen())
                video.append_data(env.render(mode="rgb_array"))
                next_state = torch.cat(list(screen_stack), dim=1) if not done else None
                state = next_state

                if done or num_steps > max_episode_time:
                    break
