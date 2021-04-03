 def generate_policy_video(agent, environment, filename="rl_model", num_episodes=1, fps=30):
      filename = filename + ".mp4"

       with imageio.get_writer(filename, fps=fps) as video:

            for episode in range(num_episodes):
                time_step = environment.reset()
                done = False
                video.append_data(environment.render(mode="rgb_array"))
                last_screen = self.get_screen()
                current_screen = self.get_screen()
                state = current_screen - last_screen

                for i in range(self.max_episode_time):
                    action = self.select_deterministic_action(state)
                    # action = self.select_action(state)
                    _, reward, done, _ = environment.step(
                        self.action_space[action.item()])
                    video.append_data(environment.render(mode="rgb_array"))
                    last_screen = current_screen
                    current_screen = self.get_screen()
                    state = current_screen-last_screen

                    if(done):
                        break

            torch.cuda.empty_cache()
            gc.collect()

        return True
