import gym


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
        print("Purging Environment")
        self.env.close()
        del self.env
        self.env = self.make_env()
