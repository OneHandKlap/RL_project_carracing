import gym

env = gym.make('CarRacing-v0')
env.reset()

for _ in range(10000):
    env.render()
    env.step([0,.99,0])
    
env.close()