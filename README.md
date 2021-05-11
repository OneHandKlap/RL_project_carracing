# CPS824-Project
## Crashing now so we don't crash later. Reinforcement Learning and self driving cars.

## Check out a compilation gif of our best model [Here!](deepq.gif)
## Check out our Summary Video [Here!](https://www.youtube.com/watch?v=YS-fcZ05Z9Y)
## Check out our Project Report [Here!](project_report.pdf)

# Installation
## On Windows: UNDER CONSTRUCTION

## On Linux: 
### Install Conda 
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html 

### Create Conda Environment and Install System Dependencies
```
sudo make install_sys
```

### Activate and Install Pip Dependencies
```
conda activate smort_cars
make install_dep
```

# Running the Project
## Run the Main Script to start training an agent
```
python src/main.py
```

# Changing Architectures or Setting Agent Parameters
## Creating an Agent
```python
from agents.dqn_agent import DQN_Agent
from models.dqn_basic import DQN_Basic
from config.baseline import config
import gym

env = gym.make("env_name_here")
agent = DQN_Agent(env, DQN_Basic, config)
```

## Training an Agent
``` python
agent.train(number_of_epochs=20,render=False, callbacks=[])
```

## Generating a Policy Video
```python
generate_policy_video(agent, env, filename="filename")
```

## Saving and Loading
```python
# saving
agent.save("output_name")

# loading
agent.load("input_name")
```

## Settings can be changed by passing in a different config file
### See src/config/baseline.py
``` python
# src/config/baseline.py
config = {
    "BATCH_SIZE": 128,  # minibatch size
    "MEMORY_CAPACITY": 7000,  # replay memory
    "EPISODES_PER_EPOCH": 100,  # for training
    "NUMBER_OF_EPOCHS": 10,  # for training
    "FRAME_SKIP": 2,  # number of frames to skip per action
    "FRAME_STACK": 3,  # number of frames to stack
    "GAMMA": 0.999,  # discount factor
    "EPSILON": 1.0,  # exploration rate
    "EPSILON_MIN": 0.1,  # min epsilon
    "LEARNING_RATE": 0.0001, #alpha learning
    "EPSILON_DECAY": 0.9999,  # rate at which epsilon decays
    "TARGET_UPDATE_INTERVAL": 10,  # interval at which to update target Q,
    "ACTION_SPACE": list({"turn_left": [-1, 0, 0], "turn_right": [1, 0, 0], "go": [0, 1, 0],
                          "go_left": [-1, 1, 0], "go_right": [1, 1, 0], "brake": [0, 0, 1],
                          "brake_left": [-1, 0, 1], "brake_right": [1, 0, 1], "slight_turn_left": [-.3, 0, 0],
                          "slight_turn_right": [.3, 0, 0], "slight_go": [0, .3, 0], "slight_go_left": [-.3, .3, 0],
                          "slight_go_right": [.3, .3, 0], "slight_brake": [0, 0, .3], "slight_brake_left": [-.3, 0, .3],
                          "slight_brake_right": [.3, 0, .3]}.values())  # action space [direction, throttle, brake]
}
```

