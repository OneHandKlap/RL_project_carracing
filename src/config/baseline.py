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
    "LEARNING_RATE": 0.0001 #alpha learning
    "EPSILON_DECAY": 0.9999,  # rate at which epsilon decays
    "TARGET_UPDATE_INTERVAL": 10,  # interval at which to update target Q,
    "ACTION_SPACE": list({"turn_left": [-1, 0, 0], "turn_right": [1, 0, 0], "go": [0, 1, 0],
                          "go_left": [-1, 1, 0], "go_right": [1, 1, 0], "brake": [0, 0, 1],
                          "brake_left": [-1, 0, 1], "brake_right": [1, 0, 1], "slight_turn_left": [-.3, 0, 0],
                          "slight_turn_right": [.3, 0, 0], "slight_go": [0, .3, 0], "slight_go_left": [-.3, .3, 0],
                          "slight_go_right": [.3, .3, 0], "slight_brake": [0, 0, .3], "slight_brake_left": [-.3, 0, .3],
                          "slight_brake_right": [.3, 0, .3]}.values())  # action space [direction, throttle, brake]
}
