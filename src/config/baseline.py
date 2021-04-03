config = {
    "BATCH_SIZE": 128,  # minibatch size
    "MEMORY_CAPACITY": 7000,  # replay memory
    "EPISODES_PER_EPOCH": 100,  # for training
    "NUMBER_OF_EPOCHS": 10,  # for training
    "SAVE_INTERVAL": 100,  # episode intervals to save model
    "FRAME_SKIP": 1,  # number of frames to skip per action
    "FRAME_STACK": 1,  # number of frames to stack
    "GAMMA": 0.999,  # discount factor
    "EPSILON": 1.0,  # exploration rate
    "EPSILON_MIN": 0.1,  # min epsilon
    "EPSILON_DECAY": 0.9999,  # rate at which epsilon decays
    "TARGET_Q_UPDATE": 10,  # interval at which to update target Q,
    "ACTION_SPACE": [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1]]  # action space [direction, throttle, brake]
}
