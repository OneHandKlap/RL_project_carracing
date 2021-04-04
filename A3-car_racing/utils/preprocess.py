import numpy as np

def greyscale(state):
    """
    Preprocess state (210, 160, 3) image into
    a (80, 80, 1) image in grey scale
    """
    #print(state.shape)
    state = np.reshape(state, [96, 96, 3]).astype(np.float32)
    # grey scale
    state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114

    # karpathy
    #state = state[35:195]  # crop
    state = state[12:92, 12:92]  # crop
    #state = state[::2,::2] # downsample by factor of 2

    state = state[:, :, np.newaxis]
    #print(state.shape)
    return state.astype(np.uint8)


def blackandwhite(state):
    """
    Preprocess state (210, 160, 3) image into
    a (80, 80, 1) image in grey scale
    """
    # erase background
    state[state==144] = 0
    state[state==109] = 0
    state[state!=0] = 1

    # karpathy
    state = state[35:195]  # crop
    state = state[::2,::2, 0] # downsample by factor of 2

    state = state[:, :, np.newaxis]

    return state.astype(np.uint8)