import numpy as np
import matplotlib.pyplot as plt

def generate_track():
    x = np.arange(0, 64)
    y = np.arange(0, 64)
    arr = np.zeros((y.size, x.size))

    cx = 32
    cy =32
    r = 25

    # The two lines below could be merged, but I stored the mask
    # for code clarity.
    mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2

    mask2 = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < 15**2
    arr[mask] = 1
    arr[mask2] = 0
    return arr

class Track():
    def __init__(self):
        self.action_space=[[1,1,0],[1,1,1],[0,1,1],[1,0,0],[0,0,0],[0,0,1]]
        
        self.state_space=generate_track()
        
    

if __name__=="__main__":
    track=Track()
    print(track.state_space)
    