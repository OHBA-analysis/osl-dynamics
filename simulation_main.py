import os
import numpy as np
from rotation.simulation import HMM_simulation

if __name__ == '__main__':
    n_scans = 100
    n_states = 8
    save_dir = './data/node_timeseries/simulation/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

