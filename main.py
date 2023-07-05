import glob
import pathlib

import numpy as np
from osl_dynamics.data import Data

if __name__ == '__main__':
    data_dir = pathlib.Path('/vols/Data/HCP/Phase2/group1200/node_timeseries/3T_HCP1200_MSMAll_d15_ts2/')
    subjs = []
    datas = []
    for file in data_dir.glob('*.txt'):
        subjs.append(file.stem)
        temp = np.loadtxt(file)
        assert temp.shape == (4800,15)
        datas.append(temp)

    print('The length of subjs: ',len(subjs))
    print('Subject list: ',subjs)
