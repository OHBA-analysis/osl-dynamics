import glob
import pathlib

import numpy as np
from osl_dynamics.data import Data

if __name__ == '__main__':
    data_dir = pathlib.Path('/vols/Data/HCP/Phase2/group1200/node_timeseries/3T_HCP1200_MSMAll_d15_ts2/')
    subjs = []
    np_datas = []
    for file in data_dir.glob('*.txt'):
        subjs.append(file.stem)
        temp = np.loadtxt(file)
        assert temp.shape == (4800,15)
        np_datas.append(temp)

    print('Mean of original data: ',np.mean(np_datas[0],axis=0))


    dataset = Data(np_datas)
    dataset.prepare()
    ts = dataset.time_series()
    print('###################################')
    print('shape of the first subject: ', ts[0].shape)
    print('Mean of the first subject: ',np.mean(ts[0],axis=0))
