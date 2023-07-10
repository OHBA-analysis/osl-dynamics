import pathlib

import scipy.stats as stats
import numpy as np

from osl_dynamics.data import Data

def HMM_post(dataset):
    from osl_dynamics.models import load

    # Load the trained model
    model = load("results/model")
    alpha = model.get_alpha(dataset)
    print('The shape of alpha sample is: ', alpha[0].shape)
    data = model.get_training_time_series(dataset, prepared=False)

    for a, x in zip(alpha, dataset.time_series()):
        print(a.shape, x.shape)
    print('#################################')

if __name__ == '__main__':
    '''
    data_dir = pathlib.Path('/vols/Data/HCP/Phase2/group1200/node_timeseries/3T_HCP1200_MSMAll_d15_ts2/')
    subjs = []
    np_datas = []
    for file in data_dir.glob('*.txt'):
        subjs.append(file.stem)
        temp = np.loadtxt(file)
        temp = stats.zscore(temp, axis=0)

        assert temp.shape == (4800, 15)
        np_datas.append(temp)

    dataset = Data(np_datas)
    HMM_post(dataset)
    '''
    
    