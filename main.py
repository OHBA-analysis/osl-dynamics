import glob
import pathlib

import numpy as np
import scipy.stats as stats
from osl_dynamics.data import Data
from osl_dynamics.analysis import connectivity

def swc_analysis(dataset):
    ts = dataset.time_series()
    swc = connectivity.sliding_window_connectivity(ts, window_length=100, step_size=50, conn_type="corr")
    swc_concat = np.concatenate(swc)
    swc_concat = np.abs(swc_concat)

    print(swc_concat.shape)
    connectivity.save(
        swc_concat[:5],
        threshold=0.95,  # only display the top 5% of connections
    )


if __name__ == '__main__':
    data_dir = pathlib.Path('/vols/Data/HCP/Phase2/group1200/node_timeseries/3T_HCP1200_MSMAll_d15_ts2/')
    subjs = []
    np_datas = []
    for file in data_dir.glob('*.txt'):
        subjs.append(file.stem)
        temp = np.loadtxt(file)
        temp = stats.zscore(temp,axis=0)

        assert temp.shape == (4800,15)
        np_datas.append(temp)

    print('Mean of the standardised data: ',np.mean(np_datas[0],axis=0))
    print('Std of the standardised data: ', np.std(np_datas[0], axis=0))


    dataset = Data(np_datas)

    # Step 1: Sliding window analysis
    #swc_analysis(dataset)
