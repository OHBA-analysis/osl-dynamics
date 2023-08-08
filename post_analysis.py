import sys
import pathlib

import scipy.stats as stats
import numpy as np

from osl_dynamics.data import Data
from rotation.utils import *
from rotation.preprocessing import PrepareData
from rotation.analysis import HMM_analysis, Dynemo_analysis, \
    MAGE_analysis, SWC_analysis, comparison_analysis

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
    # Index
    # 1-30: HMM
    # 31-60: Dynemo
    # 61-90: MAGE
    # 91-96: SWC (training)
    # 91-120: SWC (analysis)
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
    models = ['HMM','Dynemo','MAGE','SWC']
    list_channels = [15, 25, 50, 100, 200, 300]
    list_states = [4,8,12,16,20]
    index = int(sys.argv[1]) - 1

    # index = 120 represent comparison analysis.
    if index == 120:
        save_dir = './result/comparison/'
        comparison_analysis(models,list_channels,list_states,save_dir)

    model,n_channels, n_states = parse_index(index,models,list_channels,list_states,training=False)

    save_dir = f'./results/{model}_ICA_{n_channels}_state_{n_states}/'
    spatial_map_dir = f'./data/spatial_maps/groupICA_3T_HCP1200_MSMAll_d{n_channels}.ica/melodic_IC_sum.nii.gz'
    
    print(f'Number of channels: {n_channels}')
    print(f'Number of states: {n_states}')
    print(f'The model: {model}')

    if model == 'SWC':
        old_dir = f'./results/{model}_ICA_{n_channels}/'
        SWC_analysis(save_dir,old_dir,n_channels,n_states)
    else:
        # Work on Jalapeno
        #data_dir = pathlib.Path(f'/vols/Data/HCP/Phase2/group1200/node_timeseries/3T_HCP1200_MSMAll_d{n_channels}_ts2/')

        # Work on BMRC
        data_dir = pathlib.Path(f'./data/node_timeseries/3T_HCP1200_MSMAll_d{n_channels}_ts2/')
        prepare_data = PrepareData(data_dir)
        subj, dataset = prepare_data.load()
        print(f'Number of subjects: {len(subj)}')

        if model == 'HMM':
            HMM_analysis(dataset, save_dir, spatial_map_dir, n_channels,n_states)
        elif model == 'Dynemo':
            Dynemo_analysis(dataset, save_dir, spatial_map_dir, n_channels, n_states)
        elif model == 'MAGE':
            MAGE_analysis(dataset, save_dir, spatial_map_dir, n_channels, n_states)
        else:
            raise ValueError('The model name is incorrect!')


    