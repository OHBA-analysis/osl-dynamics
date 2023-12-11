import glob
import warnings
import sys
import os
import pathlib

import numpy as np
import scipy.stats as stats
from sklearn.model_selection import KFold
from rotation.preprocessing import PrepareData
from rotation.training import HMM_training, Dynemo_training, MAGE_training,SWC_computation
from rotation.utils import *
from rotation.analysis import HMM_analysis
from osl_dynamics.data import Data
from osl_dynamics.analysis import connectivity


def swc_analysis(dataset):
    ts = dataset.time_series()
    swc = connectivity.sliding_window_connectivity(ts, window_length=100, step_size=50, conn_type="corr")
    swc_concat = np.concatenate(swc)
    swc_concat = np.abs(swc_concat)

    print(swc_concat.shape)
    np.save('results/model_swc/dfc.npy')
    '''
    connectivity.save(
        swc_concat[:5],
        threshold=0.95,  # only display the top 5% of connections
    )
    '''

def model_train(model,dataset,n_channels, n_states,save_dir,learn_means=True,learn_covariances=True,learn_trans_prob=True,learning_rate=1e-3):
    if model == 'HMM':
        HMM_training(dataset,n_states,n_channels,save_dir,
                     learn_means=learn_means,
                     learn_covariances=learn_covariances,
                     learn_trans_prob=learn_trans_prob,
                     learning_rate=learning_rate)
    elif model == 'Dynemo':
        Dynemo_training(dataset, n_states, n_channels,save_dir,learn_means=learn_means)
    elif model == 'MAGE':
        MAGE_training(dataset,n_states,n_channels,save_dir,learn_means=learn_means)
    elif model == 'SWC':
        SWC_computation(dataset,window_length=143,step_size=118,save_dir=save_dir)
    else:
        raise ValueError('The model name is incorrect!')


if __name__ == '__main__':
    # Index
    # 1-30: HMM
    # 31-60: Dynemo
    # 61-90: MAGE
    # 91-96: SWC (training)
    # 91-120: SWC (analysis)

    # Mode should be encoded in sys.argv[2]
    # sys.argv[2] == 'training' (or no sys.argv[2]): train the model
    # sys.argv[2] == 'repeat': reproduce the model training to get reliable KL divergence
    # sys.argv[2] == 'split': split the data in half to test reproducibility
    models = ['HMM','Dynemo','MAGE','SWC']
    list_channels = [15, 25, 50, 100, 200, 300]
    #list_states = [4,8,12,16,20]
    # Update swimming 20231015: try train HMM model with more states
    #list_states = [25,30,35,40,45]
    list_states = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    learn_means = False
    learn_covariances = True
    learn_trans_prob = False
    learning_rate = 0.01

    index = int(sys.argv[1]) - 1
    #index = 6

    # Default mode is training:
    mode = 'training'
    if len(sys.argv) >= 3:
        mode = sys.argv[2]

    # This either represent run repeat_i or the first/second half
    sub_index = 1
    if len(sys.argv) >= 4:
        sub_index = int(sys.argv[3])

    # Default strategy for splitting
    strategy = '0'
    if len(sys.argv) >= 5:
        strategy = sys.argv[4]



    # For debugging
    #index = 0
    #mode = 'repeat'
    #strategy = '1'

    model,n_channels, n_states = parse_index(index,models,list_channels,list_states,training=True)
    
    if n_states is None:
        save_dir = f'./results_simulation_202311_toy_6_lr_0.01/{model}_ICA_{n_channels}/'
    else:
        save_dir = f'./results_simulation_202311_toy_6_lr_0.01/{model}_ICA_{n_channels}_state_{n_states}'
    
    print(f'Number of channels: {n_channels}')
    print(f'Number of states: {n_states}')
    print(f'The model: {model}')
    
   # data_dir = pathlib.Path(f'/vols/Data/HCP/Phase2/group1200/node_timeseries/3T_HCP1200_MSMAll_d{n_channels}_ts2/')
    #data_dir = pathlib.Path(f'./data/node_timeseries/3T_HCP1200_MSMAll_d{n_channels}_ts2/')
    data_dir = pathlib.Path(f'./data/node_timeseries/simulation_toy_6/')

    if mode == 'training':
        prepare_data = PrepareData(data_dir)
        subj, dataset = prepare_data.load(z_score_data=False)
        print(f'Number of subjects: {len(subj)}')
        model_train(model,dataset,n_channels,n_states,save_dir,
                    learn_means=learn_means,
                    learn_covariances=learn_covariances,
                    learn_trans_prob=learn_trans_prob,
                    learning_rate=learning_rate)
    elif mode == 'repeat':
        prepare_data = PrepareData(data_dir)
        subj, dataset = prepare_data.load(z_score_data=False)
        print(f'Number of subjects: {len(subj)}')

        save_dir_sub = f'{save_dir}/repeat_{sub_index}'
        print(f'save dir sub is: {save_dir_sub}')
        if not os.path.exists(save_dir_sub):
            #os.rmdir(save_dir_sub)
            model_train(model,dataset,n_channels,n_states,save_dir_sub,
                        learn_means=learn_means,
                        learn_covariances=learn_covariances,
                        learn_trans_prob=learn_trans_prob,
                        learning_rate=learning_rate
                        )

    elif mode == 'split':
        prepare_data = PrepareData(data_dir)
        subj, dataset_1,dataset_2 = prepare_data.load(z_score_data=False, split_strategy = strategy)
        print(f'Number of subjects: {len(subj)}')
        if sub_index == 1:
            save_dir_sub = f'{save_dir}/split_{strategy}_first_half'
            dataset = dataset_1
        else:
            save_dir_sub = f'{save_dir}/split_{strategy}_second_half'
            dataset = dataset_2
        model_train(model, dataset_1, n_channels, n_states, save_dir_sub,
                    learn_means=learn_means,
                    learn_covariances=learn_covariances,
                    learn_trans_prob=learn_trans_prob,
                    learning_rate=learning_rate
                    )
    elif mode == "cross_validation":
        prepare_data = PrepareData(data_dir)
        subj, dataset = prepare_data.load(z_score_data=False)
        kf = KFold(shuffle=True,random_state=42)
        for j, (train_index, test_index) in enumerate(kf.split(range(len(dataset.arrays)))):
            save_dir_sub = f'{save_dir}/cross_validation_{j}/'
            if not os.path.exists(save_dir_sub):
                os.makedirs(save_dir_sub)

            with dataset.set_keep(list(train_index)):
                print(f'Cross validation number{j}')
                print(f'Please check the length of training dataset: {len(dataset.arrays)}')
                model_train(model,dataset,n_channels,n_states,save_dir_sub,
                            learn_means=learn_means,
                            learn_covariances=learn_covariances,
                            learn_trans_prob=learn_trans_prob,
                            learning_rate=learning_rate
                            )
                HMM_analysis(dataset, save_dir_sub, None, None, n_channels, n_states)

            with dataset.set_keep(list(test_index)):
                save_dir_sub_validation = f'{save_dir_sub}/validation/'
                print(f'Please check the length of validation dataset: {len(dataset.arrays)}')
                if not os.path.exists(save_dir_sub_validation):
                    os.makedirs(save_dir_sub_validation)
                HMM_analysis(dataset, save_dir_sub_validation, None, None, n_channels, n_states,model_dir=save_dir_sub)

    else:
        raise ValueError('Mode is not available now!')



        
    '''
    data_dir =pathlib.Path('/vols/Data/HCP/Phase2/group1200/node_timeseries/3T_HCP1200_MSMAll_d15_ts2/')
    subjs = []
    np_datas = []
    for file in data_dir.glob('*.txt'):
        subjs.append(file.stem)
        temp = np.loadtxt(file)
        temp = stats.zscore(temp,axis=0)

        assert temp.shape == (4800,15)
        np_datas.append(temp)

        if len(np_datas)>10:
            continue

    print('Number of subjects: ',len(subjs))
    print('Mean of the standardised data: ',np.mean(np_datas[0],axis=0))
    print('Std of the standardised data: ', np.std(np_datas[0], axis=0))


    dataset = Data(np_datas)
    '''
    # Step 1: Sliding window analysis
    #swc_analysis(dataset)

    # Step 2: HMM analysis
    #HMM_analysis(dataset)

    # Step 3: Dynemo analysis
    #Dynemo_analysis(dataset)
