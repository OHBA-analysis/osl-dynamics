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

if __name__ == '__main__':
    model = 'HMM'
    n_channels = 50
    n_states = 10
    learn_means = False
    data_dir = pathlib.Path(f'./data/node_timeseries/3T_HCP1200_MSMAll_d{n_channels}_ts2/')

    index = int(sys.argv[1]) - 1
    mode = 'training'
    if len(sys.argv) >= 3:
        mode = sys.argv[2]

    # This either represent run repeat_i or the first/second half
    sub_index = 1
    if len(sys.argv) >= 4:
        sub_index = int(sys.argv[3])

    #sequence_lengths = [600,400,200,100,50,25]
    #batch_sizes = [1024,512,256,128,64,32]
    #learning_rates = [0.0005,0.001,0.005,0.01]

    #sequence_length, batch_size, learning_rate = parse_index(index,sequence_lengths,batch_sizes,learning_rates,training=True)
    sequence_lengths = [400,400,400,400,200,200,200,200,100,100,100,100]
    batch_sizes = [256,256,256,256,512,512,512,512,1024,1024,1024,1024]
    learning_rates = [0.0005,0.001,0.005,0.01,0.0005,0.001,0.005,0.01,0.0005,0.001,0.005,0.01,0.0005,0.001,0.005,0.01]

    sequence_length = sequence_lengths[index]
    batch_size = batch_sizes[index]
    learning_rate = learning_rates[index]




    save_dir = f'./results_training_config_202402/sl_{sequence_length}_bs_{batch_size}_lr_{learning_rate}'

    print(f'sequence length = {sequence_length}')
    print(f'batch size = {batch_size}')
    print(f'learning rate = {learning_rate}')

    if mode == 'training':
        prepare_data = PrepareData(data_dir)
        subj, dataset = prepare_data.load()
        print(f'Number of subjects: {len(subj)}')
        HMM_training(dataset,n_states,n_channels,save_dir,
                     sequence_length=sequence_length,
                     batch_size = batch_size,
                     learning_rate = learning_rate,
                     n_epochs = 40,
                     learn_means=learn_means
                     )
    elif mode == 'repeat':
        prepare_data = PrepareData(data_dir)
        subj, dataset = prepare_data.load()
        print(f'Number of subjects: {len(subj)}')

        save_dir_sub = f'{save_dir}/repeat_{sub_index}'
        print(f'save dir sub is: {save_dir_sub}')
        if not os.path.exists(save_dir_sub):
            # os.rmdir(save_dir_sub)
            HMM_training(dataset,n_states,n_channels,save_dir_sub,
                         sequence_length=sequence_length,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         n_epochs=40,
                         learn_means=learn_means
                         )