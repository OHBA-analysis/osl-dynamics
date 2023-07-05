import glob
import pathlib

import numpy as np
import scipy.stats as stats
from osl_dynamics.data import Data
from osl_dynamics.analysis import connectivity
from osl_dynamics.models.hmm import Config, Model

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

def HMM_analysis(dataset):
    # Create a config object
    config = Config(
        n_states=8,
        n_channels=80,
        sequence_length=1000,
        learn_means=False,
        learn_covariances=True,
        batch_size=16,
        learning_rate=1e-3,
        n_epochs=10,  # for the purposes of this tutorial we'll just train for a short period
    )

    model = Model(config)
    model.summary()

    # Initialisation
    init_history = model.random_state_time_course_initialization(dataset, n_epochs=1, n_init=3)

    # Model training
    history = model.fit(dataset)

    # Save the model
    model.save("results/model")



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

        if len(np_datas)>10:
            continue

    print('Number of subjects: ',len(subjs))
    print('Mean of the standardised data: ',np.mean(np_datas[0],axis=0))
    print('Std of the standardised data: ', np.std(np_datas[0], axis=0))


    dataset = Data(np_datas)

    # Step 1: Sliding window analysis
    #swc_analysis(dataset)

    # Step 2: HMM analysis
    HMM_analysis(dataset)
