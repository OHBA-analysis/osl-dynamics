import os
import numpy as np
from rotation.simulation import HMM_single_subject_simulation

if __name__ == '__main__':
    save_dir = './data/node_timeseries/simulation_no_mean/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the parameters
    n_scans = 100
    n_states = 4
    n_samples = 1200
    n_channels = 50

    # Read from early TPM
    tpm = np.load('./results_HCP_202311_no_mean/HMM_ICA_50_state_4/trans_prob.npy')
    #means = np.load('./results_202310/HMM_ICA_50_state_4/state_means.npy')
    means = np.zeros((n_states,n_channels))
    covariances = np.load('./results_HCP_202311_no_mean/HMM_ICA_50_state_4/state_covariances.npy')

    HMM_single_subject_simulation(save_dir=save_dir,
                                  n_scans=n_scans,
                                  n_states=n_states,
                                  n_samples=n_samples,
                                  n_channels=n_channels,
                                  trans_prob=tpm,
                                  means=means,
                                  covariances=covariances)

