import os
import numpy as np
from rotation.simulation import HMM_single_subject_simulation

if __name__ == '__main__':
    save_dir = './data/node_timeseries/simulation_toy/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the parameters
    n_scans = 100
    n_states = 2
    n_samples = 1200
    n_channels = 2

    # Read from early TPM
    #tpm = np.load('./results_HCP_202311_no_mean/HMM_ICA_50_state_4/trans_prob.npy')
    tpm = np.array([[0.8,0.2],[0.2,0.8]])
    #means = np.load('./results_202310/HMM_ICA_50_state_4/state_means.npy')
    means = np.zeros((n_states,n_channels))
    #covariances = np.load('./results_HCP_202311_no_mean/HMM_ICA_50_state_4/state_covariances.npy')
    covariances = np.array([[[0.25,0.2],[0.2,0.25]],[[1.,0.8],[0.8,1.]]])


    HMM_single_subject_simulation(save_dir=save_dir,
                                  n_scans=n_scans,
                                  n_states=n_states,
                                  n_samples=n_samples,
                                  n_channels=n_channels,
                                  trans_prob=tpm,
                                  means=means,
                                  covariances=covariances)

