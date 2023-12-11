import os
import numpy as np
from rotation.simulation import HMM_single_subject_simulation, perturb_covariances

if __name__ == '__main__':
    save_dir = './data/node_timeseries/simulation_toy_9/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the parameters
    n_scans = 100
    n_states = 8
    n_samples = 1200
    n_channels = 25

    # Read from early TPM
    #tpm = np.load('./results_HCP_202311_no_mean/HMM_ICA_50_state_4/trans_prob.npy')
    #tpm = np.array([[0.8,0.2],[0.2,0.8]])
    #tpm = np.array([[0.96,0.01,0.01,0.01,0.01],
    #                 [0.01,0.96,0.01,0.01,0.01],
    #                 [0.01,0.01,0.96,0.01,0.01],
    #                 [0.01,0.01,0.01,0.96,0.01],
    #                 [0.01,0.01,0.01,0.01,0.96]])
    #tpm = np.load('./results_HCP_202311_no_mean/HMM_ICA_25_state_8/trans_prob.npy')

    # tpm for simulation_toy_6
    #diagonal_value = 0.99
    #off_diagonal_value = (1 - diagonal_value) / (n_states - 1)
    #tpm = diagonal_value * np.eye(n_states) + off_diagonal_value * (1 - np.eye(n_states))

    # tpm for simulation_toy_9 (two subjects)
    tpm = np.load('./results_HCP_202311_no_mean/HMM_ICA_25_state_8/trans_prob.npy')


    #means = np.load('./results_202310/HMM_ICA_50_state_4/state_means.npy')
    means = np.zeros((n_states,n_channels))
    #covariances = np.load('./results_HCP_202311_no_mean/HMM_ICA_50_state_4/state_covariances.npy')
    #covariances = np.array([[[0.25,0.2],[0.2,0.25]],[[1.,0.8],[0.8,1.]]])
    #covariances = np.array([
    #    [[1.,-0.1],[-0.1,1.]],
    #    [[1.,-0.05],[-0.05,1.]],
    #    [[1.,0.],[0.,1.]],
    #    [[1.,0.05],[0.05,1.]],
    #    [[1.,0.1],[0.1,1.]]
    #])
    covariances = np.load('./results_HCP_202311_no_mean/HMM_ICA_25_state_8/state_covariances.npy')


    HMM_single_subject_simulation(save_dir=save_dir,
                                  n_scans=n_scans,
                                  n_states=n_states,
                                  n_samples=n_samples,
                                  n_channels=n_channels,
                                  trans_prob=tpm,
                                  means=means,
                                  covariances=perturb_covariances(covariances),
                                  subj_name = '10001')
    HMM_single_subject_simulation(save_dir=save_dir,
                                  n_scans=n_scans,
                                  n_states=n_states,
                                  n_samples=n_samples,
                                  n_channels=n_channels,
                                  trans_prob=tpm,
                                  means=means,
                                  covariances=perturb_covariances(covariances),
                                  subj_name='10002')
    '''
    ### Update 6th Dec 2023
    ### This is a simulation from Chet, it's only for comparison purposes.
    ### Please comment out all previous codes when doing the following simulation
    save_dir = './data/node_timeseries/simulation_toy_8/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    from osl_dynamics import data, simulation
    sim = simulation.HMM_MVN(
        n_samples=25600,
        n_states=5,
        n_channels=11,
        trans_prob="sequence",
        stay_prob=0.9,
        means="zero",
        covariances="random",
        random_seed=123,
    )

    from osl_dynamics.data import Data
    # Create Data object for training
    data = Data(sim.time_series)
    np.savetxt(f'{save_dir}10001.txt', sim.time_series)
    np.save(f'{save_dir}10001_state_time_course.npy', sim.state_time_course)
    np.save(f'{save_dir}10001_state_covariances.npy',sim.obs_mod.covariances)
    '''