import numpy as np
from osl_dynamics import data, simulation
from osl_dynamics.inference import tf_ops
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.utils import plotting
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

def HMM_single_subject_simulation(save_dir:str, n_scans:int, n_states:int, n_samples:int,n_channels:int,
                                  trans_prob:np.ndarray,means:np.ndarray,covariances:np.ndarray):
    tf_ops.gpu_growth()
    time_series = []
    for i in range(n_scans):
        sim = simulation.HMM_MVN(
            n_samples=n_samples,
            n_states=n_states,
            n_channels=n_channels,
            trans_prob=trans_prob,
            means=means,
            covariances=covariances,
        )
        time_series.append(sim.time_series)
    time_series = np.concatenate(time_series,axis=0)
    print(time_series.shape)
    np.savetxt(f'{save_dir}10001.txt',time_series)