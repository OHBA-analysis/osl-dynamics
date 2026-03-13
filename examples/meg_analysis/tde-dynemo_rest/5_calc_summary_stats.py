"""Calculate summary statistics.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of modes and run id, e.g. python 5_calc_summary_stats.py 6 1")
    exit()
n_modes = int(argv[1])
run = int(argv[2])

#%% Import packages

print("Importing packages")
import os
import pickle
import numpy as np
from scipy import stats
from osl_dynamics.inference import modes

#%% Setup directories

inf_params_dir = f"results/{n_modes}_modes/run{run:02d}/inf_params"
summary_stats_dir = f"results/{n_modes}_modes/run{run:02d}/summary_stats"

os.makedirs(summary_stats_dir, exist_ok=True)

#%% Load mode time courses

covs = np.load(f"{inf_params_dir}/covs.npy")
alp = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))
norm_alp = modes.reweight_alphas(alp, covs)

#%% Calculate summary statistics

mean_norm_alp = np.array([np.mean(a, axis=0) for a in norm_alp])
std_norm_alp = np.array([np.std(a, axis=0) for a in norm_alp])
kurt_norm_alp = np.array([stats.kurtosis(a, axis=0) for a in norm_alp])

np.save(f"{summary_stats_dir}/mean_norm_alp.npy", mean_norm_alp)
np.save(f"{summary_stats_dir}/std_norm_alp.npy", std_norm_alp)
np.save(f"{summary_stats_dir}/kurt_norm_alp.npy", kurt_norm_alp)
