"""Calculate summary statistics.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of states and run id, e.g. python 6_calc_summary_stats.py 8 1")
    exit()
n_states = int(argv[1])
run = int(argv[2])

#%% Import packages

print("Importing packages")

import os
import pickle
import numpy as np

from osl_dynamics.inference import modes

#%% Setup directories

# Directories
inf_params_dir = f"results/{n_states}_states/run{run:02d}/inf_params"
summary_stats_dir = f"results/{n_states}_states/run{run:02d}/summary_stats"

os.makedirs(summary_stats_dir, exist_ok=True)

#%% Load state time course

# State probabilities
alp = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))

# Calculate a state time course by taking the most likely state
stc = modes.argmax_time_courses(alp)

#%% Calculate summary statistics

print("Calculating summary stats")

# Fractional occupancy
fo = modes.fractional_occupancies(stc)

# Mean lifetime
lt = modes.mean_lifetimes(stc, sampling_frequency=250)

# Mean interval
intv = modes.mean_intervals(stc, sampling_frequency=250)

# Mean switching rate
sr = modes.switching_rates(stc, sampling_frequency=250)

# Save
np.save(f"{summary_stats_dir}/fo.npy", fo)
np.save(f"{summary_stats_dir}/lt.npy", lt)
np.save(f"{summary_stats_dir}/intv.npy", intv)
np.save(f"{summary_stats_dir}/sr.npy", sr)
