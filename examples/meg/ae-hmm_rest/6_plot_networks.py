"""Plot power maps and coherence networks for each state.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of states and run id, e.g. python 6_plot_networks.py 8 1")
    exit()
n_states = int(argv[1])
run = int(argv[2])

#%% Import packages

print("Importing packages")
import os
import numpy as np
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting

#%% Setup directories and files

results_dir = f"results/{n_states:02d}_states/run{run:02d}"
inf_params_dir = f"{results_dir}/inf_params"
networks_dir = f"{results_dir}/networks"

os.makedirs(networks_dir, exist_ok=True)

mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"

#%% Plot mean activity spatial maps

means = np.load(f"{inf_params_dir}/means.npy")
power.save(
    means,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    filename=f"{networks_dir}/mean_.png",
)

#%% Plot AEC networks

covs = np.load(f"{inf_params_dir}/covs.npy")
covs = connectivity.threshold(covs, percentile=98, subtract_mean=True)
connectivity.save(
    covs,
    parcellation_file=parcellation_file,
    filename=f"{networks_dir}/cov_.png",
)
