"""Compare two groups.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of states and run id, e.g. python 7_compare_groups.py 8 1")
    exit()
n_states = int(argv[1])
run = int(argv[2])

#%% Import packages

print("Importing packages")

import os
import numpy as np
import pandas as pd

from osl_dynamics.analysis import statistics, power, connectivity
from osl_dynamics.utils import plotting

#%% Setup directories and files

results_dir = f"results/{n_states}_states/run{run:02d}"
spectra_dir = f"{results_dir}/spectra"
summary_stats_dir = f"{results_dir}/summary_stats"
group_diff_dir = f"{results_dir}/group_diff"

os.makedirs(group_diff_dir, exist_ok=True)

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

#%% Get group assignments

# Load ages
demographics = pd.read_csv("training_data/demographics.csv")
age = demographics["age"].values

# Create and young and old group
assignments = np.ones_like(age)
# Subjects with age > 34 are assigned to the old group
# and age <= 34 are assigned to the young group
for i, a in enumerate(age):
    if a > 34:
        assignments[i] += 1

# Group labels
group = ["Old" if a == 1 else "Young" for a in assignments]

#%% Compare summary stats

# Load
fo = np.load(f"{summary_stats_dir}/fo.npy")
lt = np.load(f"{summary_stats_dir}/lt.npy")
intv = np.load(f"{summary_stats_dir}/intv.npy")
sr = np.load(f"{summary_stats_dir}/sr.npy")

sum_stats = np.swapaxes([fo, lt, intv, sr], 0, 1)

# Do statistical significance testing
_, p = statistics.group_diff_max_stat_perm(
    data=sum_stats,
    assignments=assignments,
    n_perm=1000,
    n_jobs=8,
)
p = p.reshape(4, n_states)

# Plot
summary_stat_names = [
    "Fractional Occupancy",
    "Mean Lifetime (s)",
    "Mean Interval (s)",
    "Switching Rate (Hz)",
]
for i, name in enumerate(summary_stat_names):
    plotting.plot_summary_stats_group_diff(
        name,
        sum_stats[:, i],
        p[i],
        assignments=group,
        filename=f"{group_diff_dir}/sum_stats_{i + 1}.png",
    )

#%% Load spectra

f = np.load(f"{spectra_dir}/f.npy")
coh = np.load(f"{spectra_dir}/coh.npy")
psd = np.load(f"{spectra_dir}/psd.npy")
nnmf = np.load(f"{spectra_dir}/nnmf_2.npy")

#%% Compare power maps

# Calculate power maps
power_maps = power.variance_from_spectra(f, psd, nnmf)[:, 0]

# Do statistical significance testing
power_map_diff, p = statistics.group_diff_max_stat_perm(
    data=power_maps,
    assignments=assignments,
    n_perm=1000,
    n_jobs=8,
)

# Zero non-significant differences
significant = p < 0.05
power_map_diff[~significant] = 0

# Plot
power.save(
    power_map_diff,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
    filename=f"{group_diff_dir}/pow_diff_.png",
)

#%% Compare coherence networks

# Calculate coherences networks
c = connectivity.mean_coherence_from_spectra(f, coh, nnmf)[:, 0]
n_states = c.shape[1]
n_channels = c.shape[2]

# Just keep the upper triangle
m, n = np.triu_indices(c.shape[-1], k=1)
c = c[:, :, m, n]

# Do statistical significance testing
c_diff, p = statistics.group_diff_max_stat_perm(
    data=c,
    assignments=assignments,
    n_perm=1000,
    n_jobs=8,
)

# Zero non-significant differences
significant = p < 0.05
c_diff[~significant] = 0

# Convert back into a full matrix
coh_diff = np.zeros([n_states, n_channels, n_channels])
for i in range(n_states):
    coh_diff[i, m, n] = c_diff[i]
    coh_diff[i, n, m] = c_diff[i]

# Plot
connectivity.save(
    coh_diff,
    parcellation_file=parcellation_file,
    filename=f"{group_diff_dir}/coh_diff_.png",
)
