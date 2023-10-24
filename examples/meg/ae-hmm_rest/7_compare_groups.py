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
dual_estimates_dir = f"{results_dir}/dual_estimates"
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

#%% Compare mean activity maps

# Load subject-specific mean activity maps
means = np.load(f"{dual_estimates_dir}/means.npy")

# Do statistical significance testing
means_diff, p = statistics.group_diff_max_stat_perm(
    data=means,
    assignments=assignments,
    n_perm=1000,
    n_jobs=8,
)

# Zero non-significant differences
significant = p < 0.05
means_diff[~significant] = 0

# Plot
power.save(
    means_diff,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
    filename=f"{group_diff_dir}/means_diff_.png",
)

#%% Compare AEC networks

# Load subject-specific AEC matrices
covs = np.load(f"{dual_estimates_dir}/covs.npy")
n_states = covs.shape[1]
n_channels = covs.shape[2]

# Just keep the upper triangle
m, n = np.triu_indices(covs.shape[-1], k=1)
covs = covs[:, :, m, n]

# Do statistical significance testing
c_diff, p = statistics.group_diff_max_stat_perm(
    data=covs,
    assignments=assignments,
    n_perm=1000,
    n_jobs=8,
)

# Zero non-significant differences
significant = p < 0.05
c_diff[~significant] = 0

# Convert back into a full matrix
covs_diff = np.zeros([n_states, n_channels, n_channels])
for i in range(n_states):
    covs_diff[i, m, n] = c_diff[i]
    covs_diff[i, n, m] = c_diff[i]

# Plot
connectivity.save(
    covs_diff,
    parcellation_file=parcellation_file,
    filename=f"{group_diff_dir}/covs_diff_.png",
)
