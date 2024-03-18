"""Plot results.

"""

#%% Run ID and output directories

import os
from sys import argv

# We get the run ID from the command line arguments
if len(argv) != 2:
    print("Please pass a run id, e.g. python 5_plot_results.py 1")
    exit()
id = int(argv[1])

# Check results directory exists
results_dir = f"results/run{id:02d}"
plots_dir = f"{results_dir}/plots"

# Check results directory exists
if not os.path.isdir(results_dir):
    print(f"{results_dir} does not exist")
    exit()

# Create a directory to save plots to
os.makedirs(plots_dir, exist_ok=True)

#%% Import packages

print("Importing packages")

import numpy as np

from osl_dynamics.analysis import power, connectivity

#%% Plot group-level amplitude maps

# Load inferred state means
means = np.load(f"{results_dir}/inf_params/means.npy")

# Plot
power.save(
    means,
    mask_file="MNI152_T1_2mm_brain.nii.gz",
    parcellation_file="melodic_IC.nii.gz",
    plot_kwargs={"symmetric_cbar": True},
    filename=f"{plots_dir}/means_.png",
)

#%% Plot group-level functional connectivity

# Load inferred covariances
covs = np.load(f"{results_dir}/inf_params/covs.npy")

# Plot
connectivity.save(
    covs,
    parcellation_file="melodic_IC.nii.gz",
    plot_kwargs={
        "edge_cmap": "Reds",
        "display_mode": "xz",
        "annotate": False,
    },
    filename=f"{plots_dir}/corrs_.png",
)
