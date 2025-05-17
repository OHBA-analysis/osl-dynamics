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

from osl_dynamics.analysis import power, workbench

#%% Plot group-level amplitude maps

# Load inferred state means
means = np.load(f"{results_dir}/inf_params/means.npy")

# Save cifti containing the state maps
power.independent_components_to_surface_maps(
    ica_spatial_maps="melodic_IC.dscalar.nii",
    ic_values=means,
    output_file=f"{results_dir}/inf_params/means.dscalar.nii",
)

# Plot
workbench.render(
    img=f"{results_dir}/inf_params/means.dscalar.nii",
    gui=False,
    save_dir="tmp",
    image_name="plots/means_.png",
    input_is_cifti=True,
)
