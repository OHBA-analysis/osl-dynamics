"""Dual estimation.

In this script we estimate subject-specific means and covariances
given the group-level model.
"""

#%% Run ID and output directories

import os
from sys import argv

# We get the run ID from the command line arguments
if len(argv) != 2:
    print("Please pass a run id, e.g. python 4_dual_estimation.py 1")
    exit()
id = int(argv[1])

# Directories
results_dir = f"results/run{id:02d}"
model_dir = f"{results_dir}/model"
inf_params_dir = f"{results_dir}/inf_params"
dual_estimates_dir = f"{results_dir}/dual_estimates"

os.makedirs(dual_estimates_dir, exist_ok=True)

# Check results directory exists
if not os.path.isdir(results_dir):
    print(f"{results_dir} does not exist")
    exit()

#%% Import packages

import pickle
import numpy as np

from osl_dynamics.models import load
from osl_dynamics.data import Data

#%% Load trained model

model = load(model_dir)

#%% Load training data

# Load the list of file names created by 1_get_data_files.py
with open("data_files.txt", "r") as file:
    inputs = file.read().split("\n")

# Create Data object for training
data = Data(
    inputs,
    store_dir=f"tmp_{id}",
    n_jobs=8,
)

# Prepare data
data.standardize()

#%% Load inferred state probabilities

alpha = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))

#%% Dual estimation

# Calcualte subject-specific means and covariances
means, covs = model.dual_estimation(data, alpha)

# Save
np.save(f"{dual_estimates_dir}/means.npy", means)
np.save(f"{dual_estimates_dir}/covs.npy", covs)
