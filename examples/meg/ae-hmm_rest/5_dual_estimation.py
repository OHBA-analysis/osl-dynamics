"""Dual estimation of observation model parameters.

"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of states and run id, e.g. python 5_dual_estimation.py 8 1")
    exit()
n_states = int(argv[1])
run = int(argv[2])

#%% Import packages

print("Importing packages")

import os
import pickle
import numpy as np

from osl_dynamics.models import load
from osl_dynamics.data import Data

#%% Directories

results_dir = f"results/{n_states}_states/run{run:02d}"
model_dir = f"{results_dir}/model"
inf_params_dir = f"{results_dir}/inf_params"
dual_estimates_dir = f"{results_dir}/dual_estimates"

os.makedirs(dual_estimates_dir, exist_ok=True)

#%% Load trained model

model = load(model_dir)

#%% Load data

data = Data(
    "training_data/networks",
    store_dir=f"tmp_{n_states}_{run}",
    n_jobs=8,
)

methods = {
    "amplitude_envelope": {},
    "moving_average": {"n_window": 5},
    "standardize": {},
}
data.prepare(methods)

#%% Load inferred state probabilities

alpha = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))

#%% Dual estimation

# Calculate subject-specific means and covariances
means, covs = model.dual_estimation(data, alpha)

# Save
np.save(f"{dual_estimates_dir}/means.npy", means)
np.save(f"{dual_estimates_dir}/covs.npy", covs)

#%% Delete temporary directory

data.delete_dir()
