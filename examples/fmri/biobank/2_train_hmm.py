"""Train an HMM.

"""

#%% Run ID and output directory

import os
from sys import argv

# We get the run ID from the command line arguments
if len(argv) != 2:
    print("Please pass a run id, e.g. python 2_train_hmm.py 1")
    exit()
id = int(argv[1])

# Create directory for results
results_dir = f"results/run{id:02d}"
os.makedirs(results_dir, exist_ok=True)

#%% Import packages

print("Importing packages")

import pickle
import numpy as np

from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.inference import modes

#%% Load data

# Load the list of file names created by 1_get_data_files.py
with open("data_files.txt", "r") as file:
    inputs = file.read().split("\n")

# Create Data object for training
data = Data(
    inputs,
    use_tfrecord=True,
    store_dir=f"tmp_{id}",
    n_jobs=8,
)

# Prepare data
data.standardize()

#%% Build model

config = Config(
    n_states=10,
    n_channels=data.n_channels,
    sequence_length=50,
    learn_means=True,
    learn_covariances=True,
    batch_size=512,
    learning_rate=0.01,
    n_epochs=20,
)

model = Model(config)
model.summary()

#%% Train model

# Initialization
init_history = model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)

# Full training
history = model.fit(data)

# Save model
model_dir = f"{results_dir}/model"
model.save(model_dir)

# Calculate the free energy
free_energy = model.free_energy(data)
history["free_energy"] = free_energy

# Save training history and free energy
pickle.dump(init_history, open(f"{model_dir}/init_history.pkl", "wb"))
pickle.dump(history, open(f"{model_dir}/history.pkl", "wb"))

#%% Get inferred parameters

# Inferred state probabilities
alp = model.get_alpha(data)

# Observation model parameters
means, covs = model.get_means_covariances()

# Save
inf_params_dir = f"{results_dir}/inf_params"
os.makedirs(inf_params_dir, exist_ok=True)

pickle.dump(alp, open(f"{inf_params_dir}/alp.pkl", "wb"))
np.save(f"{inf_params_dir}/means.npy", means)
np.save(f"{inf_params_dir}/covs.npy", covs)

#%% Calculate summary statistics

# State time course
stc = modes.argmax_time_courses(alp)

# Calculate summary statistics
fo = modes.fractional_occupancies(stc)
lt = modes.mean_lifetimes(stc)
intv = modes.mean_intervals(stc)
sr = modes.switching_rates(stc)

# Save
summary_stats_dir = f"{results_dir}/summary_stats"
os.makedirs(summary_stats_dir, exist_ok=True)

np.save(f"{summary_stats_dir}/fo.npy", fo)
np.save(f"{summary_stats_dir}/lt.npy", lt)
np.save(f"{summary_stats_dir}/intv.npy", intv)
np.save(f"{summary_stats_dir}/sr.npy", sr)

#%% Delete temporary directory

data.delete_dir()
