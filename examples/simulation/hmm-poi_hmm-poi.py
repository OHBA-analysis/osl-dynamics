"""Example script for running HMM_Poisson inference on simulated HMM-POI data.

This script should take less than a couple minutes to run and
currently the  dice coefficient is ~0.80.
"""

import os
import pickle
import numpy as np

from osl_dynamics.simulation import HMM_Poi
from osl_dynamics.data import Data
from osl_dynamics.models.hmm_poi import Config, Model
from osl_dynamics.inference import modes, metrics

# Create directory for results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

#%% Simulate data

print("Simulating data")
sim = HMM_Poi(
    n_samples=25600,
    rates="random",
    n_states=5,
    n_channels=11,
    trans_prob="sequence",
    stay_prob=0.9,
    random_seed=123,
)

# Create Data object for training
data = Data(sim.time_series)


#%% Build model

config = Config(
    n_states=5,
    n_channels=11,
    sequence_length=200,
    learn_log_rates=True,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=50,
)

model = Model(config)
model.summary()

#%% Train model

# Initialization
init_history = model.random_subset_initialization(data, n_init=3, n_epochs=1, take = 0.25)

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
rates = model.get_rates()

# Save
inf_params_dir = f"{results_dir}/inf_params"
os.makedirs(inf_params_dir, exist_ok=True)

pickle.dump(alp, open(f"{inf_params_dir}/alp.pkl", "wb"))
np.save(f"{inf_params_dir}/rates.npy", rates)

#%% Calculate summary statistics

# Viterbi path
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

#%% Compare inferred parameters to ground truth simulation

# Re-order simulated state time courses to match inferred
inf_stc, sim_stc = modes.match_modes(stc, sim.state_time_course)

# Calculate dice coefficient
dice = metrics.dice_coefficient(inf_stc, sim_stc)

print("Dice coefficient:", dice)

#%% Delete temporary directory

data.delete_dir()
