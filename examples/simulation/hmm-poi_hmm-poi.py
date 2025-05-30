"""Example script for running HMM_Poisson inference on simulated HMM-POI data.

This script should take less than a couple minutes to run and
currently the  dice coefficient is ~0.99.
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
    lr_decay=0,
    n_epochs=20,
    n_init=5,
    n_init_epochs=2,
)

model = Model(config)
model.summary()

#%% Train model

model.train(data)

# Save
model_dir = f"{results_dir}/model"
model.save(model_dir)

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

#%% Compare inferred parameters to ground truth simulation

# Re-order simulated state time courses to match inferred
inf_stc, sim_stc = modes.match_modes(stc, sim.state_time_course)

# Calculate dice coefficient
dice = metrics.dice_coefficient(inf_stc, sim_stc)

print("Dice coefficient:", dice)
