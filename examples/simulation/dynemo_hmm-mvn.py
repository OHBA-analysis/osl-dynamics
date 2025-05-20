"""Example script for running DyNeMo on simulated HMM-MVN data.

This script should take less than a couple minutes to run and
achieve a dice coefficient of ~0.99.
"""

print("Importing packages")

import os
import pickle
import numpy as np

from osl_dynamics.simulation import HMM_MVN
from osl_dynamics.data import Data
from osl_dynamics.models.dynemo import Config, Model
from osl_dynamics.inference import modes, metrics

# Create directory for results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

#%% Simulate data

print("Simulating data")
sim = HMM_MVN(
    n_samples=25600,
    n_modes=5,
    n_channels=20,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
)

# Create Data object for training
data = Data(sim.time_series)

# Prepare data
data.standardize()

#%% Build model

config = Config(
    n_modes=5,
    n_channels=20,
    sequence_length=100,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=20,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=40,
)

model = Model(config)
model.summary()

# Add regularization for the observation model
model.set_regularizers(data)

#%% Train model

model.train(data)

# Save
model_dir = f"{results_dir}/model"
model.save(model_dir)

#%% Get inferred parameters

# Inferred mode mixing coefficients
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

# Statistics summarising the mixing coefficients
alp_mean = np.array([np.mean(a, axis=0) for a in alp])
alp_std = np.array([np.std(a, axis=0) for a in alp])
alp_corr = np.array([np.corrcoef(a, rowvar=False) for a in alp])

# Save
summary_stats_dir = f"{results_dir}/summary_stats"
os.makedirs(summary_stats_dir, exist_ok=True)

np.save(f"{summary_stats_dir}/alp_mean.npy", alp_mean)
np.save(f"{summary_stats_dir}/alp_std.npy", alp_std)
np.save(f"{summary_stats_dir}/alp_corr.npy", alp_corr)

#%% Compare inferred parameters to ground truth simulation

# Binarize the mixing coefficients
stc = modes.argmax_time_courses(alp)

# Re-order simulated state time course to match inferred
inf_stc, sim_stc = modes.match_modes(stc, sim.state_time_course)

# Calculate dice coefficient
dice = metrics.dice_coefficient(inf_stc, sim_stc)

print("Dice coefficient:", dice)
