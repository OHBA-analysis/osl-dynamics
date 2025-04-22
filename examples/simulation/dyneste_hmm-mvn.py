"""Example script for running DyNeStE on simulated HMM-MVN data.

This script should take less than a couple minutes to run and
achieve a dice coefficient of ~0.99.
"""

print("Importing packages")

import os
import pickle
import numpy as np

from osl_dynamics.simulation import HMM_MVN
from osl_dynamics.data import Data
from osl_dynamics.models.dyneste import Config, Model
from osl_dynamics.inference import modes, metrics

# Create directory for results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# %% Simulate data

print("Simulating data")
sim = HMM_MVN(
    n_samples=25600,
    n_states=5,
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

# %% Build model

config = Config(
    n_states=5,
    n_channels=20,
    sequence_length=200,
    inference_n_units=128,
    inference_normalization="layer",
    model_n_units=128,
    model_normalization="layer",
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=50,
    do_gs_annealing=True,
    gs_annealing_curve="exp",
    initial_gs_temperature=1.0,
    final_gs_temperature=0.06,
    gs_annealing_slope=0.04,
    n_gs_annealing_epochs=120,
    batch_size=16,
    learning_rate=5e-3,
    lr_decay=0.01,
    n_epochs=120,
)

model = Model(config)
model.summary()

# Add regularization for the observation model
model.set_regularizers(data)

# %% Train model

# Initialization
init_history = model.random_subset_initialization(data, n_init=5, n_epochs=1, take=1)

# Full training
history = model.fit(data)

# Save model
model_dir = f"{results_dir}/model"
model.save(model_dir)

# Calculate the free energy
free_energy = model.free_energy(data)
print("Free energy", free_energy)
history["free_energy"] = free_energy

# Save training history and free energy
pickle.dump(init_history, open(f"{model_dir}/init_history.pkl", "wb"))
pickle.dump(history, open(f"{model_dir}/history.pkl", "wb"))

# %% Get inferred parameters

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

# %% Calculate summary statistics

# State time courses
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

# %% Compare inferred parameters to ground truth simulation

# Re-order simulated state time course to match inferred
inf_stc, sim_stc = modes.match_modes(stc, sim.state_time_course)

# Calculate dice coefficient
dice = metrics.dice_coefficient(inf_stc, sim_stc)

print("Dice coefficient:", dice)
