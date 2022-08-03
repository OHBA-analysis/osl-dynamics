"""Example script for running HMM inference on simulated HMM-MVN data.

"""

import os
import numpy as np
from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.utils import plotting

# Directory for plots
os.makedirs("figures", exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_states=5,
    n_channels=11,
    sequence_length=200,
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=30,
    learn_transprob=True,
)

# Simulate data
print("Simulating data")
sim = simulation.HMM_MVN(
    n_samples=25600,
    n_states=config.n_states,
    n_channels=config.n_channels,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
    random_seed=123,
)
sim.standardize()

# Use ground truth covariances as initialization
config.initial_covariances = sim.covariances

# Create training dataset
training_data = data.Data(sim.time_series)

# Build model
model = Model(config)
model.summary()

# Train model
history = model.fit(training_data)

# Loss
plotting.plot_line(
    [range(1, len(history["loss"]) + 1)],
    [history["loss"]],
    x_label="Epoch",
    y_label="Loss",
    filename="figures/loss.png",
)

# Get inferred parameters
inf_stc = model.get_alpha(training_data)
inf_means, inf_covs = model.get_means_covariances()
inf_tp = model.get_transprob()

# Re-order with respect to the simulation
sim_stc = sim.mode_time_course
_, order = modes.match_modes(sim_stc, inf_stc, return_order=True)
inf_stc = inf_stc[:, order]
inf_covs = inf_covs[order]
inf_tp = inf_tp[np.ix_(order, order)]

plotting.plot_alpha(
    sim_stc,
    inf_stc,
    n_samples=2000,
    y_labels=["Ground Truth", "Inferred"],
    filename="figures/stc.png",
)

plotting.plot_matrices(
    sim.covariances, main_title="Ground Truth", filename="figures/sim_covs.png"
)
plotting.plot_matrices(inf_covs, main_title="Inferred", filename="figures/inf_covs.png")

plotting.plot_matrices(inf_tp[np.newaxis, ...], filename="figures/transprob.png")

# Compare the inferred mode time course to the ground truth
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

print("Fractional occupancies (Simulation):", modes.fractional_occupancies(sim_stc))
print("Fractional occupancies (Inferred):", modes.fractional_occupancies(inf_stc))
