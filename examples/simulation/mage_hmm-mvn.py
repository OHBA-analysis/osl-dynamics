"""Example script for running inference on simulated MDyn-HMM-MVN data.

- Multi-dynamic version for sage_hmm-mvn.py
- Should achieve a dice of close to one for alpha and gamma.
"""

print("Setting up")
import os
import numpy as np
from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.mage import Config, Model
from osl_dynamics.utils import plotting

# Create directory to hold plots
os.makedirs("figures", exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=5,
    n_channels=40,
    sequence_length=200,
    inference_n_units=128,
    inference_normalization="layer",
    model_n_units=128,
    model_normalization="layer",
    discriminator_n_units=32,
    discriminator_normalization="layer",
    learn_means=True,
    learn_stds=True,
    learn_fcs=True,
    batch_size=16,
    learning_rate=0.005,
    n_epochs=400,
)

# Simulate data
print("Simulating data")
sim = simulation.MDyn_HMM_MVN(
    n_samples=25600,
    n_modes=config.n_modes,
    n_channels=config.n_channels,
    trans_prob="sequence",
    stay_prob=0.9,
    means="random",
    covariances="random",
    random_seed=123,
)
sim.standardize()
training_data = data.Data(sim.time_series)

# Build model
model = Model(config)

print("Training model")
history = model.fit(training_data)

# Inferred parameters
inf_means, inf_stds, inf_fcs = model.get_means_stds_fcs()

# Inferred mode mixing factors
inf_alpha, inf_gamma = model.get_mode_time_courses(training_data)

inf_alpha = modes.argmax_time_courses(inf_alpha)
inf_gamma = modes.argmax_time_courses(inf_gamma)

# Simulated mode mixing factors
sim_alpha, sim_gamma = sim.mode_time_course

# Match the inferred and simulated mixing factors
_, order = modes.match_modes(sim_alpha, inf_alpha, return_order=True)
inf_alpha = inf_alpha[:, order]

_, order = modes.match_modes(sim_gamma, inf_gamma, return_order=True)
inf_gamma = inf_gamma[:, order]

# Compare with simulated parameters
sim_means = sim.means
sim_stds = np.array([np.diag(s) for s in sim.stds])
sim_fcs = sim.fcs

inf_means = inf_means[order]
inf_stds = inf_stds[order]
inf_fcs = inf_fcs[order]

plotting.plot_matrices(inf_means - sim_means, filename="figures/means_diff.png")
plotting.plot_matrices(inf_stds - sim_stds, filename="figures/stds_diff.png")
plotting.plot_matrices(inf_fcs - sim_fcs, filename="figures/fcs_diff.png")

# Dice coefficients
dice_alpha = metrics.dice_coefficient(sim_alpha, inf_alpha)
dice_gamma = metrics.dice_coefficient(sim_gamma, inf_gamma)

print("Dice coefficient for mean:", dice_alpha)
print("Dice coefficient for fc:", dice_gamma)

# Fractional occupancies
fo_sim_alpha = modes.fractional_occupancies(sim_alpha)
fo_sim_gamma = modes.fractional_occupancies(sim_gamma)

fo_inf_alpha = modes.fractional_occupancies(inf_alpha)
fo_inf_gamma = modes.fractional_occupancies(inf_gamma)

print("Fractional occupancies mean (Simulation):", fo_sim_alpha)
print("Fractional occupancies mean (MAGE):", fo_inf_alpha)

print("Fractional occupancies fc (Simulation):", fo_sim_gamma)
print("Fractional occupancies fc (MAGE):", fo_inf_gamma)

# Delete temporary directory
training_data.delete_dir()
