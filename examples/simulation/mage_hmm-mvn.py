"""Example script for running inference on simulated MS-HMM-MVN data.

- Multiple scale version for sage_hmm_mvn.py
- Should achieve a dice of close to one for alpha and gamma.
"""
print("Setting up")
import numpy as np
from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.mage import Config, Model
from osl_dynamics.utils import plotting

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
    discriminator_n_layers=1,
    discriminator_normalization="layer",
    learn_means=True,
    learn_stds=True,
    learn_fcs=True,
    batch_size=16,
    learning_rate=0.005,
    n_epochs=400,
    multiple_dynamics=True,
)

# Simulate data
print("Simulating data")
sim = simulation.MS_HMM_MVN(
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

# Prepare datasets
training_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=True,
)
prediction_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
)

# Build model
model = Model(config)

print("Training model")
history = model.fit(training_dataset)

# Inferred parameters
inf_means, inf_stds, inf_fcs = model.get_means_stds_fcs()

# Inferred mode mixing factors
inf_alpha, inf_gamma = model.get_mode_time_courses(prediction_dataset)

inf_alpha = modes.argmax_time_courses(inf_alpha)
inf_gamma = modes.argmax_time_courses(inf_gamma)

# Simulated mode mixing factors
sim_alpha, sim_gamma = sim.mode_time_course

# Match the inferred and simulated mixing factors
# Calculate the dice coefficient between mode time courses
orders = modes.match_modes(sim_alpha, inf_alpha, return_order=True)
inf_alpha = inf_alpha[:, orders[1]]

orders = modes.match_modes(sim_gamma, inf_gamma, return_order=True)
inf_gamma = inf_gamma[:, orders[1]]

# Compare with simulated parameters
sim_means = sim.means
sim_stds = np.array([np.diag(s) for s in sim.stds])
sim_fcs = sim.fcs

inf_means = inf_means[orders[1]]
inf_stds = inf_stds[orders[1]]
inf_fcs = inf_fcs[orders[1]]

plotting.plot_matrices(inf_means - sim_means, filename="means_diff.png")
plotting.plot_matrices(inf_stds - sim_stds, filename="stds_diff.png")
plotting.plot_matrices(inf_fcs - sim_fcs, filename="fcs_diff.png")

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
print("Fractional occupancies mean (MAGE):    ", fo_inf_alpha)

print("Fractional occupancies fc (Simulation):", fo_sim_gamma)
print("Fractional occupancies fc (MAGE):    ", fo_inf_gamma)
