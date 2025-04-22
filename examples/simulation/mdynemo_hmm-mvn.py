"""Example script for running inference on simulated MDyn_HMM_MVN data.

- Multi-dynamic version for dynemo_hmm-mvn.py
- Should achieve a dice of ~0.99 for alpha and ~0.99 for beta.
"""

print("Setting up")
import os
import numpy as np
from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.mdynemo import Config, Model
from osl_dynamics.utils import plotting

os.makedirs("figures", exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=5,
    n_channels=20,
    sequence_length=100,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_means=True,
    learn_stds=True,
    learn_corrs=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=30,
    lr_decay=0.1,
    batch_size=8,
    learning_rate=0.01,
    n_epochs=60,
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
)
sim.standardize()
training_data = data.Data(sim.time_series)

training_data.prepare({"pca": {"n_pca_components": config.n_channels}})
config.pca_components = training_data.pca_components

# Build model
model = Model(config)
model.summary()

# Set regularisers
model.set_regularizers(training_data)

print("Training model")
history = model.fit(training_data)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(training_data)
print(f"Free energy: {free_energy}")

# Inferred mode mixing factors
inf_alpha, inf_beta = model.get_mode_time_courses(training_data)

inf_alpha = modes.argmax_time_courses(inf_alpha)
inf_beta = modes.argmax_time_courses(inf_beta)

# Simulated mode mixing factors
sim_alpha, sim_beta = sim.mode_time_course

# Inferred means, stds, corrs
inf_means, inf_stds, inf_corrs = model.get_means_stds_corrs()
sim_means = sim.means
sim_stds = sim.stds
sim_corrs = sim.corrs

# Match the inferred and simulated mixing factors
_, order_alpha = modes.match_modes(sim_alpha, inf_alpha, return_order=True)
_, order_beta = modes.match_modes(sim_beta, inf_beta, return_order=True)

inf_alpha = inf_alpha[:, order_alpha]
inf_beta = inf_beta[:, order_beta]

inf_means = inf_means[order_alpha]
inf_stds = np.array([np.diag(std) for std in inf_stds[order_alpha]])
inf_corrs = inf_corrs[order_beta]

# Dice coefficients
dice_alpha = metrics.dice_coefficient(sim_alpha, inf_alpha)
dice_beta = metrics.dice_coefficient(sim_beta, inf_beta)

print("Dice coefficient for power:", dice_alpha)
print("Dice coefficient for FC:", dice_beta)

# Fractional occupancies
fo_sim_alpha = modes.fractional_occupancies(sim_alpha)
fo_sim_beta = modes.fractional_occupancies(sim_beta)

fo_inf_alpha = modes.fractional_occupancies(inf_alpha)
fo_inf_beta = modes.fractional_occupancies(inf_beta)

print("Fractional occupancies mean (Simulation):", fo_sim_alpha)
print("Fractional occupancies mean (DyNeMo):", fo_inf_alpha)

print("Fractional occupancies FC (Simulation):", fo_sim_beta)
print("Fractional occupancies FC (DyNeMo):", fo_inf_beta)

# Plots
plotting.plot_alpha(
    sim_alpha,
    n_samples=2000,
    title="Ground truth " + r"$\alpha$",
    y_labels=r"$\alpha_{jt}$",
    filename="figures/sim_alpha.png",
)
plotting.plot_alpha(
    inf_alpha,
    n_samples=2000,
    title="Inferred " + r"$\alpha$",
    y_labels=r"$\alpha_{jt}$",
    filename="figures/inf_alpha.png",
)
plotting.plot_alpha(
    sim_beta,
    n_samples=2000,
    title="Ground truth " + r"$\beta$",
    y_labels=r"$\beta_{jt}$",
    filename="figures/sim_beta.png",
)
plotting.plot_alpha(
    inf_beta,
    n_samples=2000,
    title="Inferred " + r"$\beta$",
    y_labels=r"$\beta_{jt}$",
    filename="figures/inf_beta.png",
)
plotting.plot_matrices(
    sim_means, main_title="Ground Truth", filename="figures/sim_means.png"
)
plotting.plot_matrices(
    inf_means, main_title="Inferred", filename="figures/inf_means.png"
)

plotting.plot_matrices(
    sim_stds, main_title="Ground Truth", filename="figures/sim_stds.png"
)
plotting.plot_matrices(inf_stds, main_title="Inferred", filename="figures/inf_stds.png")
plotting.plot_matrices(
    sim_corrs, main_title="Ground Truth", filename="figures/sim_corrs.png"
)
plotting.plot_matrices(
    inf_corrs, main_title="Inferred", filename="figures/inf_corrs.png"
)
