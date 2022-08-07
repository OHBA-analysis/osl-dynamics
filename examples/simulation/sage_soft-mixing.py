"""Example script for demonstrating SAGE's ability to infer a soft mixture of modes.

"""

print("Setting up")
import os
import numpy as np
from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.sage import Config, Model
from osl_dynamics.utils import plotting
from tqdm import trange

# GPU settings
tf_ops.gpu_growth()

# Create directory to hold plots
os.makedirs("figures", exist_ok=True)

# Settings
config = Config(
    n_modes=6,
    n_channels=80,
    sequence_length=200,
    inference_n_units=64,
    inference_n_layers=2,
    inference_normalization="layer",
    model_n_units=32,
    model_normalization="layer",
    discriminator_n_units=16,
    discriminator_normalization="layer",
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=0.001,
    n_epochs=500,
)

print("Simulating data")
sim = simulation.MixedSine_MVN(
    n_samples=25600,
    n_modes=config.n_modes,
    n_channels=config.n_channels,
    relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.1],
    amplitudes=[6, 5, 4, 3, 2, 1],
    frequencies=[1, 2, 3, 4, 6, 8],
    sampling_frequency=250,
    means="zero",
    covariances="random",
    random_seed=123,
)

sim_stc = sim.mode_time_course
training_data = data.Data(sim.time_series)

# Plot ground truth logits
plotting.plot_separate_time_series(
    sim.logits, n_samples=2000, filename="figures/sim_logits.png"
)

# Prepare tensorflow datasets for training and model evaluation
training_dataset = training_data.dataset(
    config.sequence_length, config.batch_size, shuffle=True
)
prediction_dataset = training_data.dataset(
    config.sequence_length, config.batch_size, shuffle=False
)

# Build model
model = Model(config)

print("Training model")
history = model.fit(training_dataset)

# Inferred alpha and mode time course
inf_alp = model.get_alpha(prediction_dataset)
sim_stc, inf_stc = modes.match_modes(sim_stc, inf_alp)

# Compare the inferred mode time course to the ground truth
plotting.plot_separate_time_series(
    sim_stc, inf_stc, n_samples=2000, filename="figures/stc.png"
)
plotting.plot_alpha(
    sim_stc,
    n_samples=2000,
    title="Ground Truth",
    y_labels=r"$\alpha_{jt}$",
    filename="figures/sim_stc.png",
)
plotting.plot_alpha(
    inf_stc,
    n_samples=2000,
    title="Sage",
    y_labels=r"$\alpha_{jt}$",
    filename="figures/inf_stc.png",
)

# Correlation between mode time courses
corr = metrics.alpha_correlation(inf_stc, sim_stc)
print("Correlation (Sage vs Simulation):", corr)

# Reconstruction of the time-varying covariance
sim_cov = sim.covariances
inf_cov = model.get_covariances()

sim_tvcov = np.sum(
    sim_stc[:, :, np.newaxis, np.newaxis] * sim_cov[np.newaxis, :, :, :], axis=1
)
inf_tvcov = np.sum(
    inf_alp[:, :, np.newaxis, np.newaxis] * inf_cov[np.newaxis, :, :, :], axis=1
)

# Calculate the Riemannian distance between the ground truth and inferred covariance
print("Calculating riemannian distances")
rd = np.empty(2000)
for i in trange(2000, ncols=98):
    rd[i] = metrics.riemannian_distance(sim_tvcov[i], inf_tvcov[i])

plotting.plot_line(
    [range(2000)],
    [rd],
    labels=["Sage"],
    x_label="Sample",
    y_label="$d$",
    fig_kwargs={"figsize": (15, 1.5)},
    filename="figures/rd.png",
)
