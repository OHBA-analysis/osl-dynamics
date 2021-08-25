"""Example script for demonstrating VRAD's ability to infer a soft mixture of states.

"""

print("Setting up")
import os
import numpy as np
from vrad import data, simulation
from vrad.inference import metrics, states, tf_ops
from vrad.models import Config, Model
from vrad.utils import plotting
from tqdm import trange


# Create directory to hold plots
os.makedirs("figures", exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_states=6,
    sequence_length=200,
    inference_rnn="lstm",
    inference_n_units=64,
    inference_normalization="layer",
    model_rnn="lstm",
    model_n_units=64,
    model_normalization="layer",
    theta_normalization=None,
    alpha_xform="softmax",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=100,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=200,
)

print("Simulating data")
sim = simulation.MixedSine_MVN(
    n_samples=25600,
    n_states=6,
    n_channels=80,
    relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.1],
    amplitudes=[6, 5, 4, 3, 2, 1],
    frequencies=[1, 2, 3, 4, 6, 8],
    sampling_frequency=250,
    means="zero",
    covariances="random",
    random_seed=123,
)
sim_stc = sim.state_time_course
meg_data = data.Data(sim.time_series)

config.n_channels = meg_data.n_channels

# Plot ground truth logits
plotting.plot_separate_time_series(
    sim.logits, n_samples=2000, filename="figures/sim_logits.png"
)

# Prepare tensorflow datasets for training and model evaluation
training_dataset = meg_data.dataset(
    config.sequence_length, config.batch_size, shuffle=True
)
prediction_dataset = meg_data.dataset(
    config.sequence_length, config.batch_size, shuffle=False
)

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath="tmp/weights",
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred alpha and state time course
inf_alp = model.predict_states(prediction_dataset)
sim_stc, inf_stc = states.match_states(sim_stc, inf_alp)

# Compare the inferred state time course to the ground truth
plotting.plot_separate_time_series(
    sim_stc, inf_stc, n_samples=2000, filename="figures/stc.png"
)
plotting.state_stackplots(
    sim_stc,
    n_samples=2000,
    title="Ground Truth",
    y_label=r"$\alpha_{jt}$",
    filename="figures/sim_stc.png",
)
plotting.state_stackplots(
    inf_stc,
    n_samples=2000,
    title="VRAD",
    y_label=r"$\alpha_{jt}$",
    filename="figures/inf_stc.png",
)

# Correlation between state time courses
corr = metrics.alpha_correlation(inf_stc, sim_stc)
print("Correlation (VRAD vs Simulation):", corr)

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
    labels=["VRAD"],
    x_label="Sample",
    y_label="$d$",
    figsize=(15, 1.5),
    filename="figures/rd.png",
)

# Delete the temporary folder holding the data
meg_data.delete_dir()
