"""Example script for running inference on simulated HSMM data.

- Should achieve a dice coefficient of ~0.99.
- A seed is set for the random number generators for reproducibility.
"""

print("Setting up")
from pathlib import Path

import numpy as np
from vrad import data, simulation
from vrad.inference import metrics, states, tf_ops
from vrad.models import config, RIGO
from vrad.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600
observation_error = 0.2
gamma_shape = 10
gamma_scale = 5

dimensions = config.Dimensions(
    n_states=5,
    n_channels=80,
    sequence_length=200,
)

inference_network = config.RNN(
    rnn="lstm",
    n_layers=1,
    n_units=64,
    dropout_rate=0.0,
    normalization="layer",
)

model_network = config.RNN(
    rnn="lstm",
    n_layers=1,
    n_units=64,
    dropout_rate=0.0,
    normalization="layer",
)

alpha = config.Alpha(
    theta_normalization=None,
    xform="softmax",
    learn_temperature=False,
    initial_temperature=0.25,
)

observation_model = config.ObservationModel(
    model="multivariate_normal",
    learn_covariances=True,
    learn_alpha_scaling=False,
    normalize_covariances=False,
)

kl_annealing = config.KLAnnealing(
    do=True,
    curve="tanh",
    sharpness=10,
    n_epochs=100,
)

training = config.Training(
    batch_size=16,
    learning_rate=0.01,
    n_epochs=200,
)

# Load covariances for each state
example_file_directory = Path(__file__).parent / "files"
cov = np.load(example_file_directory / "hmm_cov.npy")

# Simulate data
print("Simulating data")
sim = simulation.HSMM_MVN(
    n_samples=n_samples,
    gamma_shape=gamma_shape,
    gamma_scale=gamma_scale,
    means="zero",
    covariances=cov,
    observation_error=observation_error,
    random_seed=123,
)
sim.standardize()
meg_data = data.Data(sim.time_series)
n_channels = meg_data.n_channels

# Prepare dataset
training_dataset = meg_data.training_dataset(
    dimensions.sequence_length, trainign.batch_size
)
prediction_dataset = meg_data.prediction_dataset(
    dimensions.sequence_length, training.batch_size
)

# Build model
model = RIGO(
    dimensions,
    inference_network,
    model_network,
    alpha,
    observation_model,
    kl_annealing,
    training,
)
model.summary()

print("Training model")
history = model.fit(
    training_dataset,
    epochs=training.n_epochs,
    save_best_after=kl_annealing.n_epochs,
    save_filepath="tmp/model",
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred state mixing factors and state time course
inf_alpha = model.predict_states(prediction_dataset)
inf_stc = states.time_courses(inf_alpha)
sim_stc = sim.state_time_course

sim_stc, inf_stc = states.match_states(sim_stc, inf_stc)
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

# Sample from the model RNN
mod_alpha = model.sample_alpha(n_samples=20000)
mod_stc = states.time_courses(mod_alpha)

# Plot lifetime distributions for the ground truth, inferred state time course
# and sampled state time course
plotting.plot_state_lifetimes(sim_stc, filename="sim_lt.png")
plotting.plot_state_lifetimes(inf_stc, filename="inf_lt.png")
plotting.plot_state_lifetimes(mod_stc, filename="mod_lt.png")

# Compare lifetime statistics
sim_lt_mean, sim_lt_std = metrics.lifetime_statistics(sim_stc)
inf_lt_mean, inf_lt_std = metrics.lifetime_statistics(inf_stc)
mod_lt_mean, mod_lt_std = metrics.lifetime_statistics(mod_stc)

print("Lifetime mean (Simulation):", sim_lt_mean)
print("Lifetime mean (Inferred):  ", inf_lt_mean)
print("Lifetime mean (Sample):    ", mod_lt_mean)

# Delete the temporary folder holding the data
meg_data.delete_dir()
