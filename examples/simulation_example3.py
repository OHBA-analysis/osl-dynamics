"""Example script for running inference on data with a mixture of HSMM states.

- Demonstrates VRAD's ability to infer a soft mixture of states.
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
gamma_shape = 20
gamma_scale = 10

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
    initial_temperature=0.25,
    learn_temperature=False,
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
    n_epochs=50,
)

training = config.Training(
    batch_size=16,
    learning_rate=0.01,
    n_epochs=100,
)

# Load covariances for each state
example_file_directory = Path(__file__).parent / "files"
cov = np.load(example_file_directory / "hmm_cov.npy")

# Mixtures of states to include in the simulation
mixed_state_vectors = np.array(
    [[0.5, 0.5, 0, 0, 0], [0, 0.3, 0, 0.7, 0], [0, 0, 0.6, 0.4, 0]]
)

# Simulate data
print("Simulating data")
sim = simulation.MixedHSMM_MVN(
    n_samples=n_samples,
    mixed_state_vectors=mixed_state_vectors,
    gamma_shape=gamma_shape,
    gamma_scale=gamma_scale,
    means="zero",
    covariances=cov,
    observation_error=observation_error,
    random_seed=123,
)
sim.standardize()
meg_data = data.Data(sim.time_series)

# Prepare dataset
training_dataset = meg_data.training_dataset(
    dimensions.sequence_length, training.batch_size
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
    save_filepath="tmp/weights",
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Compare the inferred state time course to the ground truth
alpha = model.predict_states(prediction_dataset)
matched_sim_stc, matched_alpha = states.match_states(sim.state_time_course, alpha)
plotting.plot_separate_time_series(
    matched_alpha, matched_sim_stc, n_samples=10000, filename="stc.png"
)

corr = metrics.correlation(matched_alpha, matched_sim_stc)
print("Correlation (VRAD vs Simulation):", corr)

# Delete the temporary folder holding the data
meg_data.delete_dir()
