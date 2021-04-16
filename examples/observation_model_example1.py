"""Example script for fitting the observation model to HMM data.

- A seed is set for the random number generators for reproducibility.
"""

print("Setting up")
from pathlib import Path

import numpy as np
from vrad import data, simulation
from vrad.inference import tf_ops
from vrad.models import config, GO
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

observation_model = config.ObservationModel(
    model="multivariate_normal",
    learn_alpha_scaling=False,
    normalize_covariances=False,
)

training = config.Training(
    batch_size=16,
    learning_rate=0.01,
    n_epochs=20,
)

# Load state transition probability matrix and covariances of each state
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
    dimensions.sequence_length,
    training.batch_size,
    alpha=[sim.state_time_course],
)

# Build model
model = GO(dimensions, observation_model, training)
model.summary()

print("Training model")
history = model.fit(training_dataset, epochs=training.n_epochs)

covariances = model.get_covariances()
plotting.plot_matrices(covariances - sim.covariances, filename="cov_diff.png")

# Delete the temporary folder holding the data
meg_data.delete_dir()
