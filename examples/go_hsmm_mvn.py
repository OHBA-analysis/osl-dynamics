"""Example script for fitting a multivariate normal observation model to data.

"""

print("Setting up")
import numpy as np
from dynemo import data, files, simulation
from dynemo.inference import tf_ops
from dynemo.models.go import Config, Model
from dynemo.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600
observation_error = 0.2
gamma_shape = 20
gamma_scale = 10

# Load mode transition probability matrix and covariances of each mode
cov = np.load(files.example.path / "hmm_cov.npy")

# Mixtures of modes to include in the simulation
mixed_mode_vectors = np.array(
    [[0.5, 0.5, 0, 0, 0], [0, 0.3, 0, 0.7, 0], [0, 0, 0.6, 0.4, 0]]
)

# Simulate data
print("Simulating data")
sim = simulation.MixedHSMM_MVN(
    n_samples=n_samples,
    mixed_mode_vectors=mixed_mode_vectors,
    gamma_shape=gamma_shape,
    gamma_scale=gamma_scale,
    means="zero",
    covariances=cov,
    observation_error=observation_error,
    random_seed=123,
)
sim.standardize()
meg_data = data.Data(sim.time_series)

# Settings
config = Config(
    n_modes=sim.n_modes,
    n_channels=sim.n_channels,
    sequence_length=200,
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=20,
)

# Prepare dataset
training_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    alpha=[sim.mode_time_course],
    shuffle=True,
)

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(training_dataset, epochs=config.n_epochs)

covariances = model.get_covariances()
plotting.plot_matrices(covariances - sim.covariances, filename="cov_diff.png")
