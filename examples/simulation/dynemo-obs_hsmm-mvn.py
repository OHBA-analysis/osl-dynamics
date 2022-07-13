"""Example script for fitting a multivariate normal observation model to data.

"""

print("Setting up")
import numpy as np
from osl_dynamics import data, simulation
from osl_dynamics.inference import tf_ops
from osl_dynamics.models.dynemo_obs import Config, Model
from osl_dynamics.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=5,
    n_channels=11,
    sequence_length=200,
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=50,
)

# Mixtures of modes to include in the simulation
mixed_mode_vectors = np.array(
    [[0.5, 0.5, 0, 0, 0], [0, 0.3, 0, 0.7, 0], [0, 0, 0.6, 0.4, 0]]
)

# Simulate data
print("Simulating data")
sim = simulation.MixedHSMM_MVN(
    n_samples=25600,
    n_channels=config.n_channels,
    mixed_mode_vectors=mixed_mode_vectors,
    gamma_shape=20,
    gamma_scale=10,
    means="zero",
    covariances="random",
    random_seed=123,
)
training_data = data.Data(sim.time_series)

# Prepare dataset
training_dataset = training_data.dataset(
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

# Inferred covariances
covariances = model.get_covariances()
plotting.plot_matrices(covariances - sim.covariances, filename="cov_diff.png")

# Delete temporary directory
training_data.delete_dir()
