"""Example script for fitting a multi-time-scale observation model to data.

"""

print("Setting up")
import numpy as np
from osl_dynamics import data, files, simulation
from osl_dynamics.inference import tf_ops
from osl_dynamics.models.mdynemo_obs import Config, Model
from osl_dynamics.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Hyperparameters
config = Config(
    n_modes=5,
    n_channels=40,
    sequence_length=200,
    learn_means=True,
    learn_stds=True,
    learn_fcs=True,
    batch_size=16,
    learning_rate=0.005,
    n_epochs=100,
)

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
training_data = data.Data(sim.time_series)

# Prepare dataset
training_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size,
    alpha=[sim.mode_time_course[0]],
    gamma=[sim.mode_time_course[1]],
    shuffle=True,
)

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(training_dataset, epochs=config.n_epochs)

# Get inferred parameters
inf_means, inf_stds, inf_fcs = model.get_means_stds_fcs()

# Compare with simulated parameters
sim_means = sim.means
sim_stds = np.array([np.diag(s) for s in sim.stds])
sim_fcs = sim.fcs

plotting.plot_matrices(inf_means - sim_means, filename="means_diff.png")
plotting.plot_matrices(inf_stds - sim_stds, filename="stds_diff.png")
plotting.plot_matrices(inf_fcs - sim_fcs, filename="fcs_diff.png")
