"""Example script for fitting a multivariate normal observation model to data.

"""

print("Setting up")
import numpy as np
from dynemo import data, files, simulation
from dynemo.inference import tf_ops
from dynemo.models.mgo import Config, Model
from dynemo.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Hyperparameters
config = Config(
    n_modes=5,
    n_channels=11,
    sequence_length=200,
    learn_means=True,
    learn_stds=True,
    learn_fcs=True,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=100,
)

print("Simulating data")
sim = simulation.MS_HMM_MVN(
    n_samples=25600,
    n_modes=config.n_modes,
    n_channels=config.n_channels,
    trans_prob='sequence',
    stay_prob=0.9,
    means="random",
    covariances="random",
    random_seed=123,
    fix_std=True,
    uni_std=True,
)
meg_data = data.Data(sim.time_series)

# Prepare dataset
training_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    alpha=[sim.mode_time_course[0]],
    beta=[sim.mode_time_course[1]],
    gamma=[sim.mode_time_course[2]],
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
plotting.plot_matrices(inf_fcs - sim_fcs, filename="stds_fcs.png")
