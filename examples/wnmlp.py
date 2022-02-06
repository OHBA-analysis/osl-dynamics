"""Example script for running inference with WNMLP.

"""

print("Setting up")
import numpy as np
from dynemo import data
from dynemo.inference import metrics, modes, tf_ops
from dynemo.models.wnmlp import Config, Model
from dynemo.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=5,
    n_channels=11,
    sequence_length=128,
    mod_wn_n_layers=6,
    mod_wn_n_filters=32,
    inf_mlp_n_layers=4,
    inf_mlp_n_units=128,
    obs_mlp_n_layers=4,
    obs_mlp_n_units=128,
    batch_size=16,
    learning_rate=0.005,
    n_epochs=100,
)

# Simulate data
np.random.seed(123)
sim_ts = np.zeros([25600, config.n_channels])
t = np.arange(25600) / 250
for i in range(config.n_modes):
    channels = np.random.randint(0, config.n_channels, size=config.n_channels // 3)
    p = np.random.uniform(0, 2 * np.pi)
    for j in channels:
        sim_ts[:, j] = i * np.sin(2 * np.pi * (2 * i + 1) * t + p)
sim_ts += np.random.normal(0, 0.1, size=sim_ts.shape)
sim_ts = data.manipulation.standardize(sim_ts)
meg_data = data.Data(sim_ts)

plotting.plot_separate_time_series(sim_ts, n_samples=2000, filename="sim_ts.png")

# Prepare dataset
training_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    step_size=config.sequence_length // 4,
    shuffle=True,
)
prediction_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
)

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(training_dataset, epochs=config.n_epochs)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred mode mixing factors and mode time course
inf_alp = model.get_alpha(prediction_dataset)

plotting.plot_separate_time_series(inf_alp, n_samples=2000, filename="inf_alp.png")
