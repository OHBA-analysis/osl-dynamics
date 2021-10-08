"""Example script for inferring parameters of a MAR observation model.

"""

print("Setting up")
import numpy as np
from vrad import data, simulation
from vrad.analysis import spectral
from vrad.inference import tf_ops
from vrad.models import Config, Model
from vrad.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
amplitudes = np.ones([4, 4])
frequencies = np.array(
    [[10, 30, 50, 70], [15, 35, 55, 75], [20, 40, 60, 80], [25, 45, 65, 85]]
)
sampling_frequency = 250

# Simulate data
print("Simulating data")
sim = simulation.HMM_Sine(
    n_samples=25600,
    trans_prob="sequence",
    stay_prob=0.9,
    amplitudes=amplitudes,
    frequencies=frequencies,
    sampling_frequency=sampling_frequency,
    observation_error=0.05,
)
meg_data = data.Data(sim.time_series)

# Settings
config = Config(
    n_states=sim.n_states,
    n_channels=sim.n_channels,
    sequence_length=100,
    observation_model="multivariate_autoregressive",
    n_lags=2,
    diag_covs=True,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=100,
)

# Create dataset
training_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    alpha=[sim.state_time_course],
    shuffle=True,
)

# Build model
model = Model(config)
model.summary()

# Train the observation model
history = model.fit(training_dataset, epochs=config.n_epochs)

# Inferred parameters
inf_coeffs, inf_covs = model.get_params()

print("Coefficients:")
print(np.squeeze(inf_coeffs))
print()

print("Covariances:")
print(np.squeeze(inf_covs))
print()

# Calculate power spectral densities from model parameters
f, psd = spectral.mar_spectra(inf_coeffs, inf_covs, sampling_frequency)

for i in range(sim.n_states):
    plotting.plot_line(
        [f] * sim.n_channels,
        psd[:, i, range(sim.n_channels), range(sim.n_channels)].T.real,
        labels=[f"channel {i}" for i in range(1, sim.n_channels + 1)],
        filename=f"psd_state{i}.png",
    )
