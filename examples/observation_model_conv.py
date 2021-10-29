"""Example script for fitting a convolutional neural network observation model to data.

- Simulates multiple sine waves at different frequencies and trains a WaveNet model.
- The trained model is then sampled from.
"""

print("Setting up")
import numpy as np
from scipy import signal
from dynemo import data, simulation
from dynemo.inference import tf_ops
from dynemo.models import Config, Model
from dynemo.utils import plotting


# GPU settings
tf_ops.gpu_growth()

# Simulate data
print("Simulating data")
sim = simulation.HMM_Sine(
    n_samples=25600,
    sampling_frequency=100,
    amplitudes=np.array([[2], [1]]),
    frequencies=np.array([[2.5], [5]]),
    trans_prob="sequence",
    stay_prob=0.9,
    observation_error=0.05,
)
meg_data = data.Data(sim.time_series)

# Settings
config = Config(
    observation_model="wavenet",
    wavenet_n_filters=32,
    wavenet_n_layers=7,
    n_modes=sim.n_modes,
    n_channels=sim.n_channels,
    sequence_length=128,
    batch_size=16,
    learning_rate=0.005,
    n_epochs=400,
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

# Inferred standard deviation
std_dev = model.get_std_dev()
print("std_dev:", std_dev)

# Sample from the observation model
samples = []
psds = []
for alpha in range(config.n_modes):
    sample = model.sample(2000, std_dev=std_dev, alpha=alpha)
    f, psd = signal.welch(sample.T, fs=sim.sampling_frequency, nperseg=500)
    samples.append(sample)
    psds.append(psd[0])

plotting.plot_line(
    [range(len(sample))] * config.n_modes,
    samples,
    labels=[f"Mode {i + 1}" for i in range(config.n_modes)],
    x_label="Time [s]",
    y_label="Signal [a.u.]",
    filename="samples.png",
)
plotting.plot_line(
    [f] * config.n_modes,
    psds,
    labels=[f"Mode {i + 1}" for i in range(config.n_modes)],
    x_range=[0, 20],
    x_label="Frequency [Hz]",
    y_label="PSD [a.u.]",
    filename="psds.png",
)
