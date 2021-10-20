"""Example script for fitting a convolutional neural network observation model to data.

- Simulates a sine wave at a particular frequency and trains a WaveNet model.
- The trained model is then sampled from.
"""

print("Setting up")
import numpy as np
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
    sampling_frequency=250,
    amplitudes=np.array([[1]]),
    frequencies=np.array([[3]]),
    trans_prob="sequence",
    stay_prob=0.9,
    observation_error=0.05,
)
meg_data = data.Data(sim.time_series)

# Settings
config = Config(
    observation_model="conv_net",
    n_filters=4,
    n_residual_blocks=1,
    n_conv_layers=7,
    n_modes=sim.n_modes,
    n_channels=sim.n_channels,
    sequence_length=128,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=200,
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

# Mean squared error
mse = model.loss(training_dataset)
print("MSE:", mse)

# Sample from the observation model
samples = np.empty([config.n_modes, 500, config.n_channels])
for alpha in range(config.n_modes):
    samples[alpha] = model.sample(500, std_dev=np.sqrt(mse), alpha=alpha)

plotting.plot_line([range(500)] * config.n_modes, samples, filename="samples.png")
