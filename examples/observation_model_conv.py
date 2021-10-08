"""Example script for fitting a convolutional neural network observation model to data.

- Uses a conv. net observation model to infer a sine wave.
- The kernel weights should be approx. [-1, 2] and the bias should be [0].
- There is a large run-to-run variability, suspected to be due to the initialisation
  of the CNN weights.
"""

print("Setting up")
import numpy as np
from vrad import data, simulation
from vrad.inference import tf_ops
from vrad.models import Config, Model
from vrad.utils import plotting


# GPU settings
tf_ops.gpu_growth()

# Simulate data
print("Simulating data")
sim = simulation.HMM_Sine(
    n_samples=25600,
    amplitudes=np.array([[1]]),
    frequencies=np.array([[10]]),
    sampling_frequency=250,
    trans_prob=None,
)
meg_data = data.Data(sim.time_series)

# Settings
config = Config(
    observation_model="conv_net",
    n_states=sim.n_states,
    n_channels=sim.n_channels,
    sequence_length=100,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=200,
)

# Prepare dataset
training_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    alpha=[sim.state_time_course],
    shuffle=True,
)

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(training_dataset, epochs=config.n_epochs)

cnn_layer = model.get_layer("conv_net")
w = cnn_layer.get_weights()
print(np.squeeze(w))
