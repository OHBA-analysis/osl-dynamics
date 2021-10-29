"""Example script for running inference on simulated HMM-Sine data.

"""

print("Setting up")
import numpy as np
from dynemo import data, simulation
from dynemo.inference import callbacks, tf_ops
from dynemo.models import Config, Model

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600
amplitudes = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
frequencies = np.array(
    [[10, 10, 10, 10], [30, 30, 30, 30], [50, 50, 50, 50], [70, 70, 70, 70]]
)
sampling_frequency = 250

config = Config(
    sequence_length=128,
    inference_rnn="lstm",
    inference_n_units=64,
    inference_normalization="layer",
    model_rnn="lstm",
    model_n_units=64,
    model_normalization="layer",
    theta_normalization=None,
    alpha_xform="softmax",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    observation_model="wavenet",
    wavenet_n_filters=4,
    wavenet_n_layers=2,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=100,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=200,
)

# Simulate data
print("Simulating data")
sim = simulation.HMM_Sine(
    n_samples=n_samples,
    trans_prob="sequence",
    stay_prob=0.95,
    amplitudes=amplitudes,
    frequencies=frequencies,
    sampling_frequency=sampling_frequency,
    observation_error=0.05,
    random_seed=123,
)
meg_data = data.Data(sim.time_series)

config.n_modes = sim.n_modes
config.n_channels = sim.n_channels

# Prepare dataset
training_dataset = meg_data.dataset(
    config.sequence_length, config.batch_size, shuffle=True
)
prediction_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
)

# Build model
model = Model(config)
model.summary()

# Callbacks
dice_callback = callbacks.DiceCoefficientCallback(
    prediction_dataset, sim.mode_time_course
)

print("Training model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    callbacks=[dice_callback],
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")
