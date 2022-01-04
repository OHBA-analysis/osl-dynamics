"""Example script for running inference on simulated HMM-MVN data with a WaveNet
observation model.
"""

print("Setting up")
import numpy as np
from dynemo import data, files, simulation
from dynemo.inference import callbacks, metrics, modes, tf_ops
from dynemo.models import Config, Model

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=8,
    n_channels=40,
    sequence_length=128,
    inference_rnn="lstm",
    inference_n_units=64,
    inference_n_layers=3,
    inference_normalization="batch",
    model_rnn="lstm",
    model_n_units=64,
    model_n_layers=3,
    model_normalization="batch",
    theta_normalization="batch",
    alpha_xform="softmax",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    observation_model="wavenet",
    wavenet_n_filters=16,
    wavenet_n_layers=7,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=100,
    batch_size=32,
    learning_rate=0.001,
    n_epochs=200,
)

# Simulate data
print("Simulating data")
sim = simulation.HMM_MVN(
    n_samples=25600,
    n_channels=config.n_channels,
    n_modes=config.n_modes,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
    observation_error=0.05,
    random_seed=123,
)
sim.standardize()
meg_data = data.Data(sim.time_series)

# Prepare dataset
training_dataset = meg_data.dataset(
    config.sequence_length,
    config.batch_size,
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

dice_callback = callbacks.DiceCoefficientCallback(
    prediction_dataset, sim.mode_time_course
)

print("Training model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath="tmp/weights",
    callbacks=[dice_callback],
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")
