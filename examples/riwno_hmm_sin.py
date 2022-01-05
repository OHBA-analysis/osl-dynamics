"""Example script for running inference on simulated HMM-Sine data.

- Achieves a dices of ~0.98.
"""

print("Setting up")
import numpy as np
from scipy import signal
from dynemo import data, simulation
from dynemo.inference import callbacks, tf_ops
from dynemo.models import Config, Model
from dynemo.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=8,
    n_channels=40,
    sequence_length=128,
    inference_rnn="lstm",
    inference_n_units=128,
    inference_n_layers=5,
    inference_normalization="batch",
    model_rnn="lstm",
    model_n_units=128,
    model_n_layers=5,
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
    batch_size=16,
    learning_rate=0.001,
    n_epochs=200,
)

# Simulate data
amplitudes = np.ones([config.n_modes, config.n_channels])
frequencies = np.array(
    [[5 * (i + 1)] * config.n_channels for i in range(config.n_modes)]
)

print("Simulating data")
sim = simulation.HMM_Sine(
    n_samples=25600,
    trans_prob="sequence",
    stay_prob=0.95,
    amplitudes=amplitudes,
    frequencies=frequencies,
    sampling_frequency=250,
    covariances="random",
    random_seed=123,
)
sim.standardize()
meg_data = data.Data(sim.time_series)

plotting.plot_time_series(sim.time_series, n_samples=2000, filename="train_data.png")
plotting.plot_alpha(sim.mode_time_course, n_samples=2000, filename="sim_alp.png")

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
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath="model/weights",
    callbacks=[dice_callback],
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred covariances
covs = model.get_covariances()

# Sample from the observation model
psds = []
for mode in range(config.n_modes):
    sample = model.sample(2000, covs=covs, mode=mode)
    f, psd = signal.welch(sample.T, fs=sim.sampling_frequency, nperseg=500)
    psds.append(psd[0])

plotting.plot_line(
    [f] * config.n_modes,
    psds,
    labels=[f"Mode {i + 1}" for i in range(config.n_modes)],
    x_range=[0, 60],
    x_label="Frequency [Hz]",
    y_label="PSD [a.u.]",
    filename="psds.png",
)
