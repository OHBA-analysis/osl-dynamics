"""Example script for running inference on simulated HMM-Sine data.

- Simulates an HMM with sine waves for the observation model.
- Inference is performed with a MAR observation model.
- Achieves a dice of ~0.7.
- Only works well if each mode has channels that oscillate with the same frequency.
"""

print("Setting up")
import numpy as np
from vrad import analysis, data, simulation
from vrad.inference import callbacks, metrics, modes, tf_ops
from vrad.models import Config, Model
from vrad.utils import plotting

# GPU settings
tf_ops.gpu_growth()

# Settings
n_samples = 25600
amplitudes = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
frequencies = np.array(
    [[10, 10, 10, 10], [30, 30, 30, 30], [50, 50, 50, 50], [70, 70, 70, 70]]
)
sampling_frequency = 250
covariances = np.array(
    [
        [0.1, 0.1, 0.1, 0.1],
        [0.01, 0.01, 0.01, 0.01],
        [0.2, 0.1, 0.05, 0.1],
        [0.1, 0.3, 0.1, 0.1],
    ]
)

config = Config(
    sequence_length=100,
    inference_rnn="lstm",
    inference_n_units=32,
    inference_normalization="layer",
    model_rnn="lstm",
    model_n_units=32,
    model_normalization="layer",
    theta_normalization=None,
    alpha_xform="softmax",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    observation_model="multivariate_autoregressive",
    n_lags=2,
    diag_covs=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=50,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=400,
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
    covariances=covariances,
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

# Inferred mode mixing factors and mode time course
inf_alp = model.get_alpha(prediction_dataset)
inf_stc = modes.time_courses(inf_alp)
sim_stc = sim.mode_time_course

sim_stc, inf_stc = modes.match_modes(sim_stc, inf_stc)
print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

# Fractional occupancies
print("Fractional occupancies (Simulation):", modes.fractional_occupancies(sim_stc))
print("Fractional occupancies (VRAD):      ", modes.fractional_occupancies(inf_stc))

# Inferred parameters
inf_coeffs, inf_covs = model.get_params()

print("Coefficients:")
print(np.squeeze(inf_coeffs))
print()

print("Covariances:")
print(np.squeeze(inf_covs))
print()

# Calculate power spectral densities from model parameters
f, psd = analysis.spectral.mar_spectra(inf_coeffs, inf_covs, sampling_frequency)

for i in range(sim.n_modes):
    plotting.plot_line(
        [f] * sim.n_channels,
        psd[:, i, range(sim.n_channels), range(sim.n_channels)].T.real,
        labels=[f"channel {i}" for i in range(1, sim.n_channels + 1)],
        filename=f"psd_mode{i}.png",
    )
