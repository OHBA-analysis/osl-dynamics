"""Example script for training Single-Channel DyNeMo on simulated data.

"""

print("Importing packages")

import numpy as np

from osl_dynamics.data import Data
from osl_dynamics.inference import modes, metrics
from osl_dynamics.simulation import HMM
from osl_dynamics.models.sc_dynemo import Config, Model
from osl_dynamics.utils import plotting

# Number of time points and sampling frequency
n_samples = 25600
sampling_frequency = 100

# Simulate a state time course
hmm = HMM(
    trans_prob="sequence",
    stay_prob=0.9,
    n_states=3,
    random_seed=123,
)
stc = hmm.generate_states(n_samples)

# Simulate observed data
t = np.arange(n_samples) / sampling_frequency
x = np.random.normal(0, 0.02, size=n_samples)

# State 1 - background, no oscillatory activity

# State 2 - alpha bursts
indices = stc[:, 1] == 1
phi = np.random.uniform(0, 2 * np.pi)
x[indices] = 2 * np.sin(2 * np.pi * 10 * t[indices] + phi)

# State 3 - beta bursts
indices = stc[:, 2] == 1
phi = np.random.uniform(0, 2 * np.pi)
x[indices] = np.sin(2 * np.pi * 20 * t[indices] + phi)

# Create Data object and prepare data
data = Data(x)
data.tde(n_embeddings=7)
data.standardize()

# Build model
config = Config(
    n_modes=3,
    n_channels=data.n_channels,
    sequence_length=100,
    inference_n_units=32,
    inference_normalization="layer",
    model_n_units=32,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    learn_oscillator_amplitude=True,
    oscillator_damping_limit=20,
    oscillator_frequency_limit=(1, 30),
    sampling_frequency=sampling_frequency,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=50,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=100,
)
model = Model(config)
model.summary()

# Train model
model.fit(data)

# Get inferred mixing coefficients and hard classify
alp = model.get_alpha(data)
alp = modes.argmax_time_courses(alp)

# Trim the simulate state time courses to match the inferred alphas
stc = stc[data.n_embeddings // 2 : alp.shape[0]]

# Match modes to simulation
stc, alp = modes.match_modes(stc, alp)

# Plot alphas
plotting.plot_alpha(stc, alp, n_samples=2000, filename="alpha.png")

# Print dice
dice = metrics.dice_coefficient(stc, alp)
print("Dice coefficient:", dice)
