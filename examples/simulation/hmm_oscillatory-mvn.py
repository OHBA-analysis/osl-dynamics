"""Example script for training a TDE-HMM on simulated oscillatory data.

Each state has its own covariance and a spectral peak at its own frequency.
The state spectra estimated with the inferred state time course are compared
to the ground truth spectra.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from osl_dynamics.simulation import HMM_OscillatoryMVN
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.inference import modes, metrics
from osl_dynamics.analysis import spectral
from osl_dynamics.utils import set_random_seed

set_random_seed(0)

# Create directory for results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

#%% Simulate data

sampling_frequency = 250
n_embeddings = 15
sequence_length = 200

print("Simulating data")
sim = HMM_OscillatoryMVN(
    n_samples=51200,
    trans_prob="uniform",
    stay_prob=0.98,
    frequencies=[4, 10, 20, 30],
    sampling_frequency=sampling_frequency,
    covariances="random",
    n_channels=8,
    peak_width=3,
    observation_error=0.1,
)
sim.standardize()

# Create Data object and prepare data
data = Data(sim.time_series)
data.tde_pca(n_embeddings=n_embeddings, n_pca_components=40)
data.standardize()

#%% Build model

config = Config(
    n_states=4,
    n_channels=data.n_channels,
    sequence_length=sequence_length,
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=20,
)

model = Model(config)
model.summary()

#%% Train model

model.random_state_time_course_initialization(data, n_init=3, n_epochs=2)
model.fit(data)

#%% Compare to the ground truth

# Inferred state time course
alp = model.get_alpha(data)
inf_stc = modes.argmax_time_courses(alp)

# Simulated state time course for the same time points
sim_stc = sim.state_time_course[n_embeddings // 2 : n_embeddings // 2 + len(alp)]

# Match the inferred states to the simulated states
_, order = modes.match_modes(sim_stc, inf_stc, return_order=True)
inf_stc = inf_stc[:, order]
alp = alp[:, order]

print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc))

# State spectra estimated with the inferred state time course
x = data.trim_time_series(sequence_length=sequence_length, prepared=False)
f, psd, coh = spectral.multitaper_spectra(
    x,
    sampling_frequency,
    alpha=alp,
    frequency_range=[1, 45],
    standardize=False,
)

# Ground truth spectra
true_psd = np.einsum("kfii->kif", sim.power_spectra(f)).real

# Plot the spectra averaged over channels
fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
for k, ax in enumerate(axes):
    ax.plot(f, true_psd[k].mean(0), "k--", label="Ground truth")
    ax.plot(f, psd[k].mean(0), label="Inferred")
    ax.set_title(f"State {k + 1} ({sim.frequencies[k]:g} Hz)")
    ax.set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("PSD (a.u.)")
axes[0].legend()
fig.tight_layout()
fig.savefig(f"{results_dir}/psd.png")
