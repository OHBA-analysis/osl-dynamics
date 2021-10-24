"""Example script for performing regression on a spectrogram.

"""

print("Setting up")
import numpy as np
from dynemo import simulation
from dynemo.analysis import spectral
from dynemo.utils import plotting

# Simulate alpha
n_samples = 25600
sampling_frequency = 100

sim = simulation.MixedSine(
    n_modes=6,
    relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.1],
    amplitudes=[6, 5, 4, 3, 2, 1],
    frequencies=[1, 2, 3, 4, 6, 8],
    sampling_frequency=sampling_frequency,
    random_seed=123,
)
alpha = sim.generate_modes(n_samples)

plotting.plot_separate_time_series(alpha, n_samples=2000, filename="alpha.png")

# Simulate a mix of sine waves
data = np.zeros([n_samples, 1])
f = [2.5, 5, 7.5, 10, 12.5, 15]
p = np.random.uniform(0, 2 * np.pi, size=sim.n_modes)
t = np.arange(n_samples) / sampling_frequency
for i in range(sim.n_modes):
    data[:, 0] += alpha[:, i] * np.sin(2 * np.pi * f[i] * t + p[i])

# Spectral properties of each mode
frequency_range = [0, 20]
f, psd = spectral.regression_spectra(
    data=data,
    alpha=alpha,
    window_length=51,
    sampling_frequency=sampling_frequency,
    frequency_range=frequency_range,
    step_size=1,
    psd_only=True,
)

# Plot the mode PSDs
plotting.plot_line(
    [f] * psd.shape[0],
    psd,
    labels=[f"Mode {i}" for i in range(1, psd.shape[0] + 1)],
    x_range=frequency_range,
    x_label="Frequency [Hz]",
    y_label="PSD [a.u.]",
    filename="psd.png",
)
