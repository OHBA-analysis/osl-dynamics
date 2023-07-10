"""Example code for making plots.

"""

import numpy as np

from osl_dynamics import simulation
from osl_dynamics.utils import plotting


#%% Basic plots

# Line plot
x = np.arange(100)
y1 = np.random.normal(size=100)
y2 = np.random.normal(size=100)
plotting.plot_line([x], [y1], filename="line.png")

# Plot a figure with two subplots, one line plot and one scatter plot
fig, ax = plotting.create_figure(2, 1)
plotting.plot_line([x], [y1], ax=ax[0])
plotting.plot_scatter([x], [y2], ax=ax[1])
plotting.save(fig, filename="line_scatter.png")

# Bar charts
counts = [1, 5, 8, 10, 5]
labels = [f"label {i}" for i in range(5)]
plotting.plot_bar_chart(counts, x=labels, filename="bar_chart.png")

# Histograms
x1 = np.random.normal(loc=0, scale=1, size=500)
x2 = np.random.normal(loc=1, scale=1, size=500)
plotting.plot_hist([x1, x2], bins=[50, 50], labels=["x1", "x2"], filename="hist.png")

# Violin plot
plotting.plot_violin([x1, x2], filename="violins.png")

#%% Plot time series data

# Plot continous raw time series data
X = np.random.normal(size=[1000, 5])
plotting.plot_time_series(X, filename="time_series1.png")

# Plot time series data with each channel on a separate subplot
plotting.plot_separate_time_series(X, filename="time_series2.png")

# Plot epoched time series data
sti_ind = np.arange(0, 1000, 100)
plotting.plot_epoched_time_series(
    X, time_index=sti_ind, pre=25, post=75, filename="time_series3.png"
)

#%% Plot matrices/connections

# Matrices
C = np.random.normal(size=[4, 11, 11])
plotting.plot_matrices(C, filename="matrices.png")

# Connections
C = np.random.normal(size=(38, 38))
C[[24, 25, 26, 27], [12, 13, 14, 15]] = 4
plotting.plot_connections(
    C,
    (f"ROI {i + 1}" for i in range(C.shape[0])),
    cmap="magma",
    text_color="white",
    filename="connections.png",
)

#%% Mode analysis plots

# Simulate data
hmm = simulation.HMM_MVN(
    n_modes=5,
    n_channels=11,
    n_samples=12800,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
)
hmm_stc = hmm.state_time_course

sm = simulation.MixedSine_MVN(
    n_samples=25600,
    n_modes=6,
    n_channels=80,
    relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.1],
    amplitudes=[6, 5, 4, 3, 2, 1],
    frequencies=[1, 2, 3, 4, 6, 8],
    sampling_frequency=250,
    means="zero",
    covariances="random",
)
sm_stc = sm.mode_time_course

# Alpha/mode time course
plotting.plot_alpha(hmm_stc, sm_stc, y_labels=["HMM", "SM"], filename="stc.png")

# Mode lifetimes
plotting.plot_state_lifetimes(hmm_stc, filename="slt.png")
