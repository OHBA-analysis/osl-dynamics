"""Example script for testing if evoked responses are statistically significant.

"""

import numpy as np

from osl_dynamics.analysis import statistics
from osl_dynamics.utils import plotting

# Generate some random data
n_samples = 64
n_modes = 4
n_subjects = 24

epochs = np.random.randn(n_subjects, n_samples, n_modes)

# There is an early difference for mode 1 and later difference for mode 3
epochs[:, 10:20, 0] += 1.5
epochs[:, 15:30, 2] += 2

# Baseline correct (we consider samples 0-9 as pre-stimulus)
pre = 9
epochs -= np.mean(epochs[:, :pre], axis=1, keepdims=True)

# Get time points with a p-value < 0.05
pvalues = statistics.evoked_response_max_stat_perm(data=epochs, n_perm=100)

# Time axis and average over subjects
t = np.arange(n_samples) - pre
avg_epochs = np.mean(epochs, axis=0)

# Plot epoched time courses with significant time points highlighed
plotting.plot_evoked_response(
    t,
    avg_epochs,
    pvalues,
    significance_level=0.05,
    labels=[str(i + 1) for i in range(avg_epochs.shape[1])],
    x_label="Time",
    y_label="Evoked Response",
    filename="epochs.png",
)
