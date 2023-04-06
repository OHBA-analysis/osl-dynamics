"""Example script for testing if evoked responses are statistically significant.

"""

import numpy as np
import matplotlib.pyplot as plt

from osl_dynamics.analysis import statistics

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
#
# Note:
# - We should use n_perm~=1000 for real analysis.
# - We use a threshold of 0.05 / 2 = 0.025 for the pvalue because we
#   perform a two-tail test.
pvalues = statistics.evoked_response_max_stat_perm(data=epochs, n_perm=100)
significant = pvalues < 0.025

# Epochs averaged over subjects
avg_epoch = np.mean(epochs, axis=0)

# Time axis
t = np.arange(n_samples) - pre

# Plot epoched time courses with significant time points highlighed
fig, ax = plt.subplots()
for i in range(n_modes):
    p = ax.plot(t, avg_epoch[:, i], label=f"Mode {i+1}")
    sig_times = t[significant[:, i]]
    if len(sig_times) > 0:
        y = 2.75 + i * 0.05
        ax.plot(
            (sig_times.min(), sig_times.max()),
            (y, y),
            color=p[0].get_color(),
            linewidth=4,
        )
ax.axvline(0, linestyle="--", color="black")
ax.legend(loc=1)
ax.set_xlabel("Time")
ax.set_ylabel("Evoked Response")

filename = "epochs.png"
print("Saving", filename)
plt.savefig(filename)
