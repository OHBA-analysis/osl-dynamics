"""Example script for testing a network response with cluster-based permutations.

"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from osl_dynamics import stats
from osl_dynamics.utils import plotting

# Simulate baseline-corrected network responses: (subjects, time, networks)
n_subjects, n_time, n_networks = 20, 200, 4
sampling_frequency = 250
t = np.arange(n_time) / sampling_frequency - 0.2

epochs = np.random.normal(size=(n_subjects, n_time, n_networks))
epochs = gaussian_filter1d(epochs, sigma=3, axis=1)
epochs /= epochs.std()
epochs[:, 70:120, 0] += np.hanning(50)  # response in network 1
epochs -= epochs[:, t < 0].mean(axis=1, keepdims=True)

# Cluster-based permutation test: clusters are formed along time (the first
# target dimension) within each network
result = stats.evoked_response(epochs, n_perm=1000, correction="cluster")
print(result)
for mask, p in zip(result.clusters, result.cluster_pvalues):
    times = t[mask.any(axis=1)]
    network = np.where(mask.any(axis=0))[0][0] + 1
    print(f"Network {network}: {times[0]:.3f}-{times[-1]:.3f} s, p = {p:.3f}")

plotting.plot_evoked_response(
    t,
    epochs.mean(axis=0),
    result,
    labels=[f"Network {i + 1}" for i in range(n_networks)],
    x_label="Time (s)",
    y_label="Network activation",
    filename="network_response_cluster.png",
)

# Threshold-free cluster enhancement: no cluster-forming threshold, p-value per
# time point
result = stats.evoked_response(epochs, n_perm=1000, correction="tfce")
plotting.plot_evoked_response(
    t,
    epochs.mean(axis=0),
    result,
    labels=[f"Network {i + 1}" for i in range(n_networks)],
    x_label="Time (s)",
    y_label="Network activation",
    filename="network_response_tfce.png",
)
