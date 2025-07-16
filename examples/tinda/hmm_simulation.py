"""Example script for simulating an HMM state time series and running TINDA on it.

"""

import os
import numpy as np
from matplotlib import pyplot as plt

from osl_dynamics import simulation
from osl_dynamics.analysis.tinda import tinda, plot_cycle, optimise_sequence


# Directory for plots
os.makedirs("figures", exist_ok=True)

# Simulate data
print("Simulating data")
sim = simulation.HMM_MVN(
    n_samples=25600,
    n_states=12,
    n_channels=38,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
)
sim_stc = sim.state_time_course

# Run TINDA
print("Running TINDA")
fo_density, fo_sum, stats = tinda(sim_stc)

print("Finding best circular sequence")
best_sequence = optimise_sequence(fo_density)

print("Finding strongest interval asymmetries")
mean_direction = np.squeeze(
    np.nanmean((fo_density[:, :, 0] - fo_density[:, :, 1]), axis=-1)
)
mean_direction[np.isnan(mean_direction)] = 0

# Find indices of largest 25% values
perc = 0.25
n_largest = int((np.prod(mean_direction.shape) - mean_direction.shape[0]) * perc)
largest_indices = np.abs(mean_direction).argsort(axis=None)[-n_largest:]

strongest_edges = np.zeros(mean_direction.shape)
# Set largest indices to 1
strongest_edges[np.unravel_index(largest_indices, mean_direction.shape)] = 1

# Plot
print("Plotting results")
plt.figure()
plt.imshow(mean_direction)
plt.colorbar()
plt.title("Mean direction")
plt.savefig("figures/tinda_sim_mean_direction.png")

plot_cycle(best_sequence, fo_density, strongest_edges, new_figure=True)
plt.savefig("figures/tinda_sim_cycle.png")
