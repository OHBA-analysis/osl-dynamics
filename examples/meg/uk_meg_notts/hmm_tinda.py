"""Example for running TINDA on MEG data from the Nottingham MEGUK dataset.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

from osl_dynamics.inference.modes import argmax_time_courses
from osl_dynamics.analysis.tinda import tinda, plot_cycle, optimise_sequence
from osl_dynamics.data import HMM_MAR


# Some settings
os.makedirs("figures", exist_ok=True)  # Directory for plots
do_bonferroni_correction = True

# Load precomputed HMM
print("Loading HMM")
hmm = HMM_MAR(
    "/well/woolrich/projects/uk_meg_notts/eo/natcomms18/results/Subj1-55_K-12/hmm.mat"
)

# Get state time courses
gamma = hmm.gamma()
stc = argmax_time_courses(gamma)

# Run TINDA on every subject
print("Running TINDA")
fo_density, fo_sum, stats = tinda(stc)

print("Finding best circular sequence. This might take a while.")
bestseq = optimise_sequence(fo_density)

# Stats
print("Finding statistically significant assymetries in interval densities")
fo_density = np.squeeze(fo_density)
tstat = np.nan * np.ones((fo_density.shape[:2]))
pval = np.nan * np.ones((fo_density.shape[:2]))
for i in range(fo_density.shape[0]):
    for j in range(fo_density.shape[1]):
        tstat[i, j], pval[i, j] = ttest_rel(
            fo_density[i, j, 0, :], fo_density[i, j, 1, :]
        )

n_tests_bonferroni = np.prod(pval.shape) - pval.shape[0]
if do_bonferroni_correction:
    significant_edges = pval < (0.05 / n_tests_bonferroni)
else:
    significant_edges = pval < 0.05

mean_direction = np.squeeze(
    np.mean((fo_density[:, :, 0] - fo_density[:, :, 1]), axis=-1)
)

# Plot
print("Plotting results")
plt.figure()
plt.imshow(mean_direction)
plt.colorbar()
plt.title("Mean direction")
plt.savefig("figures/tinda_meg_notts_mean_direction.png")

plot_cycle(bestseq, fo_density, significant_edges, newfigure=True)
plt.savefig("figures/tinda_meg_notts_cycle.png")
