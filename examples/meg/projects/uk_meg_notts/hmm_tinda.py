"""Example for running TINDA on MEG data from the Nottingham MEGUK dataset.

"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

from osl_dynamics.inference import modes
from osl_dynamics.analysis.tinda import tinda, plot_cycle, optimise_sequence

def get_data(name, output_dir):
    if os.path.exists(output_dir):
        print(f"{output_dir} already downloaded. Skipping..")
        return
    os.system(f"osf -p by2tc fetch trained_models/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {output_dir}")
    os.remove(f"{name}.zip")
    print(f"Data downloaded to: {output_dir}")

# Settings
os.makedirs("figures", exist_ok=True)  # Directory for plots
do_bonferroni_correction = True

# We will download example data hosted on osf.io/by2tc.
get_data("tde_hmm_notts_mrc_meguk_giles", output_dir="notts_tde_hmm")

# Get the state time course
alpha = pickle.load(open("notts_tde_hmm/alpha.pkl", "rb"))
stc = modes.argmax_time_courses(alpha)

# Run TINDA on every subject
print("Running TINDA")
fo_density, fo_sum, stats = tinda(stc)

print("Finding best circular sequence. This might take a while.")
best_sequence = optimise_sequence(fo_density)

# Stats
print("Finding statistically significant assymetries in interval densities")
fo_density = np.squeeze(fo_density)
tstat = np.nan * np.ones((fo_density.shape[:2]))
pval = np.nan * np.ones((fo_density.shape[:2]))
for i in range(fo_density.shape[0]):
    for j in range(fo_density.shape[1]):
        tstat[i, j], pval[i, j] = ttest_rel(fo_density[i, j, 0, :], fo_density[i, j, 1, :])

n_tests_bonferroni = np.prod(pval.shape) - pval.shape[0]
if do_bonferroni_correction:
    significant_edges = pval < (0.05 / n_tests_bonferroni)
else:
    significant_edges = pval < 0.05

mean_direction = np.squeeze(np.mean((fo_density[:, :, 0] - fo_density[:, :, 1]), axis=-1))

# Plot
print("Plotting results")
plt.figure()
plt.imshow(mean_direction)
plt.colorbar()
plt.title("Mean direction")
plt.savefig("figures/tinda_meg_notts_mean_direction.png")

plot_cycle(best_sequence, fo_density, significant_edges, new_figure=True)
plt.savefig("figures/tinda_meg_notts_cycle.png")
