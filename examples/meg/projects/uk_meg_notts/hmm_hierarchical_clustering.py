"""Example script for applying hierarchical clustering to match state time courses
from different runs.

The method is described here: https://www.biorxiv.org/content/10.1101/2023.01.18.524539v2
"""

import pickle
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

from osl_dynamics.inference import modes

# Load alphas (inferred state probabilities) from each run
alpha = []
for i in range(1, 6):
    a = pickle.load(open(f"results/run{i}/inf_params/alp.pkl", "rb"))  # (subjects, time, states)
    alpha.append(a)

# Labels (run and state) for each time course
labels = []
for r in range(len(alpha)):  # run
    for s in range(alpha[r][0].shape[1]):  # state
        labels.append(f"{r + 1}-{s + 1}")

# Use hierarchical clustering to match states from different runs and average:
# - alpha is (runs, subjects, time, states)
# - averaged_alpha is (subjects, time, states)
averaged_alpha, cluster_info = modes.average_runs(alpha, return_cluster_info=True)

# Plot dengrogram
fig, ax = plt.subplots()
R = hierarchy.dendrogram(
    cluster_info["linkage"],
    labels=labels,
    ax=ax,
)
ax.set_title("Hierarchical Clustering", fontsize=16)
ax.tick_params(axis="x", labelrotation=90)
ax.set_yticks([])
filename = "dendrogram.png"
print("Saving", filename)
plt.savefig(filename)
plt.close()

# Ordering of the state time course from each run after clustering
order = R["leaves"]

# Re-order the correlation between state time courses to clusters
corr_before = cluster_info["correlation"]
corr_after = corr_before.copy()[order, :][:, order]

# Plot correlation between states time courses
fig, ax = plt.subplots(ncols=2)
im = ax[0].imshow(corr_before)
im = ax[1].imshow(corr_after)
ax[0].set_title("Before")
ax[1].set_title("After")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
filename = "corr.png"
print("Saving", filename)
plt.savefig(filename)
plt.close()
