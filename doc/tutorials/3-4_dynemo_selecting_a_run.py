"""
DyNeMo: Selecting a Run
=======================

Training DyNeMo several times on the same data can give different networks: the modes come out in a different order and, where the data do not constrain the solution well, some networks can be split or combined differently from run to run. A good default is to analyse the run with the lowest variational free energy. This tutorial covers an alternative, which chooses a run based on the networks the runs found, using only the mode covariances:

1. Download the runs
2. Power maps from the mode covariances
3. Match the modes of different runs
4. Group the runs into families
5. Select a run
"""

#%%
# Download the runs
# ^^^^^^^^^^^^^^^^^
# In this tutorial, we'll download the mode covariances of 20 runs of a 6-mode TDE-DyNeMo model from `OSF <https://osf.io/by2tc/>`_. The model was trained on MEG data parcellated with a 38 region parcellation, prepared with 15 time-delay embeddings and 80 PCA components. Each run was trained with the same hyperparameters, the only difference being the random initialisation.

import os

def get_inf_params(name, rename):
    os.system(f"osf -p by2tc fetch inf_params/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

# Download the runs (approximately 3 MB)
get_inf_params("tde_dynemo_20_runs", rename="runs")

#%%
# Let's load the mode covariances of each run, the PCA components used to prepare the data and the variational free energy of each run.

import numpy as np

covariances = np.load("runs/covariances.npy")  # (runs, modes, 80, 80)
pca_components = np.load("runs/pca_components.npy")  # (38 * 15, 80)
free_energy = np.load("runs/free_energy.npy")  # (runs,)

print(covariances.shape)
print(pca_components.shape)

#%%
# Did the runs find the same solution? The free energies are very similar:

for run in np.argsort(free_energy):
    print(f"run {run}: {free_energy[run]:.3f}")

#%%
# The lowest free energy is a reasonable choice, but here the differences between runs are small, so it does not tell us which runs found the same networks or which solution is the most common. Instead, we can compare the networks directly.
#
# Power maps from the mode covariances
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To compare runs, we describe each mode by a feature that does not depend on the run: its power map. We can calculate this directly from the mode covariance, without inferring the mode time courses or calculating spectra. The covariances are of the time-delay embedded and PCA-reduced data; :func:`raw_covariances <osl_dynamics.analysis.post_hoc.raw_covariances>` takes them back to the parcels, and the diagonal gives the power of each parcel.

from osl_dynamics.analysis import post_hoc

raw_covs = post_hoc.raw_covariances(
    covariances,
    n_embeddings=15,
    pca_components=pca_components,
    zero_lag=True,
)
power_maps = np.diagonal(raw_covs, axis1=-2, axis2=-1)  # (runs, modes, parcels)

print(power_maps.shape)

#%%
# Match the modes of different runs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Next, we group the modes of all runs into networks with :func:`match_runs <osl_dynamics.inference.modes.match_runs>`. Each run's power maps are compared relative to the mean over its modes, and the modes are clustered by the correlation of their maps, never putting two modes of the same run in the same network. The correlation above which two modes count as the same network is found from the data: each mode's best match in another run is either the same network found again (a correlation close to 1) or a different network, and the threshold is the value that best separates the two.

from osl_dynamics.inference import modes

networks, threshold = modes.match_runs(power_maps, return_threshold=True)

print(f"threshold: {threshold:.2f}")
print(networks)

#%%
# `networks` gives the network of each mode of each run. Networks are numbered by the number of runs they are found in, most first. Like the modes, they are numbered from 0 in the arrays; we add 1 when printing and plotting. Let's see how often each network is found. This is a useful measure of how reproducible each network is.

n_runs = len(networks)
for network in range(networks.max() + 1):
    n = np.sum(np.any(networks == network, axis=1))
    print(f"Network {network + 1}: found in {n} of {n_runs} runs")

#%%
# Let's plot the average power map of the networks found in more than one run. We average the power maps of a network's modes, after subtracting the mean over modes in each run.

from osl_dynamics.analysis import power

relative_maps = power_maps - power_maps.mean(axis=1, keepdims=True)
n_found = [np.sum(np.any(networks == k, axis=1)) for k in range(networks.max() + 1)]
common = [k for k, n in enumerate(n_found) if n > 1]
network_maps = np.array([relative_maps[networks == k].mean(axis=0) for k in common])

fig, ax = power.save(
    network_maps,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    plot_kwargs={"symmetric_cbar": True},
    titles=[f"Network {k + 1} ({n_found[k]} runs)" for k in common],
)

#%%
# Group the runs into families
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Runs that found the same set of networks found the same solution. :func:`run_families <osl_dynamics.inference.modes.run_families>` groups them into families, largest first.

families = modes.run_families(networks)
for family in families:
    print(f"runs {family}: networks {np.sort(networks[family[0]]) + 1}")

#%%
# The largest family is the solution the model finds most often. Comparing the networks of different families shows how the solutions differ, e.g. whether a network found as one mode in some runs is split into two in others. Let's also look at the median free energy of each family.

for family in families:
    print(f"runs {family}: median free energy {np.median(free_energy[family]):.3f}")

#%%
# If the largest families are equally large, we choose the one with the lowest median free energy: among the solutions found equally often, the one that fits the data best. The median over a family is much less noisy than the free energy of a single run.
#
# Select a run
# ^^^^^^^^^^^^
# :func:`select_run <osl_dynamics.inference.modes.select_run>` does all of the above and selects the medoid of the largest family: the run whose modes agree best, on average, with the same networks in the other runs of the family. This is the most typical run of the most common solution. We pass the free energies for the tie-break.

run, info = modes.select_run(power_maps, free_energy=free_energy, return_info=True)

print(f"selected run: {run}")
print(f"runs in its family: {info['families'][info['family']]}")
print(f"typicality: {np.round(info['typicality'], 3)}")

#%%
# Let's plot the power maps of the selected run. These are the networks we would analyse. Each title gives the network the mode belongs to and the number of runs that network is found in.

fig, ax = power.save(
    relative_maps[run],
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    plot_kwargs={"symmetric_cbar": True},
    titles=[
        f"Mode {mode + 1}: network {k + 1} ({n_found[k]} runs)"
        for mode, k in enumerate(networks[run])
    ],
)

#%%
# `info` also contains the networks, the threshold, the number of runs each network is found in, and how well each family's runs agree and their median free energy. We would now use the selected run for the rest of the analysis, e.g. getting the inferred parameters, see :doc:`HMM/DyNeMo: Get Inferred Parameters <../tutorials_build/3-5_hmm_dynemo_get_inf_params>`.
#
# Notes:
#
# - **Number of runs.** The choice is only as reliable as the family sizes. With few runs, families are small and often equally large, so the tie-break decides. Train enough runs (e.g. 20) for the most common solution to stand out, and check the family sizes.
# - **Stability.** To see how stable the choice is, you can repeat the selection on random subsets of the runs (e.g. 16 of the 20, many times) and count how often the same family is chosen.
