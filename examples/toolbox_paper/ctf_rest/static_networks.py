"""CTF Rest Dataset: Static Network Analysis.

In this script we calculate static network metrics using source
reconstructed resting-state MEG data.

The examples/toolbox_paper/ctf_rest/get_data.py script can be used
to download the training data.
"""

import os
import numpy as np
import pandas as pd

from osl_dynamics.data import Data
from osl_dynamics import stats
from osl_dynamics.analysis import static, power, connectivity

def vec2mat(vec):
    n = int((1 + np.sqrt(1 + 8 * vec.shape[-1])) / 2)
    i, j = np.triu_indices(n, k=1)
    mat = np.zeros([vec.shape[0], n, n])
    mat[:, i, j] = vec
    mat[:, j, i] = vec
    return mat

def mat2vec(mat):
    i, j = np.triu_indices(mat.shape[-1], k=1)
    return mat[..., i, j]


# Output directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Load source data
data = Data(
    "training_data/networks",
    sampling_frequency=250,
    n_jobs=4,
)

# Calculate static power spectra
spectra = static.welch_spectra(data, n_jobs=4)
group = spectra.group_average()

# Frequency bands: delta, theta, alpha, beta
bands = [[1, 4], [4, 7], [7, 13], [13, 30]]

# Calculate power maps for different frequency bands
pow_map = spectra.power_maps(bands=bands)  # (subjects, freq_bands, parcels)

# Plot static networks
mean_pow_map = group.power_maps(bands=bands)  # average over subjects
power.save(
    mean_pow_map,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    plot_kwargs={
        "views": ["lateral"],
        "symmetric_cbar": True,
        "vmax": mean_pow_map.max(),  # puts all plots on the same scale
    },
    filename=f"{output_dir}/pow_.png",
)

# Calculate coherence networks for different frequency bands, keeping the
# top 5% of edges
mean_coh_net = group.coherence_networks(bands=bands, percentile=95)

# Plot coherence networks
connectivity.save(
    mean_coh_net,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    plot_kwargs={
        "edge_cmap": "Reds",
        "edge_vmin": 0,
        "edge_vmax": mean_coh_net.max(),
    },
    filename=f"{output_dir}/coh_.png",
)

# Calculate AEC networks for different frequency bands
aec = []
for l, h in [(1, 4), (4, 7), (7, 13), (13, 30)]:
    data.filter(low_freq=l, high_freq=h, use_raw=True)
    data.amplitude_envelope()
    data.standardize()
    aec.append(static.functional_connectivity(data.time_series()))
aec = np.array(aec)
aec = np.swapaxes(aec, 0, 1)  # (freq_bands, subjects, ...) -> (subjects, freq_bands, ...)

# Plot AEC networks
mean_aec = np.mean(aec, axis=0)  # average over subjects
for a in mean_aec:
    np.fill_diagonal(a, 0)
connectivity.save(
    mean_aec,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    threshold=0.95,
    plot_kwargs={
        "edge_cmap": "Reds",
        "edge_vmin": 0,
        "edge_vmax": mean_aec.max(),
    },
    filename=f"{output_dir}/aec_.png",
)

# Get group assignments (old = 1, young = 2)
demographics = pd.read_csv("training_data/demographics.csv")
age = demographics["age"].values
assignments = np.ones_like(age)
for i, a in enumerate(age):
    if a < 34:
        assignments[i] += 1

# Compare power maps for young vs old
result = stats.group_diff(pow_map, assignments, n_perm=1000)
pow_diff = result.copes  # pow_diff is group 1 (old) minus group 2 (young)
pvalues = result.pvalues

# Plot significant power map differences
power.save(
    pow_diff,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    filename=f"{output_dir}/pow_diff_.png",
)
power.save(
    pvalues,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    plot_kwargs={
        "cmap": "Greens_r",
        "symmetric_cbar": False,
        "vmin": 0,
        "vmax": 0.1,
        "bg_on_data": True,
    },
    filename=f"{output_dir}/pow_diff_pvalues_.png",
)

# Compare AEC networks for young vs old
aec = mat2vec(aec)
result = stats.group_diff(aec, assignments, n_perm=1000)
aec_diffs = result.copes  # aec_diffs is group 1 (old) minus group 2 (young)
pvalues = result.pvalues
aec_diffs[pvalues > 0.05] = 0

aec_diffs = vec2mat(aec_diffs)
connectivity.save(
    aec_diffs,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    filename=f"{output_dir}/aec_diff_.png",
)
