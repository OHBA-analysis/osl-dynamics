"""
DyNeMo: Plotting Networks
=========================

In this tutorial we will plot networks from a DyNeMo model trained on source reconstructed MEG data. This tutorial covers:

1. Calculate regression spectra
2. PSDs
3. Power maps
4. Coherence networks
"""

#%%
# Download source data and inferred parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this tutorial, we'll download example data from `OSF <https://osf.io/by2tc/>`_.

import os

def get_data(name, rename):
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

def get_inf_params(name, rename):
    os.system(f"osf -p by2tc fetch inf_params/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

# Download the source data (approximately 52 MB)
get_data("notts_meguk_giles_5_subjects", rename="source_data")

# Download the inferred mixing coefficients (approximately 11 MB)
get_inf_params("tde_dynemo_notts_meguk_giles_5_subjects", rename="results/inf_params")

#%%
# Calculate regression spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We calculate the networks based on the regression spectra. See the :doc:`DyNeMo Regression Spectra tutorial <../tutorials_build/5-1_dynemo_regression_spectra>` for details.

import pickle

from osl_dynamics.data import Data
from osl_dynamics.analysis import spectral

# Source data trimmed to match the mixing coefficients
data = Data("source_data")
trimmed_data = data.trim_time_series(n_embeddings=15, sequence_length=200)

# Mixing coefficients
alpha = pickle.load(open("results/inf_params/alp.pkl", "rb"))

# Regression spectra for each subject and mode
spectra = spectral.regression_spectra(
    data=trimmed_data,
    alpha=alpha,
    sampling_frequency=250,
    frequency_range=[1, 45],
)
print(spectra)

#%%
# If you saved the spectra with `spectra.save`, you can load them with `spectral.Spectra.load`.
#
# PSDs
# ^^^^
# We can plot the power spectra to see what oscillations typically occur when a particular mode is active. We plot the regression coefficients (`psd_coefs`): the change in the PSD from the static PSD for a one standard deviation increase in a mode's mixing coefficient. See the :doc:`Spectral Analysis </user_guide/spectra>` page of the User Guide for more details.
#
# First, we average over subjects. The group average is weighted by the number of samples in each subject.

group = spectra.group_average()
print(group.psd_coefs.shape)

#%%
# `group.psd_coefs` is a (modes, channels, frequencies) array. Now let's plot the group-average PSD for each mode, averaged over channels. The shaded area is the standard error over channels.

fig, ax = group.plot_psd()

#%%
# Power maps
# ^^^^^^^^^^
# Now let's average the power spectra over frequency to calculate the power.

from osl_dynamics.analysis import power

p = group.power_maps()
print(p.shape)

#%%
# We can see `p` is a (modes, channels) array. The power maps are the change in power from the static power: red regions have more power when a mode is active and blue regions have less.

# Display the power maps (takes a few seconds to appear)
fig, ax = power.save(
    p,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    plot_kwargs={"symmetric_cbar": True},
)

#%%
# We can see recognizable functional networks, which gives us confidence in the DyNeMo fit. We also see the networks are more localized than typical HMM states. A DyNeMo fit often has one mode with a small covariance that takes most of the mixing coefficients: its power map is negative everywhere.
#
# Coherence networks
# ^^^^^^^^^^^^^^^^^^
# Next, let's visualize the coherence networks. The `coherence_networks` method averages the change in coherence (`coherence_coefs`) over frequency.

c = group.coherence_networks()
print(c.shape)

#%%
# We now have a (modes, channels, channels) array. This is the change in the cross spectrum, normalized by the static power (change in coherence), when a mode is active: it includes changes in both coherence and power. Let's show the 3% of edges with the largest changes (increases or decreases).

from osl_dynamics.analysis import connectivity

c = group.coherence_networks(percentile=97)

connectivity.save(
    c,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
)

#%%
# Red edges are increases and blue edges are decreases. The networks are in the same regions as the changes in power. We expect these networks will improve with more subjects.

# sphinx_gallery_thumbnail_number = 7
