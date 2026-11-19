"""
HMM: Plotting MEG Networks
==========================

In this tutorial we will plot networks from an HMM trained on source reconstructed MEG data. This tutorial covers:

1. Calculate multitaper spectra
2. PSDs
3. Power maps
4. Coherence networks
5. Coherence maps
6. Coherence vs Power
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

# Download the inferred state probabilities (approximately 11 MB)
get_inf_params("tde_hmm_notts_meguk_giles_5_subjects", rename="results/inf_params")

#%%
# Calculate multitaper spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We calculate networks based on multitaper spectra. See the :doc:`HMM Multitaper Spectra tutorial <../tutorials_build/4-1_hmm_multitaper_spectra>` for details.

import pickle

import numpy as np

from osl_dynamics.data import Data
from osl_dynamics.analysis import spectral

# Source data trimmed to match the state probabilities
data = Data("source_data")
trimmed_data = data.trim_time_series(n_embeddings=15, sequence_length=2000)

# State probabilities
alpha = pickle.load(open("results/inf_params/alp.pkl", "rb"))

# Multitaper spectra for each subject and state
spectra = spectral.multitaper_spectra(
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
# We can plot the power spectra to see what oscillations typically occur when a particular state is on. To help us visualize this, we first average over subjects.

group = spectra.group_average()
print(group.psd.shape)

#%%
# The group average is weighted by the number of samples in each subject. `group.psd` is a (states, channels, frequencies) array. The `plot_psd` method plots the PSD of each state averaged over channels. The shaded area is the standard error over channels.

fig, ax = group.plot_psd(values="psd")

#%%
# The states have similar PSDs. To highlight the differences, we can plot the change in the PSD from the static PSD (the average of the state PSDs weighted by the fractional occupancies). This is the default.

fig, ax = group.plot_psd()

#%%
# Power maps
# ^^^^^^^^^^
# The PSD is a function of frequency. To calculate power we need to average over a frequency range. The `power_maps` method does this. Let's look at the power across all frequencies for each state.

from osl_dynamics.analysis import power

p = group.power_maps(values="psd")
print(p.shape)

#%%
# We have a (states, channels) array. Let's see what the power map for each state looks like.

# Takes a few seconds for the power maps to appear
fig, ax = power.save(
    p,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
)

#%%
# Because the power is always positive and a lot of the states have power in similar regions, all the power maps look similar. To highlight differences we display the change in power from the static power. This is the default.

p = group.power_maps()

# Takes a few seconds for the power maps to appear
fig, ax = power.save(
    p,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    plot_kwargs={"symmetric_cbar": True},
)

#%%
# If we were interested in a particular frequency band we could use the `frequency_range` argument, e.g. `frequency_range=[8, 13]`, or calculate several bands at once with `bands=[[1, 4], [4, 8], [8, 13], [13, 30]]`.
#
# Coherence networks
# ^^^^^^^^^^^^^^^^^^
# Next, let's visualize the coherence networks. The `coherence_networks` method averages the coherence over frequency. By default, it gives the change in coherence from the static coherence.

from osl_dynamics.analysis import connectivity

c = group.coherence_networks()
print(c.shape)

#%%
# We can see `c` is a (states, channels, channels) array. Coherence networks are often noisy so averaging over a large number of subjects (20+) is typically needed to get clean coherence networks.
#
# Let's have a look at the coherence networks. The :func:`connectivity.save <osl_dynamics.analysis.connectivity.save>` function can be used to display a connectivity matrix (or set of connectivity matrices).

connectivity.save(
    c,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
)

#%%
# We can see there are a lot of connections. We want to select the most extreme edges. The `percentile` argument keeps the edges with the largest changes (increases or decreases). Let's keep the top 3%.

c = group.coherence_networks(percentile=97)

connectivity.save(
    c,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
)

#%%
# Non-negative matrix factorization (NNMF)
# ****************************************
# In the above code, we averaged over all frequencies to calculate the power maps and coherence networks. We expect brain activity to occur with phase-locking networks with oscillations at different frequencies. A data-driven approach for finding the frequency bands (referred to as 'spectral components') for phase-locked networks is to apply non-negative matrix factorization (NNMF) to the coherence spectra for each subject. The `decompose` method does this (see :func:`analysis.spectral.decompose_spectra <osl_dynamics.analysis.spectral.decompose_spectra>`). Let's use it to separate the coherence spectra into two frequency bands.

wb_comp = spectra.decompose(n_components=2)
print(wb_comp.shape)

#%%
# `wb_comp` refers to 'wideband component' because the spectral components cover a wide frequency range (we'll see this below). If we passed `n_components=4`, we would find more narrowband components. You can interpret the `wb_comp` array as weights for how much coherent (phase-locking) activity is occurring at a particular frequency for a particular component. It can be better understood if we plot it.

from osl_dynamics.utils import plotting

f = spectra.frequencies
plotting.plot_line(
    [f, f],  # we need to repeat twice because we fitted two components
    wb_comp,
    x_label="Frequency (Hz)",
    y_label="Spectral Component",
)

#%%
# The blue line is the first spectral component and the orange line is the second component. We can see the NNMF has separated the coherence spectra into two bands, one with a lot of coherence activity below ~22 Hz (blue line) and one above (orange line). In other words, the first spectral component contains coherent activity below 22 Hz and the second contains coherent activity mainly above 22 Hz.
#
# Now, instead of averaging the coherence spectra across all frequencies, let's look at the first spectral component. We pass the spectral components as the `bands` argument, which weights each frequency using the spectral component.

c = group.coherence_networks(bands=wb_comp, percentile=97)
print(c.shape)

#%%
# We can see `c` is (components, states, channels, channels) array. We're only interested in the first spectral component. We can plot this by passing a `component=0` argument to :func:`connectivity.save <osl_dynamics.analysis.connectivity.save>`.

connectivity.save(
    c,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    component=0,
)

#%%
# Coherence maps
# ^^^^^^^^^^^^^^
# We can display the coherence as a spatial map rather than a graphical network by averaging the edges for each parcel.

c = group.coherence_networks(bands=wb_comp)
c_map = connectivity.mean_connections(c)

fig, ax = power.save(
    c_map,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
    plot_kwargs={"symmetric_cbar": True},
)

#%%
# Coherence vs Power
# ^^^^^^^^^^^^^^^^^^
# We may be interested in how the coherence relates to power for each state. We can do this by producing scatter plots of coherence vs power. Let's see how to do this.
#
# Let's first look at the group average for the first spectral component.

# Calculate power
mean_p = group.power_maps(values="psd", bands=wb_comp)[0]
print(mean_p.shape)

# Calculate coherence
mean_c = group.coherence_networks(values="coherence", bands=wb_comp)[0]
print(mean_c.shape)

#%%
# We have a `(n_channels,)` array for each state for the power, but a `(n_channels, n_channels)` array for each state for the coherence. To reduce the coherence down to a 1D array for each state, we can calculate the average pairwise coherence for each channel. We can do this with the :func:`connectivity.mean_connections <osl_dynamics.analysis.connectivity.mean_connections>` function.

mean_c = connectivity.mean_connections(mean_c)
print(mean_c.shape)

#%%
# Now we can make a scatter plot of coherence vs power for each state.

fig, ax = plotting.plot_scatter(
    mean_p,
    mean_c,
    labels=[f"State {i}" for i in range(1, mean_p.shape[0] + 1)],
    x_label="Power (a.u.)",
    y_label="Coherence",
)

#%%
# We can see the different states have different power/coherence properties. E.g. some states typically have more power/coherence at all parcels. Calling the methods on `spectra` instead of `group` gives the power and coherence for each subject. If we average over parcels we get the following.

# Power for each subject, averaged over parcels
p = spectra.power_maps(values="psd", bands=wb_comp)
mean_p = np.mean(p[:, 0], axis=-1)
print(mean_p.shape)

# Coherence for each subject, averaged over parcels
c = spectra.coherence_networks(values="coherence", bands=wb_comp)
mean_c = np.mean(c[:, 0], axis=(-2, -1))
print(mean_c.shape)

# Plot
fig, ax = plotting.plot_scatter(
    mean_p,
    mean_c,
    x_label="Power (a.u.)",
    y_label="Coherence",
)

#%%
# Each scatter point is a subject here. We also see different subjects have different power/coherence characteristics.

# sphinx_gallery_thumbnail_number = 18
