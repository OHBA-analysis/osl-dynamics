"""
Static: Power Analysis
======================

In this tutorial we will perform static power analysis on source space MEG data. This tutorial covers:

1. Getting the data
2. Calculating power from spectra
"""

#%%
# Getting the data
# ^^^^^^^^^^^^^^^^
# We will use resting-state MEG data that has already been source reconstructed. This dataset is:
#
# - Parcellated to 38 regions of interest (ROI). The parcellation file used was `atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz`.
# - Downsampled to 250 Hz.
# - Bandpass filtered over the range 1-45 Hz.
#
# Download the dataset
# ********************
# We will download example data hosted on `OSF <https://osf.io/by2tc/>`_.

import os

def get_data(name, rename):
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

# Download the dataset (approximately 52 MB)
get_data("notts_meguk_giles_5_subjects", rename="source_data")

#%%
# Load the data
# ^^^^^^^^^^^^^
# We now load the data into osl-dynamics using the Data class. See the :doc:`Loading Data tutorial <../tutorials_build/1-1_data_loading>` for further details.

from osl_dynamics.data import Data

data = Data("source_data", n_jobs=4)
print(data)

#%%
# For static analysis we just need the time series for the parcellated data. We can access this using the `time_series` method.

ts = data.time_series()

#%%
# `ts` a list of numpy arrays. Each numpy array is a `(n_samples, n_channels)` time series for each subject.
#
# Calculate spectra
# ^^^^^^^^^^^^^^^^^
# First, we calculate the subject-specific power spectra. See the Static Power Spectra Analysis tutorial for more comprehensive description of power spectra analysis.

from osl_dynamics.analysis import static

# Calculate power spectra
spectra = static.welch_spectra(
    data=ts,
    sampling_frequency=250,
    window_length=500,
    standardize=True,
)

# Save
os.makedirs("spectra", exist_ok=True)
spectra.save("spectra/static.npz")

#%%
# Calculate power
# ^^^^^^^^^^^^^^^
# Let's first load the power spectra we previously calculated.

from osl_dynamics.analysis import spectral

spectra = spectral.Spectra.load("spectra/static.npz")

#%%
# To understand the spectra it's useful to print the shape of the frequency axis and the power spectra:

print(spectra.frequencies.shape)
print(spectra.psd.shape)

#%%
# We can see `spectra.frequencies` is a 1D numpy array of length 256. This is the frequency axis of the power spectra. We can see `spectra.psd` is a subjects by channels (ROIs) by frequency array. E.g. `spectra.psd[0]` is a `(38, 256)` shaped array containing the power spectra for each of the 38 ROIs.
#
# A useful property of a power spectrum is that the integral over a frequency range gives the power (or equivalently the variance of activity over the frequency range). osl-dynamics has a :mod:`osl_dynamics.analysis.power` module for performing power analyses.
#
# Let's say we are interested in alpha (10 Hz) power. We can calculate alpha power by averaging a power spectrum over a frequency range near 10 Hz. Typically, 8-13 Hz power is referred to as the 'alpha band'. Other common frequency bands are:
#
# - Delta: 1-4 Hz.
# - Theta: 4-8 Hz.
# - Beta: 13-30 Hz.
# - Gamma: 30+ Hz.
#
# The `power_maps` method calculates power from the spectra. Let's use it to calculate power for the alpha band.

# Calculate power in the alpha band (8-13 Hz) from the spectra
p = spectra.power_maps(frequency_range=[8, 13])

#%%
# Note, if `frequency_range` is not passed, `power_maps` will average the power spectrum over all frequencies. Pass `method="integral"` to integrate the power spectrum (the variance), or `bands=[[1, 4], [4, 8], [8, 13], [13, 30]]` to calculate several bands at once.
#
# We can print the shape of the `p` array to help understand what is contained within it.

print(p.shape)

#%%
# From this, we can see it is a subjects by ROIs array. It has averaged the power spectrum for each ROI separately. If we wanted the alpha power at each ROI for the first subject, we would use `p[0]`, which would be a `(38,)` shaped array.
#
# Plot power maps
# ^^^^^^^^^^^^^^^
# We can use :func:`power.save <osl_dynamics.analysis.power.save>` to visualize the power maps. Let's plot the group average.

from osl_dynamics.analysis import power

group_p = spectra.group_average().power_maps(frequency_range=[8, 13])

fig, ax = power.save(
    group_p,
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
)

#%%
# Or we can plot power maps for individual subjects, e.g.

fig, ax = power.save(
    p[:4],  # first four
    parcellation_file="atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz",
)
