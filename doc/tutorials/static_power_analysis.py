"""
Static: Power Analysis
======================

In this tutorial we will perform static power analysis on source space MEG data. This tutorial covers:

1. Getting the data
2. Calculating power from spectra

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/uvmpa>`_ for the expected output.
"""

#%%
# Getting the data
# ^^^^^^^^^^^^^^^^
# We will use resting-state MEG data that has already been source reconstructed. This dataset is:
#
# - From 51 subjects.
# - Parcellated to 52 regions of interest (ROI). The parcellation file used was `Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz`.
# - Downsampled to 250 Hz.
# - Bandpass filtered over the range 1-45 Hz.
#
# Download the dataset
# ********************
# We will download example data hosted on `OSF <https://osf.io/by2tc/>`_.


import os

def get_data(name, rename):
    if rename is None:
        rename = name
    if os.path.exists(rename):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.makedirs(rename, exist_ok=True)
    os.system(f"unzip -o {name}.zip -d {rename}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {rename}"

# Download the dataset (approximately 720 GB)
get_data("notts_mrc_meguk_glasser", rename="source_data")

#%%
# Load the data
# ^^^^^^^^^^^^^
# We now load the data into osl-dynamics using the Data class. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details.


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


import numpy as np
from osl_dynamics.analysis import static

# Calculate power spectra
f, psd = static.welch_spectra(
    data=ts,
    sampling_frequency=250,
    window_length=500,
    standardize=True,
)

# Save
os.makedirs("spectra", exist_ok=True)
np.save("spectra/f.npy", f)
np.save("spectra/psd.npy", psd)

#%%
# Calculate power
# ^^^^^^^^^^^^^^^
# Let's first load the power spectra we previously calculated.


f = np.load("spectra/f.npy")
psd = np.load("spectra/psd.npy")

#%%
# To understand these arrays it's useful to print their shape:


print(f.shape)
print(psd.shape)

#%%
# We can see `f` is a 1D numpy array of length 256. This is the frequency axis of the power spectra. We can see `psd` is a subjects by channels (ROIs) by frequency array. E.g. `psd[0]` is a `(52, 256)` shaped array containing the power spectra for each of the 52 ROIs.
#
# A useful property of a power spectrum is that the integral over a frequency range gives the power (or equivalently the variance of activity over the frequency range). osl-dynamics has a `analysis.power <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html>`_ module for performing power analyses.
#
# Let's say we are interested in alpha (10 Hz) power. We can calculate alpha power by integrating a power spectrum over a frequency range near 10 Hz. Typically, 7-13 Hz power is referred to as the 'alpha band'. Other common frequency bands are:
#
# - Delta: 1-4 Hz.
# - Theta: 4-7 Hz.
# - Beta: 13-30 Hz.
# - Gamma: 30+ Hz.
#
# osl-dynamics has a `analysis.power.variance_from_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.variance_from_spectra>`_ function to calculate power from a spectrum. Let's use this function to calculate power for the alpha band.


from osl_dynamics.analysis import power

# Calculate power in the alpha band (8-12 Hz) from the spectra
p = power.variance_from_spectra(f, psd, frequency_range=[7, 13])

#%%
# Note, if `frequency_range` is not passed, `power.variance_from_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.variance_from_spectra>`_ will integrate the power spectrum over all frequencies.
#
# We can print the shape of the `p` array to help understand what is contained within it.


print(p.shape)

#%%
# From this, we can see it is a subjects by ROIs array. It has integrated the power spectrum for each ROI separately. If we wanted the alpha power at each ROI for the first subject, we would use `p[0]`, which would be a `(52,)` shaped array.
#
# Plot power maps
# ^^^^^^^^^^^^^^^
# We can use `power.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.save>`_ to visualise the power maps. Let's plot the group average.


group_p = np.mean(p, axis=0)

fig, ax = power.save(
    group_p,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    plot_kwargs={"symmetric_cbar": True},
)

#%%
# Or we can plot power maps for individual subjects, e.g.


fig, ax = power.save(
    p[:5],  # first five
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    plot_kwargs={"symmetric_cbar": True},
)

