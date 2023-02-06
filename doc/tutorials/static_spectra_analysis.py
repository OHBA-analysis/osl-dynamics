"""
Static: Spectral Analysis
=========================
 
MEG data is useful because it has a high temporal resolution. We can take advantage of this by examining the spectral (i.e. frequency) content of the data. In this tutorial we will perform static spectral analysis on source space MEG data. This tutorial covers:
 
1. Getting the Data
2. Subject-Level Analysis
3. Group-Level Analysis

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/74hzk>`_ for the expected output.
"""

#%%
# Getting the Data
# ^^^^^^^^^^^^^^^^
# 
# We will use task MEG data that has already been source reconstructed. The experiment was a visuomotor task. This dataset is:
#
# - From 10 subjects.
# - Parcellated to 42 regions of interest (ROI). The parcellation file used was `fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz`.
# - Downsampled to 250 Hz.
# - Bandpass filtered over the range 1-45 Hz.
# 
# Download the dataset
# ********************
# 
# We will download example data hosted on `OSF <https://osf.io/zxb6c/>`_. Note, `osfclient` must be installed. This can be done in jupyter notebook by running::
#
#     !pip install osfclient
#

import os

def get_data(name):
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p zxb6c fetch Dynamics/data/datasets/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {name}")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset (approximately 140 MB)
get_data("notts_task_10_subj")

# List the contents of the downloaded directory containing the dataset
os.listdir("notts_task_10_subj")

#%%
# Load the data
# *************
# 
# We now load the data into osl-dynamics using the Data class. See the `Loading Data tutorial <https://osf.io/ejxut>`_ for further details.

from osl_dynamics.data import Data

data = Data("notts_task_10_subj")
print(data)

#%%
# For static analysis we just need the time series for the parcellated data. We can access this using the `time_series` method.

ts = data.time_series()

#%%
# `ts` a list of numpy arrays. Each numpy array is a `(n_samples, n_channels)` time series for each subject.
# 
# Subject-Level Analysis
# ^^^^^^^^^^^^^^^^^^^^^^
# 
# In this section, we will ignore the fact this data was collected in a task paradigm and will just aim to study the power spectrum of each subject.
# 
# Calculate power spectra
# ***********************
# 
# Using the data we just loaded, we want to calculate the power spectra for each channel (ROI) for each subject. We will use the `osl-dynamics.analysis.static.power_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/static/index.html#osl_dynamics.analysis.static.power_spectra>`_ function to do this. This function implements Welch's methods for calculating power spectra.
# 
# To use this function we need to specify at least two arguments:
#
# - `sampling_frequency`. This is to ensure we get the correct frequency axis for the power spectra.
# - `window_length`. This is the number of samples in the sliding window. The longer the window the smaller the frequency resolution of the power spectrum. Twice the sampling frequency is generally a good choice for this, which gives a frequency resolution of 0.5 Hz.
# 
# We can also specify an optional argument:
#
# - `standardize`. This will z-transform the data (for each subejct separately) before calculate the power spectra. This can be helpful if you want to examine power the fraction of power in a frequency band relative to the total power (across all frequencies) of the subject.

from osl_dynamics.analysis import static

f, psd = static.power_spectra(
    data=ts,
    sampling_frequency=250,
    window_length=500,
    standardize=True,
)

#%%
# `osl-dynamics.analysis.static.power_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/static/index.html#osl_dynamics.analysis.static.power_spectra>`_ returns two numpy arrays: `f`, which is the frequency axis of the power spectra in Hz, and `p`, which contains the power spectra.
# 
# Calculating power spectra can be time consuming. We will want to use the power spectra many times to make different plots. We don't want to have to calculate them repeatedly, so often it is convinent to save the `f` and `p` numpy arrays so we can load them later (instead of calculating them again). Let's save the spectra.

import numpy as np

os.makedirs("spectra", exist_ok=True)
np.save("spectra/f.npy", f)
np.save("spectra/psd.npy", psd)

#%%
# Plot the power spectra
# **********************
# 
# Let's first load the power spectra we previously calculated.

f = np.load("spectra/f.npy")
psd = np.load("spectra/psd.npy")

#%%
# To understand these arrays it's useful to print their shape:

print(f.shape)
print(psd.shape)

#%%
# We can see `f` is a 1D numpy array of length 256. This is the frequency axis of the power spectra. We can see `psd` is a subjects by channels (ROIs) by frequency array. E.g. `psd[0]` is a `(42, 256)` shaped array containing the power spectra for each of the 42 ROIs.
# 
# Let's plot the power spectrum for each ROI for the first subject. We will use the `osl_dynamics.utils.plotting <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/utils/plotting/index.html>`_ module to do the basic plotting. See the Plotting tutorial for further info.

from osl_dynamics.utils import plotting

plotting.plot_line(
    [f] * psd.shape[1],
    psd[0],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
)

#%%
# Each line in this plot is a ROI. We can see there's a lot of activity in the 1-20 Hz range. Let's zoom into the 1-45 Hz range.

plotting.plot_line(
    [f] * psd.shape[1],
    psd[0],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    x_range=[1, 45],
)

#%%
# Note, if you wanted to save this as an image you could pass a `filename="<filename>.png"` argument to this function. All functions in `osl_dynamics.utils.plotting <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/utils/plotting/index.html>`_ have this argument.
# 
# Rather than plotting the power spectrum for each ROI, let's average over channels to give a single line for each subject.

# Average over channels
psd_mean = np.mean(psd, axis=1)

# Plot the mean power spectrum for the first subject
plotting.plot_line(
    [f],
    [psd_mean[0]],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    x_range=[1, 45],
)

#%%
# We can add a shaded area around the solid line to give an idea of the variability around the mean.

# Standard deviation around the mean
psd_std = np.std(psd, axis=1)

# Plot with one sigma shaded
plotting.plot_line(
    [f],
    [psd_mean[0]],
    errors=[[psd_mean[0] - psd_std[0]], [psd_mean[0] + psd_std[0]]],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    x_range=[1, 45],
)

#%%
# Finally, let's plot the average power spectrum for each subject in the same figure.

plotting.plot_line(
    [f] * psd.shape[0],
    psd_mean,
    labels=[f"Subject {i + 1}" for i in range(psd.shape[0])],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    x_range=[1, 45],
)

#%%
# We can see there's quite a bit of variation in the static power spectrum for each subject. Some subjects have a pronounced alpha (around 10 Hz) peak. Some subjects has significant beta (around 20 Hz) activity, others don't.
# 
# Group-Level Analysis: Comparing Groups
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# We maybe interested in calculating the average power spectrum for a group of subjects. We have two options for how we do this:
# 
# 1. Concatenate the subject data for each group.
# 2. Average the subject-specific power spectra.
# 
# Calculate the group average power spectrum (all subjects)
# *********************************************************
# 
# First let's look at the group-level power spectrum for all 5 subjects in our dataset and we'll calculate this using the first option (concatenating the data).

# Concatenate the data
group_ts = np.concatenate(ts)
print(group_ts.shape)

# Calculate the static power spectra
f, group_psd1 = static.power_spectra(
    data=group_ts,
    sampling_frequency=250,
    window_length=500,
    standardize=True,
)
print(group_psd1.shape)

#%%
# We see `static.power_spectra` now returns a 2D numpy array containing the power spectrum at each ROI. Let's plot the group-level power spectrum averaged over channels.

# Average over channels
group_psd1_mean = np.mean(group_psd1, axis=0)

# Plot
plotting.plot_line(
    [f],
    [group_psd1_mean],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    x_range=[1, 45],
)

#%%
# Now, let's compare this to the group-level power spectrum we get by averaging the subject-specific power spectra we calculated previously.

# Average the subject-specific power spectra
# Note, psd_mean is the power spectrum for each subject averaged over channels
group_psd2_mean = np.mean(psd_mean, axis=0)

# Plot
plotting.plot_line(
    [f, f],
    [group_psd1_mean, group_psd2_mean],
    labels=["Data concatenated", "Averaged subject-specific"],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    x_range=[1, 45],
)

#%%
# We can see we get virtually identical power spectra with the two methods.
# 
# Calculate the group average power spectrum (sub-groups)
# *******************************************************
# 
# Now rather than computing a group average for all subjects, we could calculate the average over groups of subjects. For example, a common analysis is to compare healthy vs diseased groups. Let's divide our 5 subjects into two groups and compare the static power spectra (averaged over all channels) for each group.

# Group assignments:
# - 0 indicates assignment to the first group
# - 1 indicates assignment to the second group
assignments = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 1])

# Get the subject-specific power spectra for the subjects in each group
psd1 = psd[assignments == 0]
psd2 = psd[assignments == 1]
print(psd1.shape)
print(psd2.shape)

#%%
# We can see from the shape of each array we have the correct number of power spectra for each group. Now, let's average over channels and plot the spectra for each group.

# Average over channels
psd1_mean = np.mean(psd1, axis=1)
psd2_mean = np.mean(psd2, axis=1)

# Calculate group average
group_psd1_mean = np.mean(psd1_mean, axis=0)
group_psd2_mean = np.mean(psd2_mean, axis=0)

# Plot
plotting.plot_line(
    [f, f],
    [group_psd1_mean, group_psd2_mean],
    labels=["Group 1", "Group 2"],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    x_range=[1, 45],
)

#%%
# We can see group 2 shows much more alpha (10 Hz) activity compared to group 1.
# 
# Statistical Significance Testing
# ********************************
# 
# When we see differences in groups like this, we should perform a statistical significance test to rule out the possibility that we're observing the effect purely by chance. To compare the groups we'll use a **maximum statistic permutation test**. See the `Statistical Significance Testing tutorial <https://osf.io/ft3rm>`_ for a detailed explanation.

def null_distribution(vectors, real_assignments, n_perm):
    # Randomly generate group assignments by shuffling the real assignments
    # Note, for the first permutation we use the real group assignments
    group_assignments = [real_assignments]
    for i in range(n_perm - 1):
        random_assignments = np.copy(real_assignments)
        np.random.shuffle(random_assignments)
        group_assignments.append(random_assignments)

    # Make sure we don't have any duplicate permutations
    group_assignments = np.unique(group_assignments, axis=0)

    # Calculate null distribution
    null = []
    for assignment in group_assignments:
        # Assign subjects to their group
        group1 = vectors[assignment == 0]
        group2 = vectors[assignment == 1]

        # Calculate group means and absolute difference
        mean1 = np.mean(group1, axis=0)
        mean2 = np.mean(group2, axis=0)
        abs_diff = np.abs(mean1 - mean2)

        # Keep max stat
        null.append(abs_diff.max())

    return np.array(null)

# Prepare subject-specific vectors to compare
# We'll use the PSD over the range 1-45 Hz
keep = np.logical_and(f > 1, f < 45)
vectors = psd_mean[:, keep]

# Generate a null distribution
null = null_distribution(vectors, assignments, n_perm=1000)

# Calculate a threshold for significance
p_value = 0.05
thres = np.percentile(null, 100 * (1 - p_value), axis=0)

# See which elements are significant
vectors1 = np.mean(vectors[assignments == 0], axis=0)
vectors2 = np.mean(vectors[assignments == 1], axis=0)
abs_diff = np.abs(vectors1 - vectors2)
sig = abs_diff > thres

print("Significant frequencies:")
print(f[keep][sig])

# Plot with significant values highlighted
fig, ax = plotting.plot_line(
    [f, f],
    [group_psd1_mean, group_psd2_mean],
    labels=["Group 1", "Group 2"],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    x_range=[1, 45],
)
ax.axvspan(f[keep][sig].min(), f[keep][sig].max(), color="red", alpha=0.2)

#%%
# We can see the difference at 10 Hz between the two groups is unlikely to be due to chance.
# 
# Wrap Up
# ^^^^^^^
# 
# - We have shown how to calculate power spectra for individual subjects and groups.
# - We have shown how to test if group-level spectra are significantly different.
