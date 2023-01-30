"""
Static Network Analysis
=======================

In this tutorial we will perform static network analysis on source space MEG data. This tutorial covers:

1. Getting the Data
2. Power Spectra Analysis
3. Power Analysis
4. Network Analysis
5. Statistical Significance Testing

The input to this script is:

- A set of time series (one for each subject you have). In this tutorial we will download some example data.

The output of this script is:

- A plot of the power spectrum for each subject.
- Plots of the static power as a surface heat map for a particular frequency band.
- Glass brain plots of static networks for a particular frequency band. We will use the amplitude envelope correlation (AEC) for our measure of connectivity.

Note, this webpage does not contain the output of each cell. We advise downloading the notebook and working through it locally on your machine. The expected output of this script can be found `here <https://osf.io/a24bn>`_.
"""

#%%
# Getting the Data
# ^^^^^^^^^^^^^^^^
# 
# We will use eyes open resting-state data that has already been source reconstructed. We call this the 'Nottingham dataset'. This dataset is:
#
# - From 10 subjects.
# - Parcellated to 42 regions of interest (ROI). The parcellation file used was `fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz`.
# - Downsampled to 250 Hz.
# - Bandpass filtered over the range 1-45 Hz.
# 
# Download the dataset
# ********************
# 
# We will download the data from a project hosted on OSF.

import os

def get_notts_data():
    """Downloads the Nottingham dataset from OSF."""
    if os.path.exists("notts_dataset"):
        return "notts_dataset already downloaded. Skipping.."
    os.system("osf -p zxb6c fetch Dynamics/notts_dataset.zip")
    os.system("unzip -o notts_dataset.zip -d notts_dataset")
    os.remove("notts_dataset.zip")
    return "Data downloaded to: notts_dataset"

# Download the dataset (it is 113 MB)
get_notts_data()

# List the contents of the downloaded directory containing the dataset
os.listdir('notts_dataset')

#%%
# Load the data
# *************
# 
# We now load the data into osl-dynamics using the Data object. This is a python class which has a lot of useful methods that can be used to modify the data. See the `API reference guide <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_ for further details.

from osl_dynamics.data import Data

data = Data("notts_dataset")

# Display some summary information
print(data)

#%%
# Note, when we load data using the Data object, it creates a `/tmp` directory which is used for storing temporary data. This directory can be safely deleted after you run your script. You can specify the name of the temporary directory by pass the `store_dir="..."` argument to the Data object.
# 
# For static analysis we just need the time series for the parcellated data. We can access this using the `Data.time_series` method. This returns a list of numpy arrays. Each numpy array is a `(n_samples, n_channels)` time series for each subject.

# Get the parcellated data time series as a list of numpy arrays
ts = data.time_series()

#%%
# Power Spectra Analysis
# ^^^^^^^^^^^^^^^^^^^^^^
# 
# MEG data is useful because it has a high temporal resolution. We can take advantage of this by examining the spectral (i.e. frequency) content of the data.
# 
# Calculate the power spectra
# ***************************
# 
# Using the data we just loaded, we want to calculate the power spectra for each channel (ROI) for each subject. We will use the `osl-dynamics.analysis.static.power_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/static/index.html#osl_dynamics.analysis.static.power_spectra>`_ function to do this. This function implements Welch's methods for calculating power spectra.
# 
# To use this function we need to specify at least two arguments:
#
# - `sampling_frequency`. This is to ensure we get the correct frequency axis for the power spectra.
# - `window_length`. This is the number of samples in the sliding window. The longer the window the smaller the frequency resolution of the power spectrum. Twice the sampling frequency is generally a good choice for this, which gives a frequency resolution of 0.5 Hz.
# 
# We can also specify an optional argument:
# - `standardize`. This will z-transform the data (for each subejct separately) before calculate the power spectra. This can be helpful if you want to examine power the fraction of power in a frequency band relative to the total power (across all frequencies) of the subject.

from osl_dynamics.analysis import static

# Calculate static power spectra
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
# We can see `f` is a 1D numpy array of length 256. This is the frequency axis of the power spectra. We can see `psd` is a subjects by channels (ROIs) by frequency array. E.g. `p[0]` is a `(42, 256)` shaped array containing the power spectra for each of the 42 ROIs.
# 
# Let's plot the power spectrum for each ROI for the first subject. We will use the `osl_dynamics.utils.plotting <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/utils/plotting/index.html>`_ module to do the basic plotting.

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
# Power Analysis
# ^^^^^^^^^^^^^^
# 
# Next, we will use the static power spectra we previously calculated to plot power maps for each subject. This is a plot of power at each ROI as a 2D heat map on the surface of the brain.
# 
# Calculate power (in a frequency band)
# *************************************
# 
# A useful property of a power spectrum is that the integral over a frequency range gives the power (or equivalently the variance of activity over the frequency range). osl-dynamics has a `analysis.power <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html>`_ module for performing power analyses.
# 
# Let's say we are interested in alpha (10 Hz) power. We can calculate alpha power by integrating a power spectrum over a frequency range near 10 Hz. Typically, 8-12 Hz power is referred to as the 'alpha band'. Other common frequency bands are:
#
# - Delta: 1-4 Hz.
# - Theta: 4-8 Hz.
# - Alpha: 8-12 Hz.
# - Beta: 12-30 Hz.
# - Gamma: 30+ Hz.
# 
# osl-dynamics has a `analysis.power.variance_from_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.variance_from_spectra>`_ function to calculate power from a spectrum. Let's use this function to calculate power for the alpha band.

from osl_dynamics.analysis import power

# Calculate power in the alpha band (8-12 Hz) from the spectra
p = power.variance_from_spectra(f, psd, frequency_range=[8, 12])

#%%
# Note, if `frequency_range` is not passed, `power.variance_from_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.variance_from_spectra>`_ will integrate the power spectrum over all frequencies.
# 
# We can print the shape of the `p` array to help understand what is contained within it.

print(p.shape)

#%%
# From this, we can see it is a subjects by ROIs array. It has integrated the power spectrum for each ROI separately. If we wanted the alpha power at each ROI for the first subject, we would use `p[0]`, which would be a `(42,)` shaped array.
# 
# Differences in power between groups
# ***********************************
# 
# We are often interested in comparing different groups of subjects. Using the `p` array we can easily calculate the group mean of subsets of the full dataset. E.g. let's say subjects \[0, 3, 4\] belong to one group and \[1, 2, 5, 6, 7, 8, 9\] belong to another group. We can calculate the group means with:

# Get the power arrays for group 1 [0, 3, 4]
p1 = p[[0, 3, 4]]

# Get the power arrays for group 2 [1, 2, 5, 6, 7, 8, 9]
p2 = p[[1, 2, 5, 6, 7, 8, 9]]

# Check the groups have the correct number of subjects
print(p1.shape)
print(p2.shape)

# Calculate group means
p1_mean = np.mean(p1, axis=0)
p2_mean = np.mean(p2, axis=0)

# Plot (it takes a few seconds for the brain maps to be shown)
power.save(
    [p1_mean, p2_mean],
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# `power.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.save>`_ also has a `filename` argument that can be used to save these images to a file.
# 
# It hard to see a difference between the groups when we plot the power. It's often more useful to plot the power relative to a reference. In this case, we can plot the difference in power between the groups, let's do this.

# Calculate the difference in power between the groups
p_diff = p1_mean - p2_mean

# Plot
power.save(
    p_diff,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# We can see there is a difference in activity in the posterior and sensorimotor regions between the two groups (group 1 has more sensorimotor activity in the alpha band).
# 
# If you just want to plot the static power across all subjects, you can do the same without separating the power array.

# Group mean across all subjects
p_mean = np.mean(p, axis=0)

# Plot
power.save(
    p_mean,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# Network Analysis
# ^^^^^^^^^^^^^^^^
# 
# Next, we will estimate static networks for each subject. For this we need to define a metric for connectivity between ROIs. There are a lot of options for this. In this tutorial we'll look at the amplitude envelope correlation (AEC).
# 
# Calculate AEC
# *************
# 
# AEC can be calculated from the parcellated time series directly. First, we need to prepare the parcellated data. Previously we loaded the data using the `Data object <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_. Fortunately, the `Data object <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_ has a `Data.prepare` method that makes this easy. Let's prepare the data for calculate the AEC network for activity in the alpha band.

# Before we can prepare the data we must specify the sampling frequency
# (this is needed to bandpass filter the data)
data.set_sampling_frequency(250)

# Calculate amplitude envelope data for the alpha band (8-12 Hz)
data.prepare(low_freq=8, high_freq=12, amplitude_envelope=True)

# Get the amplitude envelope time series for each subject (ts is a list of numpy arrays)
ts = data.time_series()

#%%
# Note, the `Data.time_series` returns the latest prepared data. If we call `Data.prepare` again, it will prepare the data starting from the original raw data, i.e. you **cannot** chain multiple calls to `Data.prepare`.
# 
# Next, we want to calculate the correlation between amplitude envelopes. osl-dynamics has the `analysis.static.functional_connectivity <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/static/index.html#osl_dynamics.analysis.static.functional_connectivity>`_ function for this.

# Calculate the correlation between amplitude envelope time series
aec = static.functional_connectivity(ts)

#%%
# We can understand the `aec` array by printing its shape.

print(aec.shape)

#%%
# We can see it is a subject by ROIs by ROIs array. It contains all pairwise connections between ROIs.
# 
# Plot AEC networks
# *****************
# 
# Now that we have the AEC network for each subject, let's visualise them. Let's first plot the AEC network for the first subject. We can use the `analysis.connectivity.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_ function to display a network as a glass brain plot.

from osl_dynamics.analysis import connectivity

# Plot the network for the first subject
connectivity.save(
    aec[0],
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# In this plot we see every pairwise connection. Often, we're just interested in the strong connections - this helps us to avoid interpreting connections that are simply due to noise.
# 
# We can use the `connectivity.threshold <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.threshold>`_ function to select the strongest connections. If we just want to select the top few connections we just need to pass the `percentile` argument. The `connectivity.threshold <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.threshold>`_ function thresholds each subject's network separately.

# Keep the top 5% of conenctions
thres_aec = connectivity.threshold(aec, percentile=95)

#%%
# Let's plot the thresholded AEC networks for the first 3 subjects.

connectivity.save(
    thres_aec[:3],
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# We can see the networks more clearly now. In general alpha band activity is in posterior regions. We also observed there is significant subject-to-subject variation.
# 
# Note, we can also plot an AEC network as a 3D glass brain plot using the `glassbrain` argument. Let's do this for the group mean across all subjects.

# Calculate group mean across all subjects
aec_mean = np.mean(aec, axis=0)

# Threshold the top 5%
thres_aec_mean = connectivity.threshold(aec_mean, percentile=95)

# Display the network
connectivity.save(
    thres_aec_mean,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    glassbrain=True,
)

#%%
# Averaging over a large group, we see the network come out more cleanly. This is simple due to the data being noisy which makes estimating networks hard. Averaging over subjects helps remove this noise.
# 
# In the group average network we can see the strongest connections are in posterior regions as expected.
# 
# Data-driven thresholding for selecting network connections
# **********************************************************
# 
# Another option is rather than specifying a percentile by hand to threshold the connections, we can use a Gaussian Mixture Model (GMM) fit with two components (an 'on' and an 'off' component) to determine a threshold for selecting connections. The way this works is we fit two Gaussians to the distribution of connections. To understand this, let's first examine the distribution of connections.

import matplotlib.pyplot as plt

def plot_dist(values):
    """Plots a histogram."""
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(values.flatten(), bins=50, histtype="step")
    ax.set_xlabel("AEC")
    ax.set_ylabel("Number of edges")

# Plot distribution of connections
plot_dist(aec_mean)

#%%
# We see there is a cluster of connections between AEC=0 and 0.4 and another at AEC=1. The AEC=1 connections are on the diagonal of the connectivity matrix. Let's remove these to examine the distribution of off-diagonal elements, which is what we're interested in.

# Fill diagonal with nan values
# (nan is prefered to zeros because a zeo value will be included in the distribution, nans won't)
np.fill_diagonal(aec_mean, np.nan)

# Note, np.fill_diagonal alters the aec_mean array in place,
# i.e. we don't need to do aec_mean = np.fill_diagonal(aec_mean, np.nan)

# Plot distribution of connections
plot_dist(aec_mean)

#%%
# We can see there is a peak around AEC=0.05 and a long tail for higher values. We want the connections around the AEC=0.05 peak to be captured by a Gaussian and the long tail to be captured by another Gaussian. Let's fit a two component Gaussian to this distribution. Fortunately, osl-dynamics has a function to do this for us: `analysis.connectivity.fit_gmm <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.fit_gmm>`_. This function returns the threshold (as a percentile) that determines the Gaussian component a connection belows to.

# Fit a two-component Gaussian mixture model to the connectivity matrix
#
# We pass the standardize=False argument because we don't want to alter the
# distribution before fitting the GMM.
percentile = connectivity.fit_gmm(aec_mean, show=True)
print("Percentile:", percentile)

#%%
# Let's now use the data-driven threshold to select connections in our network.

# Threshold
thres_aec_mean = connectivity.threshold(aec_mean, percentile=percentile)

# Display the network
connectivity.save(
    thres_aec_mean,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    glassbrain=True,
)

#%%
# We can a lot more connections now. We can be more extreme with the connections we choose by enforcing the likelihood a of a connection belonging to the 'off' component is below a certain p-value. For example, if we wanted to show the connections belonging to the 'on' GMM component, that had a likelihood of less than 0.01 of belonging to the 'off' component, we could do the following:

# Fit a two-component Gaussian mixture model to the connectivity matrix
# ensuring the threshold is beyond a p-value of 0.01 of belonging to the off component
percentile = connectivity.fit_gmm(aec_mean, p_value=0.01, show=True)
print("Percentile:", percentile)

#%%
# We can see the threshold has moved much more to the right now. Let's example the network with this threshold.

# Threshold
thres_aec_mean = connectivity.threshold(aec_mean, percentile=percentile)

# Display the network
connectivity.save(
    thres_aec_mean,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    glassbrain=True,
)

#%%
# Now we have a data driven threshold which shows good posterior connectivity in the alpha band.
# 
# Note, osl-dynamics has a wrapper function to return the thresholded network directly (so you don't need to threshold yourself): `connectivity.gmm_threshold <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.gmm_threshold>`_. Using this function, we can threshold connectivity matrix in one line::
#
#     thres_aec_mean = connectivity.gmm_threshold(aec_mean, p_value=0.01)
# 
# Statistical Significance Testing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Let's say we are interested in comparing two groups. We can calculate the mean static power (at each ROI) for each group. Alternatively, we can calculate the mean AEC for each group. We can represent each of these quantities as a vector. The mean power for each group was calculated in Section 3. This is already a vector. To turn the AEC connectivity matrix into a vector we can take the upper triangle.
# 
# Maximum statistic permutation testing
# *************************************
# 
# In this section, let's focus on the mean power for each group. We want to see if the two group means are significantly different. In other words, we want to show the difference between the mean power for the two groups is not due to chance. We will use **maximum statistic permutation testing** to do this. This involves a few steps:
# 
# 1. Obtain a estimate for the power at each parcel for each subject, in our case, this is a 42 dimensional vector for each subject.
# 2. Randomly assign each subject a binary label: 0 or 1. When we do this we take into account the number of subjects we have for each group. E.g. if we have 100 subjects, 50 in one group and 50 in the other, then we would assign group labels to the subjects with a 50/50 chance of being in each group. If we had 100 subjects with 20 in a group and 80 in the other, we would assign group labels with a 20/80 chance split.
# 3. Calculate a mean for the subjects labelled with 0 and another mean for the subjects labelled with 1. These means are vectors of length 42.
# 4. Calculate the difference between the means. This gives a single vector of length 42.
# 5. Record the maximum value in the 42 dimensional vector. This steps is why this method is known as a **maximum statistic** test. This step is necessary to account for the fact that you're making multiple comparisons with the 42 dimensional vector. Note, if we're interested in testing positive and negative differences between the means we also record the minimum value of the 42 dimensional vector.
# 6. Repeat a large number of times, e.g. 1000. This gives a distribution of possible values the maximum/minimum statistic can take when the groups have been assigned randomly. This distribution is known as the **null distribution**. Note, we include one entry in the null distribution that corresponds to the real group assignments.
# 7. For a p-value of 0.05, we look at the bottom 2.5 percentile of the minimum statistic null distribution and top 97.5 percentile of the maximum statistic null distribution.
# 8. Calculate the group means using the real assignment of subjects to each group and calculate the difference.
# 9. The above gives a 42 dimensional vector. The elements of this vector that are below (above) the 2.5 (97.5) percentile are deemed to be significant with a p-value of 0.05.
# 
# Let's implement the above in code to see if the difference between our two groups is significant.

def null_distribution(vectors, real_assignments, n_perm):
    """Builds a max-stat and min-stat null distribution."""
    n_subjects = vectors.shape[0]

    # Randomly generate group assignments by shuffling the real assignments
    # Note, for the first permutation we use the real group assignments
    group_assignments = [real_assignments]
    for i in range(n_perm - 1):
        random_assignments = np.copy(real_assignments)
        np.random.shuffle(random_assignments)
        group_assignments.append(random_assignments)

    # Calculate null distributions
    max_dist = []
    min_dist = []
    for assignment in group_assignments:
        # Assign subjects to their group
        group1 = vectors[assignment == 0]
        group2 = vectors[assignment == 1]

        # Calculate group means and difference
        mean1 = np.mean(group1, axis=0)
        mean2 = np.mean(group2, axis=0)
        diff = mean1 - mean2

        # Calculate max/min statistics
        max_stat = np.max(diff)
        if not np.isnan(max_stat):
            max_dist.append(max_stat)
        min_stat = np.min(diff)
        if not np.isnan(min_stat):
            min_dist.append(min_stat)

    return np.array(max_dist), np.array(min_dist)

# First let's create an array for the real group assignments
# group 1 subjects are [0, 3, 4] and group 2 subjects are [1, 2, 5, 6, 7, 8, 9]
real_assignments = np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 1])

# Let's add an effect in the first parcel of the 1st group
p[real_assignments == 0, 0] += 0.1

# Calculate the group difference
mean1 = np.mean(p[real_assignments == 0], axis=0)
mean2 = np.mean(p[real_assignments == 1], axis=0)
p_diff = mean1 - mean2

# Create the null using 1000 permutations
max_null_dist, min_null_dist = null_distribution(p, real_assignments, n_perm=1000)

# Get the threshold for significance for a particular p-value
p_value = 0.05
bottom_percentile = p_value * 100 / 2
bottom_threshold = np.percentile(min_null_dist, bottom_percentile)
top_percentile = (1 - p_value / 2) * 100
top_threshold = np.percentile(max_null_dist, top_percentile)

# Check what elements of the observed group means are significant
significant_elements = np.logical_or(p_diff < bottom_threshold, p_diff > top_threshold)

print("Number of significant elements:", np.sum(significant_elements))  # np.sum will count the number of Trues
print(significant_elements)

#%%
# We see the first element is significant with a p-value < 0.05. You will notice if you increase the p-value (by pushing the percentiles further out) the threshold for significance increase. E.g. let's see if the first parcel is still significant with a p-value < 0.001.

# Get the threshold for significance for a p-value of 0.001
bottom_threshold = np.percentile(min_null_dist, 0.05)
top_threshold = np.percentile(max_null_dist, 99.95)

# Check what elements of the observed group means are significant
significant_elements = np.logical_or(p_diff < bottom_threshold, p_diff > top_threshold)

print("Number of significant elements:", np.sum(significant_elements))  # np.sum will count the number of Trues

#%%
# We see now it's no longer significant.
