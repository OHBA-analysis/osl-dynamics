"""
Static: Amplitude Envelope Correlation (AEC) Analysis
=====================================================

In this tutorial we will perform static AEC analysis on source space MEG data. This tutorial covers:

1. Getting the Data
2. Calculating AEC Networks
3. Network Analysis

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/k9qzw>`_ for the expected output.
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

# Download the dataset (approximately 150 MB)
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
# Calculating AEC Networks
# ^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Next, we will estimate static networks for each subject. For this we need to define a metric for connectivity between ROIs. There are a lot of options for this. In this tutorial we'll look at the **amplitude envelope correlation (AEC)**.
# 
# Calculate AEC
# *************
# 
# AEC can be calculated from the parcellated time series directly. First, we need to prepare the parcellated data. Previously we loaded the data using the `Data class <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/data/base/index.html#osl_dynamics.data.base.Data>`_. Fortunately, the Data class has a `prepare` method that makes this easy. Let's prepare the data for calculate the AEC network for activity in the alpha band (8-12 Hz).

# Before we can prepare the data we must specify the sampling frequency
# (this is needed to bandpass filter the data)
data.set_sampling_frequency(250)

# Calculate amplitude envelope data for the alpha band (8-12 Hz)
data.prepare(low_freq=8, high_freq=12, amplitude_envelope=True)

# Get the amplitude envelope time series for each subject (ts is a list of numpy arrays)
ts = data.time_series()

#%%
# Note, other common frequency bands are:
# 
# - Delta: 1-4 Hz.
# - Theta: 4-8 Hz.
# - Beta: 12-30 Hz.
# - Gamma: 30+ Hz.
# 
# Next, we want to calculate the correlation between amplitude envelopes. osl-dynamics has the `analysis.static.functional_connectivity <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/static/index.html#osl_dynamics.analysis.static.functional_connectivity>`_ function for this.

from osl_dynamics.analysis import static

# Calculate the correlation between amplitude envelope time series
aec = static.functional_connectivity(ts)

#%%
# We can understand the `aec` array by printing its shape.

print(aec.shape)

#%%
# We can see it is a subject by ROIs by ROIs array. It contains all pairwise connections between ROIs.
# 
# Network Analysis
# ^^^^^^^^^^^^^^^^
# 
# Now that we have the AEC network for each subject, let's start by visualising them.
# 
# Visualising networks
# ********************
# 
# A common approach for plotting a network is as a matrix. We can do this with the `plotting.plot_matrices <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/utils/plotting/index.html#osl_dynamics.utils.plotting.plot_matrices>`_ function in osl-dynamics.

from osl_dynamics.utils import plotting

plotting.plot_matrices(aec, titles=[f"Subject {i+1}" for i in range(len(aec))])

#%%
# The diagonal is full of ones and is a lot larger then the off-diagonal values. This means our colour scale doesn't show the off-diagonal structure very well. We can zero the diagonal to improve this.

import numpy as np

mat = np.copy(aec)  # we don't want to change the original aec array
for m in mat:
    np.fill_diagonal(m, 0)

plotting.plot_matrices(mat, titles=[f"Subject {i+1}" for i in range(len(aec))])

#%%
# We can now see the off-diagonal structure a bit better. We also see there is a lot of variability between subjects.
# 
# Another way we can visualise the network is a glass brain plot. We can do this using the `connectivity.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_ function in osl-dynamics. This function is a wrapper for the nilearn function `plot_connectome <https://nilearn.github.io/stable/modules/generated/nilearn.plotting.plot_connectome.html>`_. Let's use `connectivity.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_ to plot the first subject's AEC network.

from osl_dynamics.analysis import connectivity

connectivity.save(
    aec[0],
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# If we wanted to save the plot to an image file we could pass the `filename` argument. If we wanted to pass any arguments to nilearn's `plot_connectome <https://nilearn.github.io/stable/modules/generated/nilearn.plotting.plot_connectome.html>`_ function, we could use the `plot_kwargs` arguement. Let's pass some extra arguments to `plot_connectome <https://nilearn.github.io/stable/modules/generated/nilearn.plotting.plot_connectome.html>`_ to adjust the color bar and color map.

connectivity.save(
    aec[0],
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    plot_kwargs={"edge_vmin": 0, "edge_vmax": 0.4, "edge_cmap": "Reds"},
)

#%%
# In the above plot we see every pairwise connection. Often, we're just interested in the strongest connections - this helps us to avoid interpreting connections that are simply due to noise. Next, let's see how we threshold the networks.
# 
# Thresholding networks by specifying a percentile
# ************************************************
# 
# We can use the `connectivity.threshold <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.threshold>`_ function in osl-dynamics to select the strongest connections. The easiest way to threshold is to pass the `percentile` argument, let's select the top 5% of connections.

thres_aec = connectivity.threshold(aec, percentile=95)

#%%
# Note, `connectivity.threshold <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.threshold>`_ acts on the connectivity matrix from each subject separately.
# 
# Subject-specific networks
# *************************
# 
# Next, let's plot the AEC network for the first 3 subjects, thresholding the top 5%.

# Keep the top 5% of connections
thres_aec = connectivity.threshold(aec, percentile=95)

# Plot
connectivity.save(
    thres_aec[:3],
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    plot_kwargs={"edge_vmin": 0, "edge_vmax": 0.5, "edge_cmap": "Reds"},
)

#%%
# We see there is some structure in the networks. We also observed there is significant subject-to-subject variation.
# 
# Group averaged networks
# ***********************
# 
# Estimating subject-specific connectivity networks is often very noisy. Cleaner networks come out when we average over groups as this removes noise. Let's plot the group average AEC network.

# Average over the group
group_aec = np.mean(aec, axis=0)

# Keep the top 5% of connections
thres_group_aec = connectivity.threshold(group_aec, percentile=95)

# Plot
connectivity.save(
    thres_group_aec,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    plot_kwargs={"edge_vmin": 0, "edge_vmax": 0.5, "edge_cmap": "Reds"},
)

#%%
# Note, we can also plot an AEC network as a 3D glass brain plot using the `glassbrain` argument.

# Display the network
connectivity.save(
    thres_group_aec,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    glassbrain=True,
)

#%%
# When we pass `glassbrain=True`, `connectivity.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_ becomes a wrapper for the nilearn function `view_connectome <https://nilearn.github.io/stable/modules/generated/nilearn.plotting.view_connectome.html>`_. We can use the `plot_kwargs` argument to pass extra arguments to nilearn's function.
# 
# We can see from the above plot averaging over a large group gives us a cleaner network.  This is simply due to the data being noisy which makes estimating networks hard. Averaging over subjects helps remove this noise. In the group average network we can see the strongest connections are in posterior regions as expected.
# 
# Data-driven thresholding
# ************************
# 
# Another option is rather than specifying a percentile by hand, we can use a Gaussian Mixture Model (GMM) fit with two components (an 'on' and an 'off' component) to determine a threshold for selecting connections. The way this works is we fit two Gaussians to the distribution of connections. To understand this, let's first examine the distribution of connections.

import matplotlib.pyplot as plt

def plot_dist(values):
    """Plots a histogram."""
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(values.flatten(), bins=50, histtype="step")
    ax.set_xlabel("AEC")
    ax.set_ylabel("Number of edges")

# Plot distribution of connections
plot_dist(group_aec)

#%%
# We see there is a cluster of connections between AEC=0 and 0.4 and another at AEC=1. The AEC=1 connections are on the diagonal of the connectivity matrix. Let's remove these to examine the distribution of off-diagonal elements, which is what we're interested in.

# Fill diagonal with nan values
# (nan is prefered to zeros because a zeo value will be included in the distribution, nans won't)
np.fill_diagonal(group_aec, np.nan)

# Note, np.fill_diagonal alters the group_aec array in place,
# i.e. we don't need to do aec_mean = np.fill_diagonal(aec_mean, np.nan)

# Plot distribution of connections
plot_dist(group_aec)

#%%
# We can see there is a peak around AEC=0.05 and a long tail for higher values. We want the connections around the AEC=0.05 peak to be captured by a Gaussian and the long tail to be captured by another Gaussian. Let's fit a two component Gaussian to this distribution. Fortunately, osl-dynamics has a function to do this for us: `analysis.connectivity.fit_gmm <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.fit_gmm>`_. This function returns the threshold (as a percentile) that determines the Gaussian component a connection belows to.

# Fit a two-component Gaussian mixture model to the connectivity matrix
#
# We pass the standardize=False argument because we don't want to alter the
# distribution before fitting the GMM.
percentile = connectivity.fit_gmm(group_aec, show=True)
print("Percentile:", percentile)

#%%
# Let's now use the data-driven threshold to select connections in our network.

# Threshold
thres_group_aec = connectivity.threshold(group_aec, percentile=percentile)

# Display the network
connectivity.save(
    thres_group_aec,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    glassbrain=True,
    plot_kwargs={"edge_cmap": "Reds", "symmetric_cmap": False},
)

#%%
# We can a lot more connections now. We can be more extreme with the connections we choose by enforcing the likelihood a of a connection belonging to the 'off' component is below a certain p-value. For example, if we wanted to show the connections belonging to the 'on' GMM component, that had a likelihood of less than 0.01 of belonging to the 'off' component, we could do the following:

# Fit a two-component Gaussian mixture model to the connectivity matrix
# ensuring the threshold is beyond a p-value of 0.01 of belonging to the off component
percentile = connectivity.fit_gmm(group_aec, p_value=0.01, show=True)
print("Percentile:", percentile)

#%%
# We can see the threshold has moved much more to the right now. Let's example the network with this threshold.

# Threshold
thres_group_aec = connectivity.threshold(group_aec, percentile=percentile)

# Display the network
connectivity.save(
    thres_group_aec,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    glassbrain=True,
    plot_kwargs={"edge_cmap": "Reds", "symmetric_cmap": False},
)

#%%
# Note, osl-dynamics has a wrapper function to return the thresholded network directly (so you don't need to threshold yourself): `connectivity.gmm_threshold <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.gmm_threshold>`_. Using this function, we can threshold connectivity matrix in one line::
#
#     thres_aec_mean = connectivity.gmm_threshold(aec_mean, p_value=0.01)
# 
# Comparing groups
# ****************
# 
# We could also average over sub-groups. For example, these sub-groups can be healthy vs diseased participants. Let's use the same grouping as the `Static Power Analysis tutorial <https://osf.io/mkd2e>`_.

# Group assignments:
# - 0 is group 1
# - 1 is group 2
assignments = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 1])

# Get AEC networks for each group
aec1 = aec[assignments == 0]
aec2 = aec[assignments == 1]
print(aec1.shape)
print(aec2.shape)

# Average the sub-groups
group1_aec = np.mean(aec1, axis=0)
group2_aec = np.mean(aec2, axis=0)

# Threshold top 5%
thres_group1_aec = connectivity.threshold(group1_aec, percentile=95)
thres_group2_aec = connectivity.threshold(group2_aec, percentile=95)

# Plot
conn_maps = np.array([thres_group1_aec, thres_group2_aec])
connectivity.save(
    conn_maps,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    plot_kwargs={"edge_vmin": 0, "edge_vmax": conn_maps.max(), "edge_cmap": "Reds"},
)

#%%
# We can see group 2 has much stronger AEC connections in the occipital lobe.
# 
# Statistical Significance Testing
# ********************************
# 
# Let's see if the difference in AEC is significant. We'll use a **maximum statistic permutation test**. See the `Statistical Significance Testing tutorial <https://osf.io/ft3rm>`_ for more details.
# 
# First, we need to turn the subject-specific connectivity matrices into subject-specific vectors. To do this we simple take the upper triangle.

n_channels = aec.shape[-1]

# Indices for the upper triangle (not including the diagonal)
i, j = np.triu_indices(n_channels, k=1)

# Create subject-specific vectors containing the pairwise AEC
vectors = aec[:, i, j]
print(vectors.shape)

#%%
# Now, let's perform the permutations test.

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

print("Significant channels:")
print(np.argwhere(sig))

#%%
# We can see there's only a couple connections that are significant. We're not expecting to see a lot of significant connections because we have a very small dataset.
# 
# Let's plot the significant connections.

# Zero the non-significant pairwise connections for each group
vectors1_sig = np.copy(vectors1)
vectors1_sig[~sig] = 0

vectors2_sig = np.copy(vectors2)
vectors2_sig[~sig] = 0

# Create a full matrix from the vector
group1_aec_sig = np.zeros([n_channels, n_channels])
group1_aec_sig[i, j] = vectors1_sig
group1_aec_sig[j, i] = vectors1_sig

group2_aec_sig = np.zeros([n_channels, n_channels])
group2_aec_sig[i, j] = vectors2_sig
group2_aec_sig[j, i] = vectors2_sig

# Plot
conn_maps = np.array([group1_aec_sig, group2_aec_sig])
connectivity.save(
    conn_maps,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    plot_kwargs={"edge_vmin": 0, "edge_vmax": conn_maps.max(), "edge_cmap": "Reds"},
)

#%%
# We can see there's a couple significant connections inbetween the right temporal and occiptal lobe.
# 
# Wrap Up
# ^^^^^^^
# 
# - We have shown how to calculate AEC networks and visual them.
# - We performed a signifance test to compare two groups of subjects.
