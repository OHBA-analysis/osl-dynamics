"""
Static: Power Analysis
====================== 
 
In this tutorial we will perform static power analysis on source space MEG data. This tutorial covers:
 
1. Getting the Data
2. Calculating Power from Spectra
3. Group-Level Power Analysis
4. Subject-Level Power Analysis

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/bk56q>`_ for the expected output.
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
# We will download example data hosted on `OSF <https://osf.io/by2tc/>`_. Note, `osfclient` must be installed. This can be done in jupyter notebook by running::
#
#     !pip install osfclient
#

import os

def get_data(name):
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch data/{name}.zip")
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
# We now load the data into osl-dynamics using the Data class. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details.

from osl_dynamics.data import Data

data = Data("notts_task_10_subj")
print(data)

#%%
# For static analysis we just need the time series for the parcellated data. We can access this using the `time_series` method.

ts = data.time_series()

#%%
# `ts` a list of numpy arrays. Each numpy array is a `(n_samples, n_channels)` time series for each subject.
# 
# Calculating Power from Spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Calculating power spectra
# *************************
# 
# First, we calculate the subject-specific power spectra. See the Static Power Spectra Analysis tutorial for more comprehensive description of power spectra analysis.

import numpy as np
from osl_dynamics.analysis import static

# Calculate power spectra
f, psd = static.power_spectra(
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
# Calculating power
# *****************
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
# Group-Level Power Analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Differences in power between groups
# ***********************************
# 
# We are often interested in comparing different groups of subjects. Using the `p` array we can easily calculate the group mean of subsets of the full dataset. E.g. let's say subjects \[0, 1, 3, 4, 5, 6, 8\] belong to one group and \[2, 7, 9\] belong to another group. We can calculate the group means with:

# Get the power arrays for group 1
p1 = p[[0, 1, 3, 4, 5, 6, 8]]

# Get the power arrays for group 2
p2 = p[[2, 7, 9]]

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
# We can see the main difference between the groups is in the posterior region.
# 
# Statistical Significance Testing
# ********************************
# 
# When we see differences in groups, we should perform a statistical significance test to rule out the possibility that we're observing the effect purely by chance. To compare the groups we'll use a **maximum statistic permutation test**. See the `Statistical Significance Testing tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/statistical_significance_testing.html>`_ for a detailed explanation.

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

# Real group assignments:
# - 0 indicates group 1
# - 1 indicates group 2
real_assignments = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 1])

# Generate a null distribution
null = null_distribution(p, real_assignments, n_perm=1000)

# Calculate a threshold for significance
p_value = 0.05
thres = np.percentile(null, 100 * (1 - p_value), axis=0)

# See which elements are significant
p1 = np.mean(p[real_assignments == 0], axis=0)
p2 = np.mean(p[real_assignments == 1], axis=0)
abs_diff = np.abs(p1 - p2)
sig = abs_diff > thres
print(sig)

#%%
# We can see we have some parcels with a significant difference in power between groups. Let's plot these parcels.

# Zero non-signicant parcels
p1_mean_sig = np.copy(p1_mean)
p1_mean_sig[~sig] = 0
p2_mean_sig = np.copy(p2_mean)
p2_mean_sig[~sig] = 0

# Plot (takes a few seconds to appear)
p_diff_sig = p1_mean_sig - p2_mean_sig
power.save(
    p_diff_sig,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# We see group 1 has significantly less power in the occiptal lob compared to group 2.
# 
# Subject-Level Power Analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# In this section, we look at subject-specific analysis we could do based on static power.
# 
# Clustering subjects based on power
# **********************************
# 
# When we integrate the subject-specific power spectra over a frequency band, we end up with a `(n_subjects, n_channels)` array, which contains the power at each parcel for each subject. Each `(n_channels,)` array can be thought of as a vector that characterises a subject. Let's plot the power vectors for each subject as a matrix.
# 

from osl_dynamics.utils import plotting

plotting.plot_matrices(p, titles=["Power at each ROI"])

#%%
# The y-axis is the subject and the x-axis shows each parcel/ROI. We can see there is a lot of variation between subjects and some subjects have similar power patterns. Another way to visualise power similarities between subjects is to calulate the correlation between the power vectors.

# Calculate the Pearson correlation between power vectors from each subject
corr = np.corrcoef(p)

# Zero the diagonal for visualisation
np.fill_diagonal(corr, 0)

# Plot
plotting.plot_matrices(corr)

#%%
# We can see there are some subjects that are very similar, e.g. subjects 2 and 9. A nice way to see if there's clusters of subjects is to re-order the rows/columns. There is a spectral re-ordering function in osl-dynamics for this: `analysis.connectivity.spectral_reordering <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.spectral_reordering>`_. Let's use this function to re-order the correlation matrix.

from osl_dynamics.analysis import connectivity

# Re-order to identitify clusters
corr, order = connectivity.spectral_reordering(corr)

print("New order:", order)

# Plot
plotting.plot_matrices(corr)

#%%
# Even though we have a very small dataset, we can see some structure in this matrix.
# 
# Using power to predict subject traits
# *************************************
# 
# Another use of the power at each parcel is as a feature vector for a classifier. For example, we could use the power vectors to train a classifier for predicting disease. A logistic regression is a common model used for this purpose. We can use sci-kit learn's `LogisticRegression class <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_ to do this.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Input features are the power vectors and targets are group assignments
X = p
y = real_assignments

# Split dataset into a training and testing set
# (Only one subject will be in the test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Create classifier and train
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Compare predictions to the true label on the left out data
y_pred = clf.predict(X_test)
print(y_pred, y_test)

#%%
# The above code is an illustrative example. We assigned subjects to each group randomly, so we don't expect the classifier to do very well at predicting the group. The limited dataset size almost means it's unlikely the classifier will perform well in this example.
# 
# If we wanted to predict a continuous variable, e.g. age, we would have used a `LinearRegression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_ rather than a logistic regression.
#
# Wrap Up
# ^^^^^^^
# 
# - We have shown how to calculate power from spectra and how to plot power maps.
# - We've calculated the difference in power between two groups and performance a statistical test for significance.
# - We have done some individual subject analysis.
