"""
Sliding Window Analysis
=======================

In this tutorial we will use a sliding window technique, which is a common approach for studying network dynamics, to analyse source space MEG data. fMRI data can easily be substituted.

This tutorial covers:

1. Getting the Data
2. Estimating Networks using a Sliding Window
3. Clustering the Networks
"""

#%%
# Getting the Data
# ^^^^^^^^^^^^^^^^
# We will use resting-state MEG data that has already been source reconstructed. This dataset is:
#
# - Parcellated to 38 regions of interest (ROI). The parcellation file used was `fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz`.
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

# Download the dataset (approximately 70 MB)
get_data("notts_mrc_meguk_giles_5_subjects", rename="source_data")

#%%
# Load the data
# *************
# We now load the data into osl-dynamics using the Data class. See the `Loading Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/data_loading.html>`_ for further details.

from osl_dynamics.data import Data

data = Data("source_data", n_jobs=4)

# Display some summary information
print(data)

#%%
# For the sliding window analysis we just need the time series for the parcellated data. We can access this using the `time_series` method.

ts = data.time_series()

#%%
# Estimating Networks using a Sliding Window
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A sliding window network analysis involves:
#
# - Taking a segment of your time series, i.e. a window. We need to pre-specify the length of the window.
# - Estimating a network using the window.
# - Sliding the window along the time series.
# - Estimating a network using the new window.
# - Repeat.
#
# This results in a series of networks. Important aspects of this technique are:
#
# - A useful advantage of this technique is you can use any metric you like for the pairwise connectivity as long as you can calculate it from the data in a window.
# - The main limitation of this approach is you have to pre-specify the window length. If you don't know the time scale of dynamics in the time series, specifying the window length is difficult.
# - There is a trade-off between having long windows which will lead to an accurate estimate of a network but long temporal specificity, whereas a short window has good temporal specificity but can give noisy estimates of network connections.
#
# Sliding window connectivity using the Pearson correlation
# *********************************************************
# Now that we have loaded the data we want to study, let's estimate sliding window networks. The first thing we need to do is choose the the metric for connectivity we will use. In this tutorial we're use the absolute value of the Pearson correlation of the source space time series. osl-dynamics has a function for calculating sliding window networks: `analysis.connectivity.sliding_window_connectivity <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.sliding_window_connectivity>`_. Let's use this function to calculate dynamics networks.

from osl_dynamics.analysis import connectivity

# Calculate the sliding window connectivity
swc = connectivity.sliding_window_connectivity(ts, window_length=100, step_size=50, conn_type="corr")

#%%
# `swc` is a list of numpy arrays. Each subject has its own numpy array which is a series of connectivity matrices. Let's concatenate these to give one time series and take the absolute value.

import numpy as np

swc_concat = np.concatenate(swc)
swc_concat = np.abs(swc_concat)

print(swc_concat.shape)

#%%
# We can see there are a total of 14,520 windows we calculated networks for. Let's plot the first few networks. We can use the `connectivity.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_ function to do this.

connectivity.save(
    swc_concat[:5],
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    threshold=0.95,  # only display the top 5% of connections
)

#%%
# Clearly, these networks are very noisy. The large number of networks also makes the sliding window networks difficult to intepret. A common next step is to cluster the sliding window networks to give a discrete set of networks. We'll do this next.
#
# Clustering the Networks
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# K-means clustering
# ******************
# The most common approach used for clustering is the K-means algorithm. We will use sci-kit learn's K-means class to cluster the sliding window connectivities. The documentation for sci-kit learn's K-means class is `here <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.
#
# First, let's initiate a K-means object. When we do this we need to specify the number of clusters. We will use 6.

from sklearn.cluster import KMeans

# Initiate the K-means class
kmeans = KMeans(n_clusters=6, n_init="auto", verbose=0)

#%%
# Next, we need to prepare the input for the K-means algorithm. It requires a set of vectors, which is stored as a 2D numpy array. To convert our connectivity matrices to vectors, we will simply take the upper right triangle. We can do this because we know correlation matrices are symmetric.

# Get indices that correspond to an upper triangle of a matrix
# (not including the diagonal)
n_channels = swc_concat.shape[-1]
i, j = np.triu_indices(n_channels, k=1)

# Now let's convert the sliding window connectivity matrices to a series of vectors
swc_vectors = swc_concat[:, i, j]

#%%
# We can check this worked by printing the shape of the `swc_vectors` array.

print(swc_vectors.shape)

#%%
# We expect the vectors to have 42 * 41 // 2 = 861 elements (upper triangle of a square matrix), which matches the second dimension.
#
# Now we can train the Kmeans algorithm using the `fit` method.

kmeans.fit(swc_vectors)

#%%
# We can access the cluster centroids using the `cluster_centers_` attribute. Let's see what this gives and print the shape to double check it's what we expected.

centroids = kmeans.cluster_centers_
print(centroids.shape)

#%%
# We see the first dimension is the number of clusters and the second dimension is the length of the input vectors, which is what we expect.
#
# Now, we just need to put the vectors back into ROIs by ROIs form and then we can visualise them as networks.

n_clusters = centroids.shape[0]
n_channels = data.n_channels

# Convert from a vector to a connectivity matrix
kmean_networks = np.empty([n_clusters, n_channels, n_channels])
kmean_networks[:, i, j] = centroids
kmean_networks[:, j, i] = centroids

#%%
# Finally, let's display the Kmeans networks using glass brain plots.

connectivity.save(
    kmean_networks,
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    threshold=0.95,  # only display the top 5% of connections
)

#%%
# We can see clustering the networks significantly improves them.
