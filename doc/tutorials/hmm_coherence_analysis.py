"""
HMM: Coherence Analysis
=======================

In this tutorial we will analyse the dynamic networks inferred by a Hidden Markov Model (HMM) on resting-state source reconstructed MEG data. This tutorial covers:

1. Downloading a Trained Model
2. State Coherence Analysis
3. Coherence vs Power Plots

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/2jcux>`_ for the expected output.
"""

#%%
# Downloading a Trained Model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First, let's download a model that's already been trained. See the `HMM Training on Real Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/hmm_training_real_data.html>`_ for how to train an HMM.

import os

def get_trained_model(name):
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch trained_models/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {name}")
    os.remove(f"{name}.zip")
    return f"Model downloaded to: {name}"

# Download the trained model (approximately 73 MB)
model_name = "hmm_notts_rest_10_subj"
get_trained_model(model_name)

# List the contents of the downloaded directory
sub_dirs = os.listdir(model_name)
print(sub_dirs)
for sub_dir in sub_dirs:
    print(f"{sub_dir}:", os.listdir(f"{model_name}/{sub_dir}"))

#%%
# We can see the `hmm_notts_rest_10_subj` directory contains two sub-directories:
#
# - `model`: contains the trained HMM.
# - `data`: contains the inferred state probabilities (`alpha.pkl`) and state spectra (`f.npy`, `psd.npy`, `coh.npy`).
#
# State Coherence Analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Next, we turn our attention to calculating and analysing coherence networks.
#
# Load coherence spectra
# **********************
# Previously we calculate the coherence spectra as the `coh` array using the multitaper. Let's load the spectra.

import numpy as np

f = np.load("hmm_notts_rest_10_subj/data/f.npy")
coh = np.load("hmm_notts_rest_10_subj/data/coh.npy")
print(f.shape)
print(coh.shape)

#%%
# We can see from the shape of these arrays, that `f` is a 1D array that contains the frequency axis of the spectra and `coh` is a (subjects, states, channels, channels, frequencies) array.
#
# Calculate coherence networks
# ****************************
# To calculate a coherence network we need to collapse the frequency dimension. For coherences we average over frequencies. osl-dynamics has the `connectivity.mean_coherence_from_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.mean_coherence_from_spectra>`_ function to do this. This function has two mandatory arguments: the frequency axis, `f`, and coherence spectra, `coh`. Let's use `mean_coherence_from_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.mean_coherence_from_spectra>`_ to calculate coherence networks averaging over all frequencies. Let's calculate the mean coherence over all frequencies.

from osl_dynamics.analysis import connectivity

c = connectivity.mean_coherence_from_spectra(f, coh)
print(c.shape)

#%%
# Note, if we were interested in a particular frequency band we could use the `frequency_range` argument to specify the range.
#
# We can see from the shape of `c` it is a (subjects, states, channels, channels) array. The `c[0][0]` element corresponds to the `(n_channels, n_channels)` coherence network for the first subject and state.
#
# Coherence networks are often noisy so averaging over a large number of subjects (20+) is typically needed to get clean coherence networks. Let's average the coherence networks over all subjects.

# Calculate state-specific coherence networks
mean_c = np.mean(c, axis=0)
print(mean_c.shape)

#%%
# We now see `c_mean` is a (states, channels, channels) array.
#
# Plotting coherence networks
# ***************************
# Let's have a look at the coherence networks for the first couple states. The `connectivity.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_ function can be used to display a connectivity matrix (or set of connectivity matrices).

# Plot the network for first 2 states
connectivity.save(
    mean_c[:2],
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# We can see there are a lot of connections. We need a method to select the most important edges. When we were studying the power maps, we displayed the power relative to the mean across states. With the coherence networks we want to do something similar. We want to select the edges that are significantly different to the mean across states. osl-dynamics has a function called `connectivity.threshold <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.threshold>`_ that makes this easy. If we wanted the top 3% of connections that show the largest value above the mean across states, we could get this with:

thres_mean_c = connectivity.threshold(mean_c, percentile=97, subtract_mean=True)
print(thres_mean_c.shape)

#%%
# Now let's visualise the thresholded networks.

connectivity.save(
    thres_mean_c,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# Now we can see some recognisable structure in the state coherences. You'll also notice that the networks show some correspondence with the power maps, where regions of high power also show strong coherence.
#
# Note, these networks are noisy because we're only looking at 10 subjects. With more subjects, these should significantly improve.
#
# Spectral factorization (Non-Negative Matrix Fractorization, NNMF)
# *****************************************************************
# In the above code, we integrated over all frequencies to calculate the power maps and coherence networks. We expect brain activity to occur with phase-locking networks with oscillations at different frequencies. A data-driven approach for finding the frequency bands (referred to as 'spectral components') for phase-locked networks is to apply non-negative matrix factorization (NNMF) to the coherence spectra for each subject. osl-dynamics has a function that can do this: `analysis.spectral.decompose_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.decompose_spectra>`_. Let's use this function to separate our coherence spectra (`coh`) into two frequency bands.

from osl_dynamics.analysis import spectral

# Perform NNMF on the coherence spectra of each subject (we concatenate each matrix)
wb_comp = spectral.decompose_spectra(coh, n_components=2)
print(wb_comp.shape)

#%%
# `wb_comp` refers to 'wideband component' because the spectral components cover a wide frequency range (we'll see this below). If we passed `n_components=4`, we would find more narrowband components. You can interpret the `wb_comp` array as weights for how much coherent (phase-locking) activity is occuring at a particular frequency for a particular component. It can be better understood if we plot it.

from osl_dynamics.utils import plotting

plotting.plot_line(
    [f, f],  # we need to repeat twice because we fitted two components
    wb_comp,
    x_label="Frequency (Hz)",
    y_label="Spectral Component",
)

#%%
# The blue line is the first spectral component and the orange line is the second component. We can see the NNMF has separated the coherence spectra into two bands, one with a lot of coherence activity below ~22 Hz (blue line) and one above (orange line). In other words, the first spectral component contains coherent activity below 22 Hz and the second contains coherent activity mainly above 22 Hz.
#
# Now, instead of averaging the coherence spectr across all frequencies, let's just look at the first spectral component (we calculate the coherence network by weighting each frequency in the coherence spectra using the spectral component).

# Calculate the coherence network for each state by weighting with the spectral components
c = connectivity.mean_coherence_from_spectra(f, coh, wb_comp)
print(c.shape)

# Average over subjects
mean_c = np.mean(c, axis=0)
print(mean_c.shape)

# Threshold each network and look for the top 3% of connections relative to the mean
thres_mean_c = connectivity.threshold(mean_c, percentile=97, subtract_mean=True)
print(thres_mean_c.shape)

#%%
# We can see the `c_mean` and `c_mean_thres` array here is (components, states, channels, channels). We're only interested in the first spectral component. We can plot this by passing a `component=0` argument to `connectivity.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_.

connectivity.save(
    thres_mean_c,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    component=0,
)

#%%
# Data-driven thresholding: Gaussian Mixture Model (GMM)
# ******************************************************
# In the above plots we arbitrarily selected a percentile to plot. However, different coherence networks would have a differing number of interesting connections. One approach to improve the thresholding in a data driven way is to use a two-component GMM to select edges. This was described in more detail in the `Static AEC Analysis tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/static_aec_analysis.html>`_.
#
# Let's look at the coherence maps we get using a GMM to threshold. We will look at the first spectral component.

# Threshold each network using a HMM
thres_mean_c = connectivity.gmm_threshold(
    mean_c,
    p_value=0.01,
    keep_positive_only=True,
    subtract_mean=True,
    show=True,
)
print(thres_mean_c.shape)

# Plot
connectivity.save(
    thres_mean_c,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    component=0,
)

#%%
# We see the GMM thresholding worked well for some of the states but not others. The GMM is particularly sensitive to the distribution of connections, if we have noisy coherence networks this might not work very well. Generally specifying the threshold using a percentile is more robust.
#
# Coherence vs Power Plots
# ^^^^^^^^^^^^^^^^^^^^^^^^
# We may be interested in how the coherence relates to power for each state. We can do this by producing scatter plots of coherence vs power. Let's see how to do this. First we load the power spectra previously calculated.

psd = np.load("hmm_notts_rest_10_subj/data/psd.npy")
print(psd.shape)

#%%
# We can see this is a (subjects, states, channels, frequencies) array. Let's integrate the power spectra over the each spectral component.

from osl_dynamics.analysis import power

# Calculate power
p = power.variance_from_spectra(f, psd, wb_comp)
print(p.shape)

#%%
# Now we want to summarise the power and coherence for each state. We could do this by:
#
# - Averaging over subjects to get the power/coherence for each parcel.
# - Averaging over parcels to get the power/coherence for each subject.
#
# Let's first average the power and coherence over subjects and select the first spectral component.

# Average power over subjects
mean_p = np.mean(p, axis=0)
print(mean_p.shape)

# Average coherence over subjects
mean_c = np.mean(c, axis=0)
print(mean_c.shape)

# Keep the first spectral component
mean_p = mean_p[0]
mean_c = mean_c[0]
print(mean_p.shape)
print(mean_c.shape)

#%%
# We have a `(n_channels,)` array for each state for the power, but a `(n_channels, n_channels)` array for each state for the coherence. To reduce the coherence down to a 1D array for each state, we can calculate the average pairwise coherence for each channels. We can do this with the `connectivity.mean_connections <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.mean_connections>`_ function.

mean_c = connectivity.mean_connections(mean_c)
print(mean_c.shape)

#%%
# Now we can make a scatter plot of coherence vs power for each state.

plotting.plot_scatter(
    mean_p,
    mean_c,
    labels=[f"State {i}" for i in range(1, mean_p.shape[0] + 1)],
    x_label="Power (a.u.)",
    y_label="Coherence",
)

#%%
# We can see the different states have different power/coherence properties. E.g. some states typically have more power/coherence at all parcels. If we average over subjects we get the following.

# Average power over parcels
mean_p = np.mean(p, axis=-1)
print(mean_p.shape)

# Average coherence over subjects
mean_c = np.mean(c, axis=(-2, -1))
print(mean_c.shape)

# Keep the first spectral component
mean_p = mean_p[:, 0]
mean_c = mean_c[:, 0]
print(mean_p.shape)
print(mean_c.shape)

# Plot
plotting.plot_scatter(
    mean_p,
    mean_c,
    x_label="Power (a.u.)",
    y_label="Coherence",
)

#%%
# Each scatter point is a subject here. We also see different subjects have different power/coherence characteristics.
#
# Wrap Up
# ^^^^^^^
# - We have shown have to calculate coherence networks from spectra.
# - We have shown how plot coherence networks and explored various methods for thresholding.
