"""
HMM: Resting-State Network Analysis (Part 2)
============================================

In this tutorial we will perform dynamic network analysis on source space MEG data using a Hidden Markov Model (HMM). We will focus on resting-state data. This tutorial covers:

1. Calculating Summary Statistics
2. Calculating Spectrally Resolved Networks

The input to this script is:

- The inferred state probabilities.
- The training data (unprepared).

The output of this script is:

- The calculation of state summary statistics that characterise dynamics of individual subjects.
- Plots of the spectral resolved networks inferred from the data: power spectrum, power map and coherence network

Note, this webpage does not contain the output of each cell. We advise downloading the notebook and working through it locally on your machine. The expected output of this script can be found `here <https://osf.io/4adyz>`_.
"""

#%%
# Calculating Summary Statistics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Now we have trained an HMM, we want to use to to help us understand the training data. A useful way of characterising the data is to use statistics that summarise the state time course, these are referred to as 'summary statistics'. The state time course captures the dynamics in the training data, therefore, summary statistics characterise the dynamics of the data.
# 
# We have the state time course for each subject, therefore, we can calculate summary statistics for individual subjects. These gives us a summary measure of dynamics for each subject. Looking at differences in summary statistics between subjects is a way of understanding how dynamics might be different for different subjects. It is often common to average summary statistics for groups of subjects and comparing the two groups.
# 
# In this section, we'll look at a couple popular summary statistics: the fractional occupancy, which is the fraction of total time spent in a particular state and the mean lifetime, which is the average duration a state is activate.
# 
# Load the inferred state probabilities
# *************************************
# 
# Before calculating summary statistics, let's load the inferred state probabilities.

import pickle

alpha = pickle.load(open("trained_model/alpha.pkl", "rb"))

#%%
# Let's also plot the state probabilities for the first few seconds of the first subject to get a feel for what they look like. We can use the `utils.plotting.plot_alpha <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/utils/plotting/index.html#osl_dynamics.utils.plotting.plot_alpha>`_ function to do this.

from osl_dynamics.utils import plotting

# Plot the state probability time course for the first subject (8 seconds)
plotting.plot_alpha(alpha[0], n_samples=2000)

#%%
# When looking at the state probability time course you want to see a good number of transitions between states.
# 
# The `alpha` list contains the state probabilities time course for each subject. To calculate summary statistics we first need to hard classify the state probabilities to give a state time course. We can use the `inference.modes.argmax_time_courses <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/inference/modes/index.html#osl_dynamics.inference.modes.argmax_time_courses>`_ function to do this.

from osl_dynamics.inference.modes import argmax_time_courses

# Hard classify the state probabilities
stc = argmax_time_courses(alpha)

# Plot the state time course for the first subject (8 seconds)
plotting.plot_alpha(stc[0], n_samples=2000)

#%%
# We can see this time series is completely binarised, only one state is active at each time point.
# 
# Fractional occupancy
# ********************
# 
# The `analysis.modes <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/modes/index.html>`_ module in osl-dynamics contains helpful functions for calculating summary statistics. Let's use the `analysis.modes.fractional_occupancies <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/modes/index.html#osl_dynamics.analysis.modes.fractional_occupancies>`_ function to calculate the fractional occupancy of each state for each subject.

from osl_dynamics.analysis import modes

# Calculate fractional occupancies
fo = modes.fractional_occupancies(stc)
print(fo.shape)

#%%
# We can see `fo` is a (subjects, states) numpy array which contains the fractional occupancy of each state for each subject.
# 
# To get a feel for how much each state is activated we can print the group average:

import numpy as np

print(np.mean(fo, axis=0))

#%%
# We can see each state shows a significant activation (i.e. non-zero), which gives us confidence in the fit. If only one state is activated, that indicates further preprocessing is likely needed.
# 
# We can examine the distribution across subjects for each state by creating a violin plot. The `utils.plotting.plot_violin <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/utils/plotting/index.html#osl_dynamics.utils.plotting.plot_violin>`_ function can be used to do this.

from osl_dynamics.utils import plotting

# Plot the distribution of fractional occupancy (FO) across subjects
plotting.plot_violin(fo, x_label="State", y_label="FO")

#%%
# We can see there is a lot of variation across subjects.
# 
# Mean Lifetime
# *************
# 
# Next, let's look at another popular summary statistic, i.e. the mean lifetime. We can calculate this using the `analysis.modes.mean_lifetimes <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/modes/index.html#osl_dynamics.analysis.modes.mean_lifetimes>`_ function.

# Calculate mean lifetimes (in seconds)
mlt = modes.mean_lifetimes(stc, sampling_frequency=250)

# Convert to ms
mlt *= 1000

# Print the group average
print(np.mean(mlt, axis=0))

# Plot distribution across subjects
plotting.plot_violin(mlt, x_label="State", y_label="Mean Lifetime (ms)")

#%%
# Calculating Spectrally Resolved Networks
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Each state inferred by the HMM corresponds to a spectrally resolved network. To visualise these networks we re-estimate them from the training data using the inferred state probabilities.
# 
# Calculating power spectra and coherences
# ****************************************
# 
# The first thing we want to do is calculate the spectral properties of each state (i.e. power spectrum and coherence). This is done by using standard calculation methods (in our case the multitaper for spectrum estimation) to the time points identified as belonging to a particular state. The `analysis.spectra.multitaper_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.multitaper_spectra>`_ function does this for us. Let's first run this function, then we'll discuss its output. The arguments we need to pass to this function are:
#
# - `data`. This is the source reconstructed data aligned to the state time course.
# - `alpha`. This is the state time course or probabilities (either can be used). Here we'll use the state probabilities.
# - `sampling_frequency` in Hz.
# - `time_half_bandwidth`. This is a parameter for the multitaper, we suggest using `4`.
# - `n_tapers`. This is another parameter for the multitaper, we suggest using `7`.
# - `frequency_range`. This is the frequency range we're interested in.

from osl_dynamics.analysis import spectral

# Get the source reconstructed data aligned to the state probabilities
data = pickle.load(open("trained_model/data.pkl", "rb"))

# Calculate multitaper spectra for each state and subject (will take a few minutes)
f, psd, coh = spectral.multitaper_spectra(
    data=data,
    alpha=alpha,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
)

#%%
# Note, there is a `n_jobs` argument that can be used to calculate the multitaper spectrum for each subject in parallel.
# 
# Calculating the spectrum can be time consuming so it is useful to save it as a numpy file, which can be loaded very quickly.

np.save("trained_model/f.npy", f)
np.save("trained_model/psd.npy", psd)
np.save("trained_model/coh.npy", coh)

#%%
# To understand the `f`, `psd` and `coh` numpy arrays it is useful to print their shape.

f = np.load("trained_model/f.npy")
psd = np.load("trained_model/psd.npy")
coh = np.load("trained_model/coh.npy")

print(f.shape)
print(psd.shape)
print(coh.shape)

#%%
# We can see the `f` array is 1D, it corresponds to the frequency axis. The `psd` array is (subjects, states, channels, frequencies) and the `coh` array is (subjects, states, channels, channels, frequencies).
# 
# Plotting the power spectra
# **************************
# 
# We can plot the power spectra to see what oscillations typically occur when a particular state is on. To help us visualise this we can average over the subjects and channels. This gives us a (states, frequencies) array.

# Average over subjects and channels
psd_mean = np.mean(psd, axis=(0,2))
print(psd_mean.shape)

# Plot
n_states = psd_mean.shape[0]
plotting.plot_line(
    [f] * n_states,
    psd_mean,
    labels=[f"State {i}" for i in range(1, n_states + 1)],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    x_range=[f[0], f[-1]],
)

#%%
# We can see there are significant differences in the spectra of the states, i.e. the different oscillations are present when each state is active.
#
# Calculating power maps
# **********************
# 
# The `psd` array contains the spectrum at each parcel (for each subject/state). This is a function of frequency. To calculate power we need to integrate over a frequency range. osl-dynamics has the `analysis.power.variance_from_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.variance_from_spectra>`_ function to help us do this. For example, if we want the power over the alpha band (8-12 Hz), we could do:

from osl_dynamics.analysis import power

# Integrate the power spectra over the alpha band (8-12 Hz)
p = power.variance_from_spectra(f, psd, frequency_range=[8, 12])
print(p.shape)

#%%
# We can see the `p` array is now (subjects, states, channels), it contains the 'power map' for each subject and state. To access the power map for the first subject and state, we could use `p[0][0]`, which would be a `(42,)` shape array. We can plot power maps using the `analysis.power.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.save>`_ function. 

power.save(
    p[0][0],
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# We see a lot of posterior activity, which is expected.
# 
# Note, the choice for the `mask_file` and `parcellation_file` arguments is determined by how we source reconstructed the data. Also note the `power.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.save>`_ function has a `filename` arugment that can used to save the image to a file.
# 
# We've seen the HMM has segmented the training data into states with different oscillotory activity (we saw this when we plotted the state power spectra). Therefore, we don't necessarily need to specify a frequency range ourselves. We could just look at the power across all frequencies for a given state. Let's integrate the power spectra for each state across all frequencies.

p = power.variance_from_spectra(f, psd)
print(p.shape)

#%%
# And this time rather than just plotting the power map for a single subject, let's average over all subjects.

p_mean = np.mean(p, axis=0)
print(p_mean.shape)

#%%
# Now we have a (states, channels) array. Let's see what the power map for each state looks like. 

# Takes a few seconds for the power maps to appear
power.save(
    p_mean,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# Because the power is always positive and a lot of the state have power in similar regions all the power maps look similar. To highlight differences we can display the power maps relative to someting. Typically we use the average across states - taking the average across states approximates the static power of the entire training dataset. We can do this by passing the `subtract_mean` argument.

# Takes a few seconds for the power maps to appear
power.save(
    p_mean,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    subtract_mean=True,
)

#%%
# We can now see recognisable functional networks, e.g. a visual, a motor, a frontal network.
# 
# One final detail is when we subtract a mean, by default we weight each state equally. However, we saw when we calculated the fractional occupancy of each state that they can be very different. We can include this when we estimate the mean across states. `power.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.save>`_ has a `mean_weights` argument that can be used for this. Let's use the fractional occupancies to weight each state before calculating the mean.

# Takes a few seconds for the power maps to appear
power.save(
    p_mean,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    subtract_mean=True,
    mean_weights=np.mean(fo, axis=0),  # np.mean calculates the mean FO over subjects here
)

#%%
# Using the fractional occupancy to calculate the reference can improve the power maps if there is a large imbalance between the fractional occupancies. However, usually the effect is small.
# 
# Calculating coherence networks
# ******************************
# 
# Next, we turn our attention to calculating coherence networks. Previously we calculate the coherence spectra as the `coh` array using the multitaper. Again we need to collapse the frequency axis. For the coherence networks we take the mean over a frequency range. osl-dynamics has the `analysis.connectivity.mean_coherence_from_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.mean_coherence_from_spectra>`_ function to do this. Let's do this and calculate the mean across all frequencies.

from osl_dynamics.analysis import connectivity

c = connectivity.mean_coherence_from_spectra(f, coh)
print(c.shape)

#%%
# We can see from the shape `c` is a (subjects, states, channels, channels) array. The `c[0][0]` element corresponds to the `(n_channels, n_channels)` coherence network for the first subject and state. Let's average over all subjects. Coherence networks are often noisy so averaging over a large number of subjects (20+) is typically needed to get clean coherence networks.

# Calculate state specific coherence networks
c_mean = np.mean(c, axis=0)
print(c_mean.shape)

#%%
# Let's have a look at the first couple coherence networks. The `connectivity.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_ function can be used to display a connectivity matrix (or set of connectivity matrices).

# Plot the network for first 2 states
connectivity.save(
    c_mean[:2],
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# We can see there are a lot of connections and we need a method to select the most imporant edges. When we were studying the power maps, we displayed the power relative to the mean across states. With the coherence networks we want to do something similar. We want to select the edges that are significantly different to the mean across states. osl-dynamics has a function called `connectivity.threshold <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.threshold>`_ that makes this easy. If we wanted the top 3% of connections that show the largest value above the mean across states, we could get this with:

c_mean_thres = connectivity.threshold(c_mean, percentile=97, subtract_mean=True)
print(c_mean_thres.shape)

#%%
# Now let's visualise the thresholded networks.

connectivity.save(
    c_mean_thres,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# Note, these networks are very noisy because we're only looking at 10 subjects. However, we can see some correspondence with the power maps, where the high power regions in each state show some connectivity in the same regions. With more subjects, these should significantly improve.
#
# Spectral factorization (Non-Negative Matrix Fractorization, NNMF)
# *****************************************************************
# 
# In the above code, we integrated over all frequencies to calculate the power maps and coherence networks. We expect brain activity to occur with phase-locking networks with oscillations at different frequencies. A data-driven approach for finding the frequency bands (referred to as 'spectral components') for phase-locked networks is to apply non-negative matrix factorization (NNMF) to the coherence spectra for each subject. osl-dynamics has a function that can do this: `analysis.spectral.decompose_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.decompose_spectra>`_. Let's use this function to separate our coherence spectra (`coh`) into two frequency bands.

# Perform NNMF on the coherence spectra of each subject (we concatenate each matrix)
wb_comp = spectral.decompose_spectra(coh, n_components=2)
print(wb_comp.shape)

#%%
# `wb_comp` refers to 'wideband component' because the spectral components cover a wide frequency range (we'll see this below). If we passed `n_components=4`, we would find more narrowband components. You can interpret the `wb_comp` array as weights for how much coherent (phase-locking) activity is occuring at a particular frequency for a particular component. It can be better understood if we plot it.

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

# First average the coherence spectra across subjects
coh_mean = np.mean(coh, axis=0)

# Calculate the coherence network for each state by weighting with the spectral components
c_mean = connectivity.mean_coherence_from_spectra(f, coh_mean, wb_comp)
print(c_mean.shape)

# Threshold each network and look for the top 3% of connections relative to the mean
c_mean_thres = connectivity.threshold(c_mean, percentile=97, subtract_mean=True)
print(c_mean_thres.shape)

#%%
# We can see the `c_mean` and `c_mean_thres` array here is (components, states, channels, channels). We're only interested in the first spectral component. We can plot this by passing a `component=0` argument to `connectivity.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_.

connectivity.save(
    c_mean_thres,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    component=0,
)

#%%
# Gaussian mixture model (GMM) thresholding
# *****************************************
# 
# In the above plots we arbitrarily selected a percentile to plot. However, different coherence networks would have a differing number of interesting connections. One approach to improve the thresholding in a data driven way is to use a two-component GMM to select edges. This was described in detail in the Static Network Analysis tutorial.
# 
# Let's look at the coherence maps we get using a GMM to threshold. We will look at the first spectral component.

# Threshold each network using a HMM
c_mean_thres = connectivity.gmm_threshold(
    c_mean,
    p_value=0.01,
    keep_positive_only=True,
    subtract_mean=True,
    show=True,
)
print(c_mean_thres.shape)

# Plot
connectivity.save(
    c_mean_thres,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    component=0,
)

#%%
# Wrap Up
# ^^^^^^^
# 
# - We have shown how to calculate some common summary statistics based on the inferred state time course.
# - We have shown have to calculate state spectra using a multitaper.
# - We have shown how to plot state PSDs, power maps and coherence networks and various methods for thresholding.
