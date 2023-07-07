"""
HMM: Power Analysis
===================

In this tutorial we will analyse the dynamic networks inferred by a Hidden Markov Model (HMM) on resting-state source reconstructed MEG data. This tutorial covers:

1. Downloading a Trained Model
2. State Power Spectra
3. State Power Maps

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/tyjac>`_ for the expected output.
"""

#%%
# Download a Trained Model
# ^^^^^^^^^^^^^^^^^^^^^^^^
# First, let's download a model that's already been trained on a resting-state dataset. See the `HMM Training on Real Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/hmm_training_real_data.html>`_ for how to train an HMM.

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
# We can see the directory contains two sub-directories:
#
# - `model`: contains the trained HMM.
# - `data`: contains the inferred state probabilities (`alpha.pkl`) and state spectra (`f.npy`, `psd.npy`, `coh.npy`).
#
# State Power Spectra
# ^^^^^^^^^^^^^^^^^^^
# To begin let's examine the power spectra of each state. This has already been calculated after we trained the HMM in the `HMM Training on Real Data tutorial <https://osf.io/cvd7m>`_.
#
# Loading the power spectra
# *************************
# First, let's load the spectra.

import numpy as np

f = np.load("hmm_notts_rest_10_subj/data/f.npy")
psd = np.load("hmm_notts_rest_10_subj/data/psd.npy")
print(f.shape)
print(psd.shape)

#%%
# From the shape of each array we can see `f` is a 1D numpy array which contains the frequency axis of the spectra and `psd` is a (subjects, states, channels, frequencies) array.
#
# Plotting the power spectra
# **************************
# We can plot the power spectra to see what oscillations typically occur when a particular state is on. To help us visualise this we can average over the subjects and channels. This gives us a (states, frequencies) array.

from osl_dynamics.utils import plotting

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
# State Power Maps
# ^^^^^^^^^^^^^^^^
# Calculating power from spectra
# ******************************
# The `psd` array contains the spectrum for each channel (for each subject/state). This is a function of frequency. To calculate power we need to integrate over a frequency range. osl-dynamics has the `analysis.power.variance_from_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.variance_from_spectra>`_ function to help us do this. For example, if we want the power over the alpha band (8-12 Hz), we could do:

from osl_dynamics.analysis import power

# Integrate the power spectra over the alpha band (8-12 Hz)
p = power.variance_from_spectra(f, psd, frequency_range=[8, 12])
print(p.shape)

#%%
# We can see the `p` array is now (subjects, states, channels), it contains the **power map** for each subject and state. To access the power map for the first subject and fourth state, we could use `p[0][3]`, which would be a `(42,)` shape array.
#
# Plotting power maps
# *******************
# We can plot power maps using the `analysis.power.save <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.save>`_ function. 

fig, ax = power.save(
    p[0][3],
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# The fourth state shows a lot of power in the alpha band (seen from the power spectra plot). We see from the power map that this activity is in posterior regions, which is expected.
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
fig, ax = power.save(
    p_mean,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# Because the power is always positive and a lot of the state have power in similar regions all the power maps look similar. To highlight differences we can display the power maps relative to someting. Typically we use the average across states - taking the average across states approximates the static power of the entire training dataset. We can do this by passing the `subtract_mean` argument.

# Takes a few seconds for the power maps to appear
fig, ax = power.save(
    p_mean,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    subtract_mean=True,
)

#%%
# We can now see recognisable functional networks, e.g. a visual, a motor, a frontal network.
#
# Wrap Up
# ^^^^^^^
# - We have shown how to plot state power spectra.
# - We have shown how to calculate power from the spectra and visualise power maps.
