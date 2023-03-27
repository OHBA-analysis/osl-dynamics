"""
DyNeMo: Plotting Networks
=========================

In this tutorial we will analyse the dynamic networks inferred by DyNeMo on resting-state source reconstructed MEG data. This tutorial covers:
 
1. Downloading a Trained Model
2. Power Maps
3. Coherence Networks

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/ejn2r>`_ for the expected output.
"""

#%%
# Downloading a Trained Model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# First, let's download a model that's already been trained on a resting-state dataset. See the `DyNeMo Training on Real Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/dynemo_training_real_data.html>`_ for how to train DyNeMo.

import os

def get_trained_model(name):
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p by2tc fetch trained_models/{name}.zip")
    os.system(f"unzip -o {name}.zip -d {name}")
    os.remove(f"{name}.zip")
    return f"Model downloaded to: {name}"

# Download the trained model (approximately 33 MB)
model_name = "dynemo_notts_rest_10_subj"
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
# - `data`: contains the inferred mixing coefficients (`alpha.pkl`) and mode spectra (`f.npy`, `psd.npy`, `coh.npy`).
# 
# Power Maps
# ^^^^^^^^^^
# 
# Let's start by plotting the inferred power maps. First we need to calculate the power maps from the mode spectra. Let's first load the mode spectra.

import numpy as np

f = np.load("dynemo_notts_rest_10_subj/data/f.npy")
psd = np.load("dynemo_notts_rest_10_subj/data/psd.npy")
print(f.shape)
print(psd.shape)

#%%
# We can see form the shape of these arrays: `f` is the frequency axis and `psd` is a (subjects, 2, modes, channels, frequencies) array. The second axis in `psd` corresponds to the regression coefficients and intercept when we calculated the mode spectra using `analysis.spectral.regression_spectra <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.regression_spectra>`_. The intercept term corresponds to the time average (static) spectra and the regression coefficients describe the variation about the average. We're not interested in the static spectra, so we just want to retain the regression coefficients.

psd_coefs = psd[:, 0]
print(psd_coefs.shape)

#%%
# Now let's integrate the power spectra to calcualte the power.

from osl_dynamics.analysis import power

p = power.variance_from_spectra(f, psd_coefs)
print(p.shape)

#%%
# We can see `p` is a (subjects, modes, channels) array. Let's average over subjects.

mean_p = np.mean(p, axis=0)
print(mean_p.shape)

#%%
# Now we can plot the power map for each mode.

# Display the power maps (takes a few seconds to appear)
power.save(
    mean_p,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
    subtract_mean=True,  # just for visualisation
)

#%%
# We can see recognisable functional networks, which gives us confidence in the DyNeMo fit. We also see the networks are more localised than typical HMM states.
# 
# Coherence Networks
# ^^^^^^^^^^^^^^^^^^
# 
# Next, let's visualise the coherence networks. First, we need to load the coherence spectra.

coh = np.load("dynemo_notts_rest_10_subj/data/coh.npy")
print(coh.shape)

#%%
# We can see the coherence spectra is a (subjects, modes, channels, channels, frequencies) array. Now let's calculate the mean coherence over all frequencies.

from osl_dynamics.analysis import connectivity

c = connectivity.mean_coherence_from_spectra(f, coh)
print(c.shape)

#%%
# We now have a (subjects, modes, channels, channels) array. Next we need to average over subjects and threshold the coherence networks.

# Average over subjects
mean_c = np.mean(c, axis=0)

# Threshold the top 3% relative to the mean
thres_mean_c = connectivity.threshold(mean_c, percentile=97, subtract_mean=True)

#%%
# Now we can visualise the networks.

connectivity.save(
    thres_mean_c,
    parcellation_file="fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz",
)

#%%
# We can see the coherence networks show high coherence in the same regions with high power. We expect these networks will improve with more subjects.
# 
# Wrap Up
# ^^^^^^^
# 
# - We have shown how to calculate power maps and coherence networks from mode spectra calculated using the regression approach.
