"""
HMM: Summary Statistic Analysis
===============================
 
In this tutorial we will analyse the dynamic networks inferred by a Hidden Markov Model (HMM) on resting-state source reconstructed MEG data. This tutorial covers:
 
1. Download a Trained Model
2. Summary Statistics

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/ryb9q>`_ for the expected output.
"""

#%%
# Download a Trained Model
# ^^^^^^^^^^^^^^^^^^^^^^^^
# 
# First, let's download a model that's already been trained on a resting-state dataset. See the `HMM Training on Real Data tutorial <https://osf.io/cvd7m>`_ for how to train an HMM.

import os

def get_trained_model(name):
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p zxb6c fetch Dynamics/data/trained_models/{name}.zip")
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
# Summary Statistics
# ^^^^^^^^^^^^^^^^^^
# 
# Now we have a trained HMM, we want to use it to help us understand the training data. A useful way of characterising the data is to use statistics that summarise the state time course, these are referred to as 'summary statistics'. The state time course captures the dynamics in the training data, therefore, summary statistics characterise the dynamics of the data.
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

alpha = pickle.load(open(f"{model_name}/data/alpha.pkl", "rb"))

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
plotting.plot_violin(fo.T, x_label="State", y_label="FO")

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
plotting.plot_violin(mlt.T, x_label="State", y_label="Mean Lifetime (ms)")

#%%
# Wrap Up
# ^^^^^^^
# 
# - We have shown how to calculate some common summary statistics based on the inferred state time course.
