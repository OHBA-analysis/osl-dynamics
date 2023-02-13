"""
HMM: Summary Statistic Analysis
===============================
 
In this tutorial we will analyse the dynamic networks inferred by a Hidden Markov Model (HMM) on resting-state source reconstructed MEG data. This tutorial covers:

1. Download a Trained Model
2. Summary Statistics
3. Comparing Groups

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
# In this section, we'll look at a three popular summary statistics:
#
# - The fractional occupancy, which is the fraction of total time spent in a particular state.
# - The mean lifetime, which is the average duration a state is activate.
# - The mean interval, which is the average duration between successive state visits.
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
# Mean interval
# *************
#
# Finally, let's look at the mean interval time for each state and subject. We can use the `analysis.modes.mean_intervals <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/modes/index.html#osl_dynamics.analysis.modes.mean_intervals>`_ function in osl-dynamics to calculate this.

# Calculate mean intervals (in seconds)
mintv = modes.mean_intervals(stc, sampling_frequency=250)

# Print the group average
print(np.mean(mintv, axis=0))

# Plot distribution across subjects
plotting.plot_violin(mintv.T, x_label="State", y_label="Mean Interval (s)")

#%%
# Comparing Groups
# ^^^^^^^^^^^^^^^^
#
# Now we have summary statistics for each subject, we can use them as features for downstream tasks. A common analysis we might want to do is compare two groups, e.g. healthy vs diseased participants. The dataset used to train the HMM in this tutorial contained healthy subjects, however for illustration, let's say subjects \[0, 1, 2\] belong to group 1 and subjects \[3, 4, 5, 6, 7, 8, 9\] belong to group 2.
#
# Plotting distributions
# **********************
#
# To start, just visualise the distribution of summary statistics for each group.

import pandas as pd

# Group assignments:
# - 0 indicates group 1.
# - 1 indicates group 2.
assignments = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

# Create a pandas dataframe containing the summary stats
# This will be helpful for plotting later
# Each line in the dataframe is a subject
ss_dict = {"fo": [], "lt": [], "intv": [], "state": [], "group": []}
n_subjects, n_states = fo.shape
for subject in range(n_subjects):
    for state in range(n_states):
        ss_dict["fo"].append(fo[subject, state])
        ss_dict["lt"].append(mlt[subject, state])
        ss_dict["intv"].append(mintv[subject, state])
        ss_dict["state"].append(state + 1)
        ss_dict["group"].append(assignments[subject] + 1)
ss_df = pd.DataFrame(ss_dict)

# Display the dataframe
ss_df

#%%
# Now, let's plot the distribution of fractional occupancies.

import seaborn as sns

sns.violinplot(data=ss_df, x="state", y="fo", hue="group", split=True)

#%%
# We can see the distribution for each group looks pretty similar for most states. However, if there were differences we would want to perform a statistical significant test to verify it's not due to change.
#
# Statistical Significance Testing
# ********************************
#
# We will use a **maximum statistic permutation test** to see if any differences in summary statistics between our groups are significant. See the `Statistical Significance Testing tutorial <https://osf.io/ft3rm>`_ for further details.

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

# Let's focus on the fractional occupancies
#
# Generate a null distribution
null = null_distribution(fo, assignments, n_perm=1000)

# Calculate a threshold for significance
p_value = 0.05
thres = np.percentile(null, 100 * (1 - p_value), axis=0)

# See which elements are significant
fo1 = np.mean(fo[assignments == 0], axis=0)
fo2 = np.mean(fo[assignments == 1], axis=0)
abs_diff = np.abs(fo1 - fo2)
sig = abs_diff > thres
print(sig)

#%%
# We see in this case there are no significant differences between the fractional occupancies of the two groups.
#
# Wrap Up
# ^^^^^^^
#
# - We have shown how to calculate some common summary statistics based on the inferred state time course.
# - We have shown how to compare the summary statistics of groups of subjects.
