"""
HMM: Summary Statistic Analysis
===============================

In this tutorial we will analyse the dynamic networks inferred by a Hidden Markov Model (HMM) on resting-state source reconstructed MEG data. This tutorial covers:

1. Download a Trained Model
2. Summary Statistics
3. Comparing Groups

Note, this webpage does not contain the output of running each cell. See `OSF <https://osf.io/2zp4e>`_ for the expected output.
"""

#%%
# Download a Trained Model
# ^^^^^^^^^^^^^^^^^^^^^^^^
# First, let's download a model that's already been trained on a resting-state dataset. See the `HMM Training on Real Data tutorial <https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/hmm_summary_stats_analysis.html>`_ for how to train an HMM.

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
# Summary Statistics
# ^^^^^^^^^^^^^^^^^^
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

from osl_dynamics.inference import modes

# Hard classify the state probabilities
stc = modes.argmax_time_courses(alpha)

# Plot the state time course for the first subject (8 seconds)
plotting.plot_alpha(stc[0], n_samples=2000)

#%%
# We can see this time series is completely binarised, only one state is active at each time point.
#
# Fractional occupancy
# ********************
# The `analysis.modes <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/modes/index.html>`_ module in osl-dynamics contains helpful functions for calculating summary statistics. Let's use the `analysis.modes.fractional_occupancies <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/modes/index.html#osl_dynamics.analysis.modes.fractional_occupancies>`_ function to calculate the fractional occupancy of each state for each subject.

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

# Plot the distribution of fractional occupancy (FO) across subjects
plotting.plot_violin(fo.T, x_label="State", y_label="FO")

#%%
# We can see there is a lot of variation across subjects.
#
# Mean Lifetime
# *************
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
# Now we have summary statistics for each subject, we can use them as features for downstream tasks. A common analysis we might want to do is compare two groups, e.g. healthy vs diseased participants. The dataset used to train the HMM in this tutorial contained healthy subjects, however for illustration, let's say subjects \[0, 1, 2\] belong to group 1 and subjects \[3, 4, 5, 6, 7, 8, 9\] belong to group 2.
#
# Statistical significance testing
# ********************************
# We would like to test is the difference in summary statistics for each group is significant. Fortunately, osl-dynamics has a function we can use for this: `analysis.statistics.group_diff_max_stat_perm <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/statistics/index.html#osl_dynamics.analysis.statistics.group_diff_max_stat_perm>`_. This function uses maximum statistic permutation testing to test for significance.

from osl_dynamics.analysis import statistics

# Test for a different in any of the summary statistics
# It's important to combine all the summary statistics into one test to
# make sure we're correcting for multiple comparisons
sum_stats = [fo, mlt, mintv]
sum_stats = np.swapaxes(sum_stats, 0, 1)  # make sure subjects is the first axis

# Group assignments
assignments = np.array([1,1,1,2,2,2,2,2,2,2])

# Do stats testing
diff, pvalues = statistics.group_diff_max_stat_perm(
    sum_stats,
    assignments,
    n_perm=100,
)

# Are any summary stat differences significant?
print("Number of significant differences:", np.sum(pvalues < 0.05))

#%%
# Unfortunely, no differences came out as significant in this case.
#
# Wrap Up
# ^^^^^^^
# - We have shown how to calculate some common summary statistics based on the inferred state time course.
# - We have shown how to compare the summary statistics of groups of subjects.
