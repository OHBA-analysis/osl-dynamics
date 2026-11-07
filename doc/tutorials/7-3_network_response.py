"""
Network Response
================

In this tutorial we test whether a **network response** is significantly different from zero. A network response is the state (or mode) time course epoched around events of interest and averaged over trials, for each subject (see the :doc:`Epoching tutorial <6-1_epoching_alpha>`). We compare the three corrections for multiple comparisons that suit time courses: the maximum statistic, cluster-based permutations and threshold-free cluster enhancement (TFCE). See the :doc:`Statistical Inference tutorial <7-1_statistical_inference>` for the background.
"""

#%%
# Get the network response
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The network response for each subject is a `(time, networks)` array, so for the group it is a `(subjects, time, networks)` array. It should be **baseline corrected** (the pre-stimulus mean subtracted) so that zero is the null hypothesis. Here we simulate it, with noise that is smooth in time as in real data.

import numpy as np
from scipy.ndimage import gaussian_filter1d

np.random.seed(0)

n_subjects, n_time, n_networks = 20, 250, 6
sampling_frequency = 250  # Hz
t = np.arange(n_time) / sampling_frequency - 0.2  # seconds, stimulus at t=0

network_response = np.random.normal(size=(n_subjects, n_time, n_networks))
network_response = gaussian_filter1d(network_response, sigma=3, axis=1)
network_response /= network_response.std()

# Network 1 responds from 0.12 to 0.32 s, network 2 from 0.32 to 0.4 s
network_response[:, 80:130, 0] += 1.5 * np.hanning(50)
network_response[:, 130:150, 1] += 1.8 * np.hanning(20)

# Baseline correct using the pre-stimulus period
network_response -= network_response[:, t < 0].mean(axis=1, keepdims=True)

#%%
# Maximum statistic
# ^^^^^^^^^^^^^^^^^
# :func:`stats.evoked_response <osl_dynamics.stats.inference.evoked_response>` fits a GLM with a constant regressor (the mean over subjects) at every time point and network and runs a sign-flip permutation test. The default correction takes the maximum over all time points and networks, so the family-wise error rate (FWER) is controlled across the whole response.

from osl_dynamics import stats

result_max = stats.evoked_response(network_response, n_perm=1000)
print(result_max)

#%%
# For network responses the maximum is taken over the copes (the mean response) rather than the t-statistics by default, because every target has the same units and the t-statistic can be unstable where the variance is close to zero (e.g. in the baseline window). Pass :code:`metric="tstats"` to change this.
#
# :func:`plotting.plot_evoked_response <osl_dynamics.utils.plotting.plot_evoked_response>` plots the group average with the significant time points marked:

from osl_dynamics.utils import plotting

labels = [f"Network {i + 1}" for i in range(n_networks)]

fig, ax = plotting.plot_evoked_response(
    t,
    network_response.mean(axis=0),
    result_max,
    labels=labels,
    x_label="Time (s)",
    y_label="Network activation",
    title="Maximum statistic",
)

#%%
# Cluster-based permutation test
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A response is extended in time, which the maximum statistic ignores. :code:`correction="cluster"` thresholds the t-statistics, groups neighbouring supra-threshold time points into clusters (within each network: clusters never span two networks) and tests the summed t-statistic of each cluster against the largest cluster found in each permutation.

result_cluster = stats.evoked_response(network_response, n_perm=1000, correction="cluster")
print(result_cluster)

for mask, stat, p in zip(
    result_cluster.clusters, result_cluster.cluster_stats, result_cluster.cluster_pvalues
):
    network = np.where(mask.any(axis=0))[0][0]
    times = t[mask.any(axis=1)]
    print(
        f"Network {network + 1}: {times[0]:.3f} to {times[-1]:.3f} s, "
        f"cluster statistic = {stat:.1f}, p = {p:.3f}"
    )

#%%
# The cluster-forming threshold defaults to the critical t-value at p = 0.05 and can be changed with :code:`cluster_forming_threshold`. Each time point in :code:`result.pvalues` holds the p-value of its cluster, so the same plotting function marks the significant clusters:

fig, ax = plotting.plot_evoked_response(
    t,
    network_response.mean(axis=0),
    result_cluster,
    labels=labels,
    x_label="Time (s)",
    y_label="Network activation",
    title="Cluster-based permutation test",
)

#%%
# Threshold-free cluster enhancement
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# TFCE avoids choosing a cluster-forming threshold by integrating the cluster extent over all thresholds, and gives each time point its own p-value.

result_tfce = stats.evoked_response(network_response, n_perm=1000, correction="tfce")

fig, ax = plotting.plot_evoked_response(
    t,
    network_response.mean(axis=0),
    result_tfce,
    labels=labels,
    x_label="Time (s)",
    y_label="Network activation",
    title="TFCE",
)

#%%
# Comparing the three
# ^^^^^^^^^^^^^^^^^^^
# Let's count the significant time points for the two networks with a response:

for name, res in [("Max statistic", result_max), ("Cluster", result_cluster), ("TFCE", result_tfce)]:
    sig = res.significant()
    print(f"{name:14s} network 1: {sig[:, 0].sum():3d} samples, network 2: {sig[:, 1].sum():3d} samples")

#%%
# The cluster-based test recovers the most of the extended response in network 1, and TFCE more than the maximum statistic. For the brief response in network 2 the three are closer. The maximum statistic is better suited to focal, strong effects and to targets without a natural neighbourhood (parcels, states).
#
# Interpreting the clusters
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# A significant cluster shows that the network responds; it does not establish *when* the response starts and ends, because the extent of a cluster depends on the signal-to-noise ratio and the threshold. Report it as "network 1 showed a significant response (cluster-based permutation test, 1000 sign-flip permutations, p = 0.001), most pronounced between 0.12 and 0.32 s", not as "network 1 was active from 0.12 to 0.32 s". See the :doc:`Statistical Inference tutorial <7-1_statistical_inference>` for more on this.
#
# Comparing conditions
# ^^^^^^^^^^^^^^^^^^^^
# Often each subject has a response per condition and the question is whether the responses differ between conditions. For two conditions, pass the difference between the subject-specific responses to :func:`stats.paired_diff <osl_dynamics.stats.inference.paired_diff>`. For any number of conditions, :func:`stats.condition_diff <osl_dynamics.stats.inference.condition_diff>` takes a `(subjects, conditions, time, networks)` array and fits a within-subject GLM: an indicator per condition plus an indicator per subject, so that the differences between subjects are modelled rather than left in the noise. The permutations shuffle the conditions within each subject (the subjects are **exchangeability blocks**). With three or more conditions it runs an F-test for any difference; :code:`conditions=(a, b)` tests a pair.
#
# Here we simulate three conditions where the response of network 1 is larger in condition 2, and the subjects differ in their overall response amplitude:

n_conditions = 3
responses = np.random.normal(size=(n_subjects, n_conditions, n_time, n_networks))
responses = gaussian_filter1d(responses, sigma=3, axis=2)
responses /= responses.std()
responses[:, :, 80:130, 0] += 1.5 * np.hanning(50)
responses[:, 2, 80:130, 0] += 2.5 * np.hanning(50)  # larger in condition 2
responses += np.random.normal(size=(n_subjects, 1, 1, 1))  # subject amplitude
responses -= responses[:, :, t < 0].mean(axis=2, keepdims=True)

result_cond = stats.condition_diff(responses, correction="cluster", cluster_dims=0, n_perm=1000)
print(result_cond)
print("Networks with a condition effect:", np.unique(np.where(result_cond.significant())[1]))

#%%
# :code:`result_cond.stats` holds the F-statistics and :code:`result_cond.copes` the differences between consecutive conditions. The F-test says the conditions differ; which ones differ is a follow-up question:

result_02 = stats.condition_diff(responses, conditions=(2, 0), correction="cluster", cluster_dims=0, n_perm=1000)
print(result_02)

#%%
# Covariates and groups
# ^^^^^^^^^^^^^^^^^^^^^
# Nuisance covariates can be passed with the :code:`covariates` argument of the between-subject tests. To compare two groups of subjects use :func:`stats.group_diff <osl_dynamics.stats.inference.group_diff>` with :code:`correction="cluster"` and :code:`cluster_dims=0` (time).
