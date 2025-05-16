"""
Network Response
================

In this tutorial we will cover how to do statistical significance testing for a 'network response'. This is the average state/mode time course epoched around events of interest. We test is the response is significantly different from zero.
"""

#%%
# Get network response
# ^^^^^^^^^^^^^^^^^^^^
# First we need to get the network response for each subject. This is the epoched state/mode time course averaged over trials, which would be a (subjects, states/modes, time) array. In this tutorial, we will simulate this.

import numpy as np

n_subjects = 20
n_time = 250
n_networks = 6

network_response = np.random.normal(size=(n_subjects, n_time, n_networks))

# Add non-zero network response for the first network for time points 30-80
network_response[:, 30:80, 0] += 1.5

# Add non-zero network response for the second network for time points 80-100
network_response[:, 80:100, 1] += 2

#%%
# Statistical significance testing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# osl-dynamics has the `analysis.statistics.evoked_response_max_stat_perm <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/statistics/index.html#osl_dynamics.analysis.statistics.evoked_response_max_stat_perm>`_ function for doing GLM permutation stats testing see if a value is significantly different from zero. This function uses the maximum test statistic to control for multiple comparisons across the time points and networks.

from osl_dynamics.analysis import statistics

pvalues = statistics.evoked_response_max_stat_perm(
    network_response,
    n_perm=1000,
    n_jobs=4,
)
print(pvalues.shape)

#%%
# Let's plot the group average network response with the significant time points highlighted.

from osl_dynamics.utils import plotting

t = np.arange(n_time)
group_network_response = np.mean(network_response, axis=0)
fig, ax = plotting.plot_evoked_response(
    t,
    group_network_response,
    pvalues,
    x_label="Sample",
    y_label="Network Activation",
)

#%%
# Note, this function also has a `covariates` argument that can be used to account for confounds.
