"""
Group Contrast
==============

In this tutorial we will cover how to do statistical significance testing with GLM permutations.
"""

#%%
# Get features
# ^^^^^^^^^^^^
# First we need to estimate subject-specific features. For example, this can be the static power map for each subject, which would be a (subjects, parcels) array. Or this could be the HMM state power maps for each subjects, this would be a (subjects, states, parcels) array.
#
# Here, we will simulate random data of shape (subjects, features).

import numpy as np

features = np.random.normal(size=(30,10))

#%%
# We have simulated 30 subjects with 10 features. Let say the first 10 subjects form one group and the second 20 are another group. These groups could correspond to disease or healthy controls for example. Let's add an effect in the first 5 features of the first group.

features[:10,:5] += 3

#%%
# Statistical significance testing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# osl-dynamics has the `analysis.statistics.group_diff_max_stat_perm <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/statistics/index.html#osl_dynamics.analysis.statistics.group_diff_max_stat_perm>`_ function for doing GLM permutation stats testing to compare two groups. This function uses the maximum test statistic to control for multiple comparisons across the features.

from osl_dynamics.analysis import statistics

# 1 = group 1, 2 = group 2
assignments = np.ones(30)
assignments[10:] += 1

group_diff, pvalues = statistics.group_diff_max_stat_perm(
    features,
    assignments,
    n_perm=1000,
    n_jobs=4,
)
print(group_diff.shape)
print(pvalues.shape)

#%%
# Note, any multidimensional array can be passed to `group_diff_max_stat_perm`, e.g. (subjects, features1, featues2, ...), as long as the first dimension corresponds to the subjects. This function also has a `covariates` argument that can be used to account for confounds.
#
# Let's see which features came out as significant based on the p-value.

print(pvalues < 0.05)

#%%
# We see the first 5 features are significantly different as expected.
