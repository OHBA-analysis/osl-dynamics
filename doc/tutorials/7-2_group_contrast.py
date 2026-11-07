"""
Group Contrast
==============

In this tutorial we test for differences between groups of subjects with :func:`stats.group_diff <osl_dynamics.stats.inference.group_diff>`. This covers the most common group-level questions in osl-dynamics:

1. Do the HMM summary statistics differ between groups?
2. Do the state power maps differ between groups?
3. Do the connectivity networks differ between groups?

We also cover nuisance covariates, missing values, controlling the family-wise error rate per state and more than two groups. See the :doc:`Statistical Inference tutorial <7-1_statistical_inference>` for the background.
"""

#%%
# Subject-specific features
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Group-level statistics start from a subject-specific quantity, e.g. the fractional occupancy of each state for each subject (a `(subjects, states)` array), the state power maps (`(subjects, states, parcels)`) or the connectivity networks (`(subjects, states, parcels, parcels)`). The first dimension must be the subjects; the remaining dimensions are the **targets** that are tested.
#
# Here, we simulate 30 subjects (the first 10 form group 1, the rest group 2) with 6 states and 38 parcels, and plant a difference in state 2 in a few parcels.

import numpy as np

np.random.seed(0)

n_subjects, n_states, n_parcels = 30, 6, 38
assignments = np.ones(n_subjects)
assignments[10:] = 2

power_maps = np.random.normal(size=(n_subjects, n_states, n_parcels))
power_maps[assignments == 1, 1, :5] += 1.5

#%%
# Comparing two groups
# ^^^^^^^^^^^^^^^^^^^^
# :func:`stats.group_diff <osl_dynamics.stats.inference.group_diff>` fits a GLM with an indicator regressor per group and tests the contrast group 1 minus group 2. By default, it uses a row-shuffle permutation test with the maximum t-statistic across all targets to control the family-wise error rate (FWER).

from osl_dynamics import stats

result = stats.group_diff(power_maps, assignments, n_perm=1000)
print(result)

#%%
# The result object holds the group difference (`copes`), the t-statistics and the corrected p-values, each with shape `(states, parcels)`:

print(result.copes.shape, result.pvalues.shape)

significant = result.significant()  # pvalues < 0.05
print("Significant (state, parcel) pairs:", np.argwhere(significant).tolist())

#%%
# The design matrix and contrast that were used can be plotted from the result:

fig, axes = result.glm.design.plot()

#%%
# To plot a power map of the significant differences, zero the non-significant parcels and pass the array to :func:`power.save <osl_dynamics.analysis.power.save>` as in the :doc:`Static Power Analysis tutorial <2-2_static_power_analysis>`::
#
#     power_diff = result.copes.copy()
#     power_diff[~result.significant()] = 0
#     power.save(power_diff, parcellation_file=..., filename="power_diff_.png")
#
# Nuisance covariates
# ^^^^^^^^^^^^^^^^^^^
# Confounds such as age, sex or head size are passed as :code:`covariates`. They are added to the design as nuisance regressors (z-scored for continuous variables) and regressed out; the permutations account for them with the Freedman-Lane scheme.

age = np.random.normal(45, 12, size=n_subjects)
sex = np.random.randint(0, 2, size=n_subjects)  # 0/1 indicator

result = stats.group_diff(
    power_maps, assignments, n_perm=1000, covariates={"age": age, "sex": sex}
)
fig, axes = result.glm.design.plot()

#%%
# Controlling the FWER per state
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# By default, the maximum statistic is taken over all targets (every state and parcel), so the FWER is controlled across the whole array. If you consider each state a separate family of tests, pass :code:`pooled_dims` to take the maximum over the parcels only (target dimension 1). Each state then gets its own null distribution, which is more powerful but does not correct across states.

result_per_state = stats.group_diff(power_maps, assignments, n_perm=1000, pooled_dims=1)
print(result_per_state.null.shape)  # one null distribution per state

#%%
# Summary statistics and missing values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# HMM summary statistics such as the mean lifetime are undefined (NaN) for a state a subject never visits. Each target is then fitted with the subjects that have a value for it, and a warning reports how many were dropped.

mean_lifetimes = np.random.gamma(shape=5, scale=0.02, size=(n_subjects, n_states))
mean_lifetimes[assignments == 1, 0] *= 1.4  # longer lifetimes for state 1 in group 1
mean_lifetimes[3, 4] = np.nan  # subject 4 never visits state 5

result = stats.group_diff(mean_lifetimes, assignments, n_perm=1000)
print(result.pvalues.round(3))

#%%
# Lifetimes are positive and skewed, so the permutation test is preferable to a parametric t-test here. :func:`plotting.plot_summary_stats_group_diff <osl_dynamics.utils.plotting.plot_summary_stats_group_diff>` shows the two groups with stars for the significant states:

from osl_dynamics.utils import plotting

fig, ax = plotting.plot_summary_stats_group_diff(
    name="Mean Lifetime (s)",
    summary_stats=mean_lifetimes,
    pvalues=result,
    assignments=assignments,
)

#%%
# Connectivity matrices
# ^^^^^^^^^^^^^^^^^^^^^
# A connectivity matrix is symmetric, so each connection appears twice. Testing the whole matrix would test every connection twice and halve the effective significance threshold. Use :func:`array_ops.mat2vec <osl_dynamics.utils.array_ops.mat2vec>` to extract the upper triangle, test the vector, and :func:`array_ops.vec2mat <osl_dynamics.utils.array_ops.vec2mat>` to go back to a matrix for plotting.

from osl_dynamics.utils import array_ops

# Simulate (subjects, parcels, parcels) connectivity matrices
conn = np.random.normal(size=(n_subjects, n_parcels, n_parcels))
conn = (conn + np.swapaxes(conn, 1, 2)) / 2
conn[assignments == 1, 0, 1] += 2
conn[assignments == 1, 1, 0] += 2

conn_vec = array_ops.mat2vec(conn)  # (subjects, n_connections)
print(conn_vec.shape)

result = stats.group_diff(conn_vec, assignments, n_perm=1000)

conn_diff = result.copes.copy()
conn_diff[~result.significant()] = 0
conn_diff = array_ops.vec2mat(conn_diff)  # (parcels, parcels)
print("Significant connections:", np.argwhere(np.triu(conn_diff) != 0).tolist())

#%%
# The matrix can be plotted with :func:`connectivity.save <osl_dynamics.analysis.connectivity.save>`. With thousands of connections, the maximum statistic can be conservative; :code:`correction="fdr"` is a reasonable alternative if a small fraction of false positives is acceptable.
#
# More than two groups
# ^^^^^^^^^^^^^^^^^^^^
# When :code:`assignments` has more than two labels and no :code:`groups` are given, :func:`stats.group_diff <osl_dynamics.stats.inference.group_diff>` tests whether *any* group differs from the others with an **F-test**: the GLM form of a one-way ANOVA, with row-shuffle permutations of all the subjects. An F-statistic has no direction, so the test is always two-tailed and :code:`result.copes` holds the differences between consecutive groups rather than a single effect; :code:`result.stats` gives the F-statistics.

assignments_3 = assignments.copy()
assignments_3[-5:] = 3

result = stats.group_diff(power_maps, assignments_3, n_perm=1000)
print(result)
print("Numerator degrees of freedom:", result.f_dof)

#%%
# A significant F-test says that the groups are not all the same, not which ones differ. Follow it up with pairwise tests by passing the two labels to compare as :code:`groups`. Every group is still modeled, so the residual variance is estimated from all subjects, but only the subjects in the two compared groups are shuffled.

result_12 = stats.group_diff(power_maps, assignments_3, groups=(1, 2), n_perm=1000)
print(result_12.contrast)

#%%
# Each pairwise test controls the FWER across the targets on its own. If you report all the pairs, correct for their number as well (e.g. Bonferroni over the three pairs) or restrict yourself to the pairs you had a hypothesis about. Other contrasts (e.g. patients versus the mean of two control groups) can be specified with a custom design, see the :doc:`Custom Designs tutorial <7-4_custom_designs>`.
#
# Effect sizes
# ^^^^^^^^^^^^
# A p-value says whether an effect is detectable, not how large it is. :meth:`result.effect_size() <osl_dynamics.stats.results.TestResult.effect_size>` returns the group difference (the cope, in the units of the data) and Cohen's d (the difference divided by the residual standard deviation of the GLM, i.e. the pooled standard deviation adjusted for any covariates), each with a confidence interval from a bootstrap over subjects. For an F-test it returns partial eta squared instead of d.

result = stats.group_diff(power_maps, assignments, n_perm=1000)
effect = result.effect_size(n_boot=1000)
print(effect)

for parcel in range(5):
    lo, hi = effect.d_ci[:, 1, parcel]
    print(f"State 2, parcel {parcel}: d = {effect.d[1, parcel]:.2f} [{lo:.2f}, {hi:.2f}]")

#%%
# Reporting
# ^^^^^^^^^
# :code:`print(result)` summarizes the test. For example, the analysis above would be reported as: "Power maps were compared between groups with a GLM permutation test (1000 row-shuffle permutations, two-tailed) with age and sex as covariates, using the maximum t-statistic across states and parcels to control the family-wise error rate at 5%. Group 1 had higher power in state 2 (Cohen's d = 1.1 to 1.6 in the significant parcels, 95% bootstrap confidence intervals in Table 1)."
