"""
Custom Designs
==============

The high-level functions in :mod:`osl_dynamics.stats` cover the common questions: a response against zero (:func:`~osl_dynamics.stats.inference.evoked_response`), paired differences (:func:`~osl_dynamics.stats.inference.paired_diff`), two groups (:func:`~osl_dynamics.stats.inference.group_diff`) and the effect of a continuous variable (:func:`~osl_dynamics.stats.inference.regression`). This tutorial shows how to go beyond them:

1. Testing the effect of a continuous variable such as age.
2. Specifying your own design matrix and contrasts.
3. Parametric tests, one-tailed tests and uncorrected permutation p-values.
4. Cluster-based tests with a custom adjacency (e.g. neighboring parcels).

See the :doc:`Statistical Inference tutorial <7-1_statistical_inference>` for the background.
"""

#%%
# Simulated data
# ^^^^^^^^^^^^^^
# We simulate fractional occupancies for 40 subjects and 8 states, where the occupancy of state 3 increases with age, and record each subject's age, sex and group (two patient groups and controls).

import numpy as np

np.random.seed(1)

n_subjects, n_states = 40, 8
age = np.random.uniform(20, 80, size=n_subjects)
sex = np.random.randint(0, 2, size=n_subjects)
group = np.random.choice([1, 2, 3], size=n_subjects)  # 1 = controls

fo = np.random.normal(size=(n_subjects, n_states))
fo[:, 2] += 0.05 * (age - age.mean())

#%%
# Effect of a continuous variable
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :func:`stats.regression <osl_dynamics.stats.inference.regression>` fits a constant plus the variable of interest (and any nuisance covariates) and tests the slope with a row-shuffle permutation test. The cope is the change in the data per standard deviation of the variable, because continuous regressors are z-scored.

from osl_dynamics import stats

result = stats.regression(fo, age, name="Age", covariates={"sex": sex}, n_perm=1000)
print(result)
print("Slope per SD of age:", result.copes.round(3))
print("Significant states:", np.where(result.significant())[0] + 1)

#%%
# Your own design
# ^^^^^^^^^^^^^^^
# Any GLM can be specified with :class:`~osl_dynamics.stats.design.DesignConfig`: a list of regressors (features) and a list of contrasts. Feature types are :code:`'constant'`, :code:`'categorical'` (0/1 indicators) and :code:`'continuous'` (z-scored). Here we compare the two patient groups together against controls with the contrast :code:`[-1, 0.5, 0.5, 0, 0]`, controlling for age and sex:

design = stats.DesignConfig(
    features=[
        {"name": "Controls", "values": group == 1, "feature_type": "categorical"},
        {"name": "Patients A", "values": group == 2, "feature_type": "categorical"},
        {"name": "Patients B", "values": group == 3, "feature_type": "categorical"},
        {"name": "Age", "values": age, "feature_type": "continuous"},
        {"name": "Sex", "values": sex, "feature_type": "continuous"},
    ],
    contrasts=[
        {"name": "Patients - Controls", "values": [-1, 0.5, 0.5, 0, 0]},
        {"name": "Patients A - Patients B", "values": [0, 1, -1, 0, 0]},
    ],
).create_design()

fig, axes = design.plot()

#%%
# A design is tested with one of the permutation classes. :class:`~osl_dynamics.stats.permutation.MaxStatPermutation` uses the maximum statistic over the targets; :class:`~osl_dynamics.stats.permutation.ClusterPermutation` the cluster-based test. :code:`contrast_index` selects the contrast. The permutation type is chosen from the design (row shuffles for a categorical difference) unless :code:`perm_type` is passed.

perm = stats.MaxStatPermutation(design, contrast_index=0, n_perm=1000)
result = perm.fit(fo)
print(result)

#%%
# For a differential contrast between categorical regressors, only the subjects in the groups with a non-zero weight are shuffled. Here that is everyone; when comparing two of three groups it is the subjects in those two.
#
# F-contrasts
# ^^^^^^^^^^^
# A t-contrast tests one difference in one direction. An **F-contrast** tests several t-contrasts jointly: is *any* of them non-zero? This is the GLM form of an ANOVA, and it is how :func:`stats.group_diff <osl_dynamics.stats.inference.group_diff>` handles three or more groups. Here we ask whether the three groups differ at all, controlling for age and sex. The two t-contrasts already in the design span every difference between the three groups, so the F-contrast can be built from them (any set of linearly independent rows spanning the same differences gives the same F-statistic):

design.add_f_contrast(
    "Any group difference",
    contrasts=["Patients - Controls", "Patients A - Patients B"],
)

result = stats.MaxStatPermutation(design, contrast_index=0, n_perm=1000, statistic="F").fit(fo)
print(result)
print("F-statistics:", np.round(result.stats, 2))

#%%
# :code:`statistic="F"` makes :code:`contrast_index` refer to the F-contrasts of the design. An F-test has no direction, so :code:`tail` must be 0 and the permutation statistic is always F. :code:`result.copes` holds the copes of each row of the F-contrast. An F-contrast can also be given directly as a matrix, one t-contrast per row, with :code:`values=[[-1, 0.5, 0.5, 0, 0], [0, 1, -1, 0, 0]]` or in :class:`~osl_dynamics.stats.design.DesignConfig` with :code:`f_contrasts=[{"name": ..., "values": ...}]`. The parametric version is :code:`stats.GLM(design).fit(fo).ftest(0)`.
#
# Parametric tests
# ^^^^^^^^^^^^^^^^
# :class:`~osl_dynamics.stats.glm.GLM` fits the design and returns every regression coefficient, cope and varcope. Its :meth:`~osl_dynamics.stats.results.GLMResult.test` method gives parametric p-values from the t-distribution with an optional Bonferroni, Holm or FDR correction. Use this when the data are approximately Gaussian or you have too few subjects for permutations to be informative.

glm_result = stats.GLM(design).fit(fo)
print(glm_result.betas.shape, glm_result.copes.shape)  # (features, states), (contrasts, states)

result = glm_result.test(contrast_index=0, correction="fdr")
print(result)

#%%
# The high-level functions give the same with :code:`correction="fdr"` (or :code:`"bonferroni"`, :code:`"holm"`, :code:`None`). With :code:`parametric=False` the uncorrected p-values come from per-target permutations instead and the correction is applied to them, which does not assume Gaussian data. This needs enough permutations for the smallest attainable p-value, :math:`1 / (N + 1)`, to pass the corrected threshold:

result = stats.regression(fo, age, correction="fdr", parametric=False, n_perm=5000)
print(result)

#%%
# One-tailed tests
# ^^^^^^^^^^^^^^^^
# If the direction of the effect is fixed in advance, :code:`tail=1` tests for a positive effect and :code:`tail=-1` for a negative one. A one-tailed test is more powerful in that direction and blind to the other; it must be decided before looking at the data.

result = stats.regression(fo, age, n_perm=1000, tail=1)
print(result.pvalues.round(3))

#%%
# Clusters with a custom adjacency
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Cluster-based tests connect neighboring indices along :code:`cluster_dims`. For dimensions that are not a lattice, such as parcels or sensors, pass an adjacency matrix: :code:`adjacency={dim: matrix}` where :code:`matrix[i, j]` is non-zero when targets :code:`i` and :code:`j` are neighbors. For sensors, MNE's :func:`mne.channels.find_ch_adjacency` provides one; for parcels it can be built from the parcel centroids.
#
# Here we simulate power maps over time for 12 parcels arranged in a ring, so that parcel :code:`i` is adjacent to :code:`i - 1` and :code:`i + 1`, with an effect in parcels 1-3 from sample 20 to 40:

from scipy import sparse

n_time, n_parcels = 60, 12
power = np.random.normal(size=(n_subjects, n_time, n_parcels))
power[:, 20:40, :3] += 0.8

ring = sparse.lil_matrix((n_parcels, n_parcels))
for i in range(n_parcels):
    ring[i, (i + 1) % n_parcels] = 1
    ring[(i + 1) % n_parcels, i] = 1

result = stats.evoked_response(
    power,
    n_perm=1000,
    correction="cluster",
    cluster_dims=(0, 1),  # time and parcels
    adjacency={1: ring},  # parcels are connected through the ring
)
print(result)
for mask, p in zip(result.clusters, result.cluster_pvalues):
    print(
        f"Samples {np.where(mask.any(axis=1))[0][[0, -1]]}, "
        f"parcels {np.where(mask.any(axis=0))[0] + 1}: p = {p:.3f}"
    )

#%%
# Without :code:`adjacency`, every cluster dimension is a lattice, which is right for time and frequency. Dimensions not in :code:`cluster_dims` (states, modes, unrelated parcels) are never spanned by a cluster, but the maximum is taken over them so the family-wise error rate still covers the whole array.
#
# Reproducibility
# ^^^^^^^^^^^^^^^
# Permutations use NumPy's global random state. Call :func:`osl_dynamics.utils.set_random_seed` at the top of a script to make them reproducible. Results can be saved and reloaded::
#
#     result.save("age_effect.pkl")
#     result = stats.TestResult.load("age_effect.pkl")
