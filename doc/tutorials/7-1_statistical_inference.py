"""
Statistical Inference
=====================

This tutorial is an introduction to statistical significance testing with the :mod:`osl_dynamics.stats <osl_dynamics.stats>` subpackage. It covers:

1. Hypothesis testing at every target (parcel, state, time point, etc.).
2. The General Linear Model (GLM) that all the tests are built on.
3. Permutation tests and when they are exact.
4. The multiple comparisons problem and the corrections available: Bonferroni, Holm, FDR, maximum statistic, cluster-based and TFCE.
5. How to choose a correction and how to interpret and report the results.

The structure follows the excellent `Statistical inference <https://mne.tools/stable/auto_tutorials/stats-sensor-space/10_background_stats.html>`_ tutorial from MNE-Python, applied to the kind of data osl-dynamics produces. The `FieldTrip <https://www.fieldtriptoolbox.org/tutorial/stats/>`_ statistics tutorials are another very good resource.

The practical tutorials that follow show the tests on realistic analyses: :doc:`group contrasts <7-2_group_contrast>`, :doc:`network responses <7-3_network_response>` and :doc:`custom designs <7-4_custom_designs>`.
"""

#%%
# Simulated data
# ^^^^^^^^^^^^^^
# We simulate a **network response**: the state (or mode) time course of a dynamic network model epoched around a task event and averaged over trials, for each subject. This gives an array with shape `(n_subjects, n_time, n_states)`. After baseline correction, the question is whether the response differs from zero at each time point and state.
#
# The noise is smoothed over time so that neighboring time points are correlated, as they are in real data. A real effect is planted in state 1 (samples 30-60) and a smaller one in state 3 (samples 70-85).

import numpy as np
from scipy.ndimage import gaussian_filter1d

np.random.seed(42)

n_subjects, n_time, n_states = 20, 100, 4

noise = np.random.normal(size=(n_subjects, n_time, n_states))
noise = gaussian_filter1d(noise, sigma=2, axis=1)
noise /= noise.std()

true_effect = np.zeros((n_time, n_states))
true_effect[30:60, 0] = 0.8 * np.hanning(30)
true_effect[70:85, 2] = 0.6 * np.hanning(15)

data = noise + true_effect
print(data.shape)

#%%
# We call the first dimension the **samples** (here subjects; it could be sessions or trials) and the remaining dimensions the **targets**: here 100 time points x 4 states = 400 targets. Every function in :mod:`osl_dynamics.stats` works with any number of target dimensions.
#
# Let's plot the true effect and the group average, with time on the x-axis and the states on the y-axis.

import matplotlib.pyplot as plt

def plot_map(values, title, ax, cmap="RdBu_r", vmax=None, **kwargs):
    """Plot a (time, states) map."""
    vmax = np.nanmax(np.abs(values)) if vmax is None else vmax
    im = ax.imshow(
        values.T, aspect="auto", origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax,
        interpolation="nearest", **kwargs,
    )
    ax.set_yticks(range(n_states))
    ax.set_yticklabels([f"State {i + 1}" for i in range(n_states)])
    ax.set_xlabel("Time (samples)")
    ax.set_title(title)
    return im

fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
plot_map(true_effect, "True effect", axes[0])
plot_map(data.mean(axis=0), "Group average (observed)", axes[1])
fig.tight_layout()

#%%
# Hypothesis testing
# ^^^^^^^^^^^^^^^^^^
# At each target we have 20 values (one per subject) and ask: is their mean different from zero? The **null hypothesis** :math:`H_0` is that the mean is zero. A **test statistic** measures the evidence against :math:`H_0`; for a mean the natural choice is the t-statistic
#
# .. math::
#     t = \frac{\bar{x}}{s / \sqrt{n}},
#
# the mean divided by its standard error. The **p-value** is the probability of a statistic at least this extreme if :math:`H_0` were true. Testing every target separately like this is called a **mass-univariate** analysis.
#
# The General Linear Model
# ************************
# All tests in osl-dynamics are expressed as a General Linear Model (GLM): at every target the data :math:`y` (one value per sample) is modeled as
#
# .. math::
#     y = X \beta + \epsilon,
#
# where :math:`X` is the **design matrix** (one row per sample, one column per regressor) and :math:`\beta` are the regression coefficients. A **contrast** :math:`c` is a vector of weights over the regressors; the quantity tested is :math:`c^T \beta`, called the **cope** (contrast of parameter estimates). Its variance is the **varcope** and the t-statistic is :math:`\text{cope} / \sqrt{\text{varcope}}`.
#
# For a one-sample test the design has a single column of ones and the contrast is :code:`[1]`: the cope is the mean. The high-level functions build the design for you, but we can look at it:

from osl_dynamics import stats

design = stats.DesignConfig(
    features=[{"name": "Mean", "values": np.ones(n_subjects), "feature_type": "constant"}],
    contrasts=[{"name": "Mean", "values": [1]}],
).create_design()

fig, axes = design.plot()

#%%
# The same machinery describes the other questions we ask in osl-dynamics. Comparing two groups uses an indicator column per group and the contrast :code:`[1, -1, 0]` (group 1 minus group 2); testing for an effect of age uses a column of ages and the contrast on that column. Nuisance covariates get extra columns with weight zero in the contrast, so their effect is regressed out:

group = np.repeat([1, 2], n_subjects // 2)
age = np.random.normal(45, 10, size=n_subjects)

group_design = stats.DesignConfig(
    features=[
        {"name": "Group 1", "values": group == 1, "feature_type": "categorical"},
        {"name": "Group 2", "values": group == 2, "feature_type": "categorical"},
        {"name": "Age", "values": age, "feature_type": "continuous"},
    ],
    contrasts=[{"name": "Group 1 - Group 2", "values": [1, -1, 0]}],
).create_design()

fig, axes = group_design.plot()

#%%
# Continuous regressors are z-scored, so the cope for a continuous regressor is the change in the data per standard deviation of the regressor.
#
# Parametric test
# ***************
# If the noise is Gaussian, the t-statistic follows a Student t-distribution under :math:`H_0` and the p-value can be read from it. This is the classic parametric t-test. Setting :code:`correction=None` gives the uncorrected parametric p-values:

result = stats.evoked_response(data, correction=None)
print(result)

#%%
# Every test returns a :class:`~osl_dynamics.stats.results.TestResult`. The most useful attributes are :code:`copes` (the effect, here the mean response), :code:`tstats`, :code:`pvalues` and the :code:`significant()` mask, all with the shape of the targets.

fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
plot_map(result.tstats, "t-statistic", axes[0])
plot_map(result.significant(), "p < 0.05 (uncorrected)", axes[1], cmap="Greys", vmax=1)
fig.tight_layout()

print("Significant targets:", result.significant().sum(), "of", result.pvalues.size)

#%%
# The true effects are found, but so are many scattered false positives. We will come back to this.
#
# Permutation tests
# ^^^^^^^^^^^^^^^^^
# The parametric p-value relies on the t-distribution, which assumes Gaussian noise. Summary statistics such as fractional occupancies (bounded between 0 and 1) or lifetimes (skewed) do not satisfy this. **Permutation tests** instead build the null distribution of the statistic from the data itself.
#
# The key idea is **exchangeability**: under :math:`H_0`, some rearrangement of the data leaves its distribution unchanged, so the statistic recomputed after that rearrangement is a draw from the null distribution. Repeating this many times gives the null distribution, and the p-value is the fraction of permuted statistics at least as extreme as the observed one. The two rearrangements used in osl-dynamics are:
#
# - **Sign flipping**: for a test of a mean (one-sample or paired differences), under :math:`H_0` each subject's value is as likely to be positive as negative, so we randomly flip the sign of each subject's data. This is a *within-subject* test and needs the noise to be symmetric about zero.
# - **Row shuffling**: for a comparison between groups or a regression on a continuous variable, under :math:`H_0` the group labels (or covariate values) carry no information, so we shuffle which label goes with which subject. This is a *between-subject* test and needs no distributional assumption at all.
#
# The permutation type is chosen automatically from the design. With nuisance covariates, osl-dynamics uses the Freedman-Lane scheme recommended by `Winkler et al. (2014) <https://doi.org/10.1016/j.neuroimage.2014.01.060>`_: the covariates are regressed out, the residuals are permuted, and the full model is refitted.
#
# Let's run per-target permutation tests (still uncorrected) with :code:`parametric=False`:

result_perm = stats.evoked_response(
    data, correction=None, parametric=False, n_perm=500, metric="tstats"
)

#%%
# The null distribution for each target is kept in :code:`result.null`. For a two-tailed test it is the distribution of :math:`|t|`, so we compare it with the folded t-distribution at a target without a true effect:

from scipy import stats as scipy_stats

t_index, state = 10, 3  # a target with no true effect
null = result_perm.null[:, t_index, state]
observed = abs(result_perm.tstats[t_index, state])

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(null, bins=30, density=True, color="tab:gray", label="Permutation null of |t|")
t_axis = np.linspace(0, 4, 200)
ax.plot(
    t_axis, 2 * scipy_stats.t.pdf(t_axis, df=n_subjects - 1), "k", label="Folded t-distribution"
)
ax.axvline(observed, color="tab:red", label="Observed |t|")
ax.set_xlabel("|t|")
ax.legend()
fig.tight_layout()

print("Permutation p =", round(result_perm.pvalues[t_index, state], 3))
print("Parametric p  =", round(result.pvalues[t_index, state], 3))

#%%
# With Gaussian noise the two agree. With non-Gaussian data only the permutation test is valid.
#
# Two practical points:
#
# - The p-value from :math:`N` random permutations is :math:`(1 + b) / (N + 1)`, where :math:`b` is the number of permuted statistics at least as extreme as the observed one. It can never be zero: the smallest p-value is :math:`1 / (N + 1)`, so use at least 1000 permutations to resolve p = 0.001.
# - With few samples (up to 9 subjects for a sign-flip test with the default 1000 permutations), every possible permutation is enumerated and the test is **exact**; :code:`result.exact` tells you when this happened.
#
# Permutations use NumPy's global random state, so :func:`osl_dynamics.utils.set_random_seed` makes them reproducible.
#
# Multiple comparisons
# ^^^^^^^^^^^^^^^^^^^^
# We tested 400 targets. At p < 0.05, we expect 5% of the targets with no effect to come out significant by chance: about 20 false positives, which is what the uncorrected map showed. With more targets (every parcel and frequency, every connection in a connectivity matrix) the problem gets worse. There are two ways to define what we want to control:
#
# - The **family-wise error rate (FWER)**: the probability of *any* false positive across all the targets. Controlling it at 5% means that in 95% of experiments with no true effect, nothing is declared significant.
# - The **false discovery rate (FDR)**: the expected *fraction* of the significant targets that are false positives. Controlling it at 5% means that, on average, 5% of what you report is wrong. This is less strict and so more powerful.
#
# The :code:`correction` argument of every test selects the correction. We now apply each one to the same data.
#
# Bonferroni and Holm
# *******************
# Bonferroni multiplies every p-value by the number of tests. It controls the FWER but is very conservative when the targets are correlated, as neighboring time points are. Holm's step-down procedure is a uniformly more powerful version with the same guarantee.

result_bonferroni = stats.evoked_response(data, correction="bonferroni")
result_holm = stats.evoked_response(data, correction="holm")

#%%
# False discovery rate
# ********************
# The Benjamini-Hochberg procedure controls the FDR. It is a good choice when a few false positives are acceptable and the targets are not strongly dependent, e.g. for connectivity matrices with thousands of connections.

result_fdr = stats.evoked_response(data, correction="fdr")

#%%
# Maximum statistic permutation test
# **********************************
# The permutation framework gives a natural FWER correction (`Nichols & Holmes, 2002 <https://doi.org/10.1002/hbm.1058>`_): in each permutation, take the **maximum** statistic over all the targets. The distribution of this maximum is the null distribution of "the largest statistic anywhere in the map by chance", and a target is significant if its observed statistic exceeds the 95th percentile of it. This accounts for the correlation between targets, so it is less conservative than Bonferroni, and it makes no distributional assumption. It is the default correction in osl-dynamics.

result_max = stats.evoked_response(data, n_perm=500, metric="tstats")
print(result_max)

#%%
# The :code:`metric` can be :code:`'tstats'` or :code:`'copes'`. Taking the maximum over copes is only valid when all the targets have the same units, as they do for a network response (the default for :func:`~osl_dynamics.stats.inference.evoked_response` is :code:`'copes'`); t-statistics are the default everywhere else. The :code:`hat` argument adds a small fraction of the largest variance to every target's variance, which stops near-zero-variance targets producing huge t-statistics (`Ridgway et al., 2012 <https://doi.org/10.1016/j.neuroimage.2011.10.027>`_).
#
# The null distribution of the maximum can be plotted:

fig, ax = result_max.plot_null()

#%%
# Cluster-based permutation test
# ******************************
# The maximum statistic treats every target on its own. But real effects in time (or frequency, or space) are extended: they form **clusters** of neighboring targets. The cluster-based permutation test (`Maris & Oostenveld, 2007 <https://doi.org/10.1016/j.jneumeth.2007.03.024>`_) uses this:
#
# 1. Threshold the t-statistics at a **cluster-forming threshold** (by default the critical t at p = 0.05).
# 2. Group neighboring supra-threshold targets into clusters and sum their t-statistics to give a **cluster statistic**.
# 3. In each permutation, do the same and record the largest cluster statistic. This is the null distribution.
# 4. A cluster's p-value is the fraction of permutations whose largest cluster was at least as big.
#
# The neighbors are defined by :code:`cluster_dims`: the target dimensions along which clusters can extend. For a network response that is time (dimension 0 of the targets, the default for :func:`~osl_dynamics.stats.inference.evoked_response`). States are not neighbors of each other, so a cluster never spans two states, but the maximum is still taken over all states and the FWER is controlled across the whole map.

result_cluster = stats.evoked_response(data, n_perm=500, correction="cluster")
print(result_cluster)

for mask, p in zip(result_cluster.clusters, result_cluster.cluster_pvalues):
    times = np.where(mask.any(axis=1))[0]
    state_idx = np.where(mask.any(axis=0))[0][0]
    print(f"State {state_idx + 1}, samples {times[0]}-{times[-1]}: p = {p:.3f}")

#%%
# Every target in :code:`result.pvalues` holds the p-value of the cluster it belongs to (1 outside clusters), so :code:`result.significant()` marks the targets in significant clusters.
#
# Threshold-free cluster enhancement
# **********************************
# The cluster-forming threshold is arbitrary: a strict threshold finds focal, strong effects and a lenient one finds broad, weak effects. **Threshold-free cluster enhancement** (TFCE, `Smith & Nichols, 2009 <https://doi.org/10.1016/j.neuroimage.2008.03.061>`_) integrates the cluster extent over all thresholds, giving each target an enhanced statistic that is then tested with the maximum statistic. There is no threshold to choose and each target gets its own p-value.

result_tfce = stats.evoked_response(data, n_perm=500, correction="tfce")
print(result_tfce)

#%%
# Comparison
# ^^^^^^^^^^
# Let's plot which targets each method declares significant. The outline shows the true effect.

results = {
    "Uncorrected (parametric)": result,
    "Bonferroni": result_bonferroni,
    "Holm": result_holm,
    "FDR": result_fdr,
    "Maximum statistic": result_max,
    "Cluster-based": result_cluster,
    "TFCE": result_tfce,
}

fig, axes = plt.subplots(2, 4, figsize=(14, 5))
plot_map(true_effect, "True effect", axes[0, 0])
for ax, (name, res) in zip(axes.flat[1:], results.items()):
    plot_map(res.significant(), f"{name}\n{res.significant().sum()} significant", ax, cmap="Greys", vmax=1)
    ax.contour(true_effect.T > 0.05, levels=[0.5], colors="tab:red", linewidths=1)
fig.tight_layout()

#%%
# What to take from this:
#
# - The uncorrected map has false positives scattered everywhere. Never report it alone.
# - Bonferroni, Holm and the maximum statistic all control the FWER. The maximum statistic is less conservative because it accounts for the correlation between targets, but it still only finds the peak of a weak, extended effect.
# - FDR finds more of the effect at the cost of a few false positives.
# - The cluster-based test uses the temporal structure of the effect and recovers the most of it, including the weaker effect in state 3. TFCE sits between the maximum statistic and the cluster-based test here. These two are the standard choices for time courses and spectra.
#
# Which correction should I use?
# ******************************
# .. list-table::
#    :header-rows: 1
#    :widths: 30 35 35
#
#    * - Data
#      - Recommended
#      - Also reasonable
#    * - Summary statistics (subjects x states), power maps (subjects x states x parcels)
#      - Maximum statistic
#      - FDR; Holm
#    * - Connectivity matrices (subjects x connections)
#      - Maximum statistic
#      - FDR (many connections, some false positives acceptable)
#    * - Network responses (subjects x time x states)
#      - Cluster-based or TFCE along time
#      - Maximum statistic on copes (focal, strong effects)
#    * - Spectra (subjects x states x parcels x frequency)
#      - Cluster-based or TFCE along frequency
#      - Maximum statistic
#
# Use a one-tailed test (:code:`tail=1` or :code:`-1`) only when the direction of the effect was fixed before looking at the data. Use :code:`pooled_dims` when you want to control the FWER within each state separately rather than across all states.
#
# Interpreting cluster-based results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The cluster-based test rejects one null hypothesis: that the data are exchangeable, i.e. that there is no effect anywhere. A significant cluster tells you *that* there is an effect and points to where it is likely to be. It does **not** establish that the effect is present at every target in the cluster, nor that it starts and ends where the cluster does: the cluster's extent depends on the signal-to-noise ratio, the number of subjects and the cluster-forming threshold. In particular, you cannot claim that an effect "starts at 150 ms" or "is confined to state 3" from a cluster test, see `Sassenhagen & Draschkow (2019) <https://doi.org/10.1111/psyp.13335>`_ and the `FieldTrip FAQ <https://www.fieldtriptoolbox.org/faq/stats/clusterstats_interpretation/>`_.
#
# Acceptable: "Network 1 showed a significant response (cluster-based permutation test, p = 0.002), most pronounced between 100 and 300 ms." Not acceptable: "Network 1 was significantly active from 100 to 300 ms."
#
# The same caveat applies, less strongly, to TFCE: its per-target p-values are still driven by neighboring targets.
#
# Reporting
# ^^^^^^^^^
# A results sentence should state the test, the design, the correction, the number of permutations, the tail and what was controlled for. For example:
#
#     "Group differences in fractional occupancy were assessed with a GLM permutation test (row-shuffle, 10,000 permutations, two-tailed) with age and sex as covariates, using the maximum t-statistic across states to control the family-wise error rate at 5%."
#
# :code:`print(result)` gives most of these values and :code:`result.summary()` returns them as a dictionary.
#
# References
# ^^^^^^^^^^
# - Maris, E. & Oostenveld, R. (2007). Nonparametric statistical testing of EEG- and MEG-data. *Journal of Neuroscience Methods*, 164(1), 177-190.
# - Nichols, T. E. & Holmes, A. P. (2002). Nonparametric permutation tests for functional neuroimaging: a primer with examples. *Human Brain Mapping*, 15(1), 1-25.
# - Phipson, B. & Smyth, G. K. (2010). Permutation p-values should never be zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1), 39.
# - Ridgway, G. R. et al. (2012). The problem of low variance voxels in statistical parametric mapping; a new hat avoids a 'haircut'. *NeuroImage*, 59(3), 2131-2141.
# - Sassenhagen, J. & Draschkow, D. (2019). Cluster-based permutation tests of MEG/EEG data do not establish significance of effect latency or location. *Psychophysiology*, 56(6), e13335.
# - Smith, S. M. & Nichols, T. E. (2009). Threshold-free cluster enhancement. *NeuroImage*, 44(1), 83-98.
# - Winkler, A. M. et al. (2014). Permutation inference for the general linear model. *NeuroImage*, 92, 381-397.
