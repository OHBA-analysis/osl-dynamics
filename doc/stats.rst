Statistical Significance Testing
================================

:mod:`osl_dynamics.stats <osl_dynamics.stats>` fits a General Linear Model (GLM) at each **target** (parcel, state, connection, time point, frequency) and corrects for the multiple comparisons across targets, by default with permutation tests that make no assumption about the distribution of the data. The :doc:`Statistical Inference tutorial <tutorials_build/7-1_statistical_inference>` explains the methods with figures; the :doc:`statistics tutorials <tutorial_sections/statistics>` apply them to group contrasts, network responses and custom designs.

Which function?
---------------

All tests take an array with the subjects as the first dimension and any number of target dimensions, and return a :class:`~osl_dynamics.stats.results.TestResult` with the effect (:code:`copes`), the statistic (:code:`stats`), corrected :code:`pvalues` and a :code:`significant()` mask of the same shape as the targets.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Question
     - Function
     - Permutations
   * - Is a (baseline-corrected) response different from zero?
     - :func:`~osl_dynamics.stats.inference.evoked_response`
     - Sign flip (within subject)
   * - Is a paired difference (condition A - B) different from zero?
     - :func:`~osl_dynamics.stats.inference.paired_diff`
     - Sign flip (within subject)
   * - Do two groups differ?
     - :func:`~osl_dynamics.stats.inference.group_diff`
     - Row shuffle (between subject)
   * - Do three or more groups differ at all? (one-way ANOVA)
     - :func:`~osl_dynamics.stats.inference.group_diff` with all the labels (an F-test); then :code:`groups=(a, b)` for pairwise follow-ups
     - Row shuffle (between subject)
   * - Does a quantity change with age, score, etc.?
     - :func:`~osl_dynamics.stats.inference.regression`
     - Row shuffle (between subject)
   * - Do several conditions measured in the same subjects differ? (repeated measures ANOVA)
     - :func:`~osl_dynamics.stats.inference.condition_diff` (an F-test, or a pair with :code:`conditions=(a, b)`)
     - Row shuffle within each subject (exchangeability blocks)
   * - Something else (custom contrasts, F-contrasts over several t-contrasts)
     - :class:`~osl_dynamics.stats.design.DesignConfig` with :class:`~osl_dynamics.stats.permutation.MaxStatPermutation`, :class:`~osl_dynamics.stats.permutation.ClusterPermutation` or :class:`~osl_dynamics.stats.glm.GLM`
     - Chosen from the design

Nuisance covariates are passed to the between-subject functions as :code:`covariates={"age": age, ...}`. Every result has an :meth:`~osl_dynamics.stats.results.TestResult.effect_size` method giving Cohen's d (or partial eta squared for an F-test) with bootstrap confidence intervals.

Which correction?
-----------------

The :code:`correction` argument selects the correction for multiple comparisons.

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - :code:`correction`
     - Controls
     - Use for
   * - :code:`'max_stat'` (default)
     - FWER
     - Targets without a natural neighbourhood: states, parcels, connections, summary statistics. Accounts for the correlation between targets. :code:`pooled_dims` restricts the family, e.g. to the parcels within each state.
   * - :code:`'cluster'`
     - FWER
     - Effects extended along time or frequency (network responses, spectra). Clusters are formed along :code:`cluster_dims`; a custom :code:`adjacency` connects e.g. neighbouring parcels. A cluster's p-value does not establish its exact extent or latency.
   * - :code:`'tfce'`
     - FWER
     - As :code:`'cluster'` without a cluster-forming threshold, with per-target p-values.
   * - :code:`'fdr'`
     - FDR
     - Many targets where a small fraction of false positives is acceptable, e.g. connectivity matrices. Parametric by default; :code:`parametric=False` uses permutation p-values.
   * - :code:`'bonferroni'`, :code:`'holm'`
     - FWER
     - Few, roughly independent targets. Conservative for correlated targets.
   * - :code:`None`
     - Nothing
     - Exploration only. Never report uncorrected p-values alone.

FWER is the family-wise error rate (the probability of any false positive); FDR the false discovery rate (the expected fraction of false positives among the significant targets).

Which data?
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - osl-dynamics output
     - Shape
     - Recommended
   * - Summary statistics (fractional occupancy, lifetimes, intervals, switching rates)
     - (subjects, states)
     - :code:`group_diff` / :code:`regression`, :code:`'max_stat'`. Undefined lifetimes (NaN) are handled per target.
   * - State/mode power maps
     - (subjects, states, parcels)
     - :code:`group_diff` / :code:`regression`, :code:`'max_stat'`; :code:`pooled_dims=1` to correct within each state.
   * - Static power maps
     - (subjects, parcels)
     - :code:`group_diff` / :code:`regression`, :code:`'max_stat'`.
   * - Connectivity (AEC, coherence)
     - (subjects, [states,] parcels, parcels)
     - Upper triangle with :func:`~osl_dynamics.utils.array_ops.mat2vec`, then :code:`'max_stat'` or :code:`'fdr'`; back with :func:`~osl_dynamics.utils.array_ops.vec2mat`.
   * - Network response (epoched state/mode time course)
     - (subjects, time, states)
     - :code:`evoked_response`, :code:`'cluster'` or :code:`'tfce'` along time (default), or :code:`'max_stat'` on copes.
   * - Network response per condition
     - (subjects, conditions, time, states)
     - :code:`condition_diff`, :code:`'cluster'` with :code:`cluster_dims=0` (time); :code:`paired_diff` on the difference for two conditions.
   * - Spectra
     - (subjects, states, parcels, frequency)
     - :code:`group_diff`, :code:`'cluster'` with :code:`cluster_dims=2` (frequency).
   * - DyNeMo mixing coefficient statistics
     - (subjects, modes)
     - As summary statistics.

Implementation notes
--------------------

- P-values from :math:`N` random permutations are :math:`(1 + b) / (N + 1)`, so the smallest attainable p-value is :math:`1 / (N + 1)`: use at least 1000 permutations (10,000 for p < 0.001). When every permutation can be enumerated the test is exact (:code:`result.exact`).
- Sign-flip tests need the noise to be symmetric about zero (true of baseline-corrected responses and paired differences); row-shuffle tests make no distributional assumption.
- Targets with missing values are fitted with the subjects that have a value for them, with a warning.
- Permutations use NumPy's global random state; :func:`osl_dynamics.utils.set_random_seed` makes them reproducible.
- F-tests have no direction (:code:`tail` is always 0); :code:`result.copes` holds the copes of the rows of the F-contrast.
- Within-subject designs shuffle only within each subject (the :code:`blocks` argument of the permutation classes).
- Effect size confidence intervals are percentile bootstrap intervals over subjects; :code:`n_boot=0` gives the parametric t interval of the cope.

Also see the :doc:`FAQ <faq>` and the `example scripts <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/statistics>`_.
