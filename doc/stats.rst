:orphan:

Statistical Significance Testing
================================

The :mod:`osl_dynamics.stats <osl_dynamics.stats>` subpackage is the single place for statistical significance testing in osl-dynamics. Every test fits a General Linear Model (GLM) at each **target** (parcel, state, connection, time point, frequency, etc.) and corrects for the multiple comparisons across the targets, by default with non-parametric permutation tests that make no assumption about the distribution of the data.

Start with the :doc:`Statistical Inference tutorial <tutorials_build/7-1_statistical_inference>`, which explains hypothesis testing, the GLM, permutation tests and every correction with figures. Then see the practical tutorials:

- :doc:`Group Contrast <tutorials_build/7-2_group_contrast>`: comparing two groups of subjects (summary statistics, power maps, connectivity), with covariates.
- :doc:`Network Response <tutorials_build/7-3_network_response>`: testing an epoched state/mode time course against zero with the maximum statistic, cluster-based permutations and TFCE.
- :doc:`Custom Designs <tutorials_build/7-4_custom_designs>`: continuous regressors, your own design matrices and contrasts, parametric and one-tailed tests, clusters with a custom adjacency.

Which function?
---------------

All tests take an array with the subjects (or sessions) as the first dimension and any number of target dimensions, and return a :class:`~osl_dynamics.stats.results.TestResult` with the effect (:code:`copes`), t-statistics, corrected p-values and a :code:`significant()` mask of the same shape as the targets.

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
   * - Does a quantity change with age, score, etc.?
     - :func:`~osl_dynamics.stats.inference.regression`
     - Row shuffle (between subject)
   * - Something else (more groups, custom contrasts)
     - :class:`~osl_dynamics.stats.design.DesignConfig` with :class:`~osl_dynamics.stats.permutation.MaxStatPermutation`, :class:`~osl_dynamics.stats.permutation.ClusterPermutation` or :class:`~osl_dynamics.stats.glm.GLM`
     - Chosen from the design

Nuisance covariates (age, sex, head size, ...) are passed to every function as :code:`covariates={"age": age, ...}` and regressed out with the Freedman-Lane permutation scheme.

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

FWER is the family-wise error rate (the probability of any false positive); FDR is the false discovery rate (the expected fraction of false positives among the significant targets).

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
   * - Spectra
     - (subjects, states, parcels, frequency)
     - :code:`group_diff`, :code:`'cluster'` with :code:`cluster_dims=2` (frequency).
   * - DyNeMo mixing coefficient statistics
     - (subjects, modes)
     - As summary statistics.

Implementation notes
--------------------

- P-values from :math:`N` random permutations are :math:`(1 + b) / (N + 1)`, so the smallest attainable p-value is :math:`1 / (N + 1)`. Use at least 1000 permutations (10,000 for p < 0.001). When every permutation can be enumerated the test is exact (:code:`result.exact`).
- Sign-flip tests need the noise to be symmetric about zero, which holds for baseline-corrected responses and paired differences. Row-shuffle tests make no distributional assumption.
- Targets with missing values are fitted with the subjects that have a value for them (per-target complete-case analysis), with a warning.
- Permutations use NumPy's global random state; :func:`osl_dynamics.utils.set_random_seed` makes them reproducible.
- Random field theory (as in SPM) is not provided: it needs a smooth Gaussian field on a lattice, which parcels, states and connections are not. The permutation tests control the FWER without these assumptions.

Also see the :doc:`FAQ <faq>` and the `example scripts <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/statistics>`_.
