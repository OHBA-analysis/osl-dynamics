"""Post-hoc analysis of inferred state/mode dynamics.

This subpackage provides tools for analysing the output of trained models
(HMM states, DyNeMo modes, etc.) and for static (time-averaged) analysis.

See also
--------
- :doc:`Static Spectral Analysis </tutorials_build/2-1_static_spectra_analysis>`
- :doc:`Static Power Analysis </tutorials_build/2-2_static_power_analysis>`
- :doc:`Static AEC Analysis </tutorials_build/2-3_static_aec_analysis>`
- :doc:`Sliding Window Analysis </tutorials_build/3-1_sliding_window_analysis>`
- :doc:`HMM Multitaper Spectra </tutorials_build/4-1_hmm_multitaper_spectra>`
- :doc:`HMM Summary Statistics </tutorials_build/4-3_hmm_summary_stats>`
- :doc:`HMM Plotting MEG Networks </tutorials_build/4-2_hmm_plotting_meg_networks>`
- :doc:`HMM fMRI Dual Estimation </tutorials_build/4-4_hmm_fmri_dual_estimation>`
- :doc:`DyNeMo Regression Spectra </tutorials_build/5-1_dynemo_regression_spectra>`
- :doc:`DyNeMo Plotting Networks </tutorials_build/5-2_dynemo_plotting_networks>`
- :doc:`DyNeMo Mixing Coefficients </tutorials_build/5-3_dynemo_mixing_coefs>`

Modules
-------
- ``connectivity.py`` — Functional connectivity analysis (coherence,
  imaginary coherence, correlation).
- ``fisher_kernel.py`` — Fisher kernel for comparing HMM dynamics across
  sessions.
- ``post_hoc.py`` — Post-hoc spectral estimation of state/mode spectra
  using multitaper or regression methods.
- ``power.py`` — Power analysis (variance from spectra, band-limited power).
- ``prediction.py`` — Decoding/prediction using inferred dynamics.
- ``spectral.py`` — Spectral decomposition of state/mode covariances into
  power maps and coherence networks.
- ``static.py`` — Static (time-averaged) power and functional connectivity.
- ``statistics.py`` — Statistical testing (permutation tests, group
  comparisons).
"""

from osl_dynamics.analysis import (
    connectivity,
    fisher_kernel,
    post_hoc,
    power,
    spectral,
    static,
    statistics,
    prediction,
)
