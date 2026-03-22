"""Post-hoc analysis of inferred state/mode dynamics.

This subpackage provides tools for analysing the output of trained models
(HMM states, DyNeMo modes, etc.) and for static (time-averaged) analysis.

Modules
-------
- :py:mod:`~osl_dynamics.analysis.connectivity` — Functional connectivity
  analysis (coherence, imaginary coherence, correlation).
- :py:mod:`~osl_dynamics.analysis.fisher_kernel` — Fisher kernel for
  comparing HMM dynamics across sessions.
- :py:mod:`~osl_dynamics.analysis.post_hoc` — Post-hoc spectral estimation
  of state/mode spectra using multitaper or regression methods.
- :py:mod:`~osl_dynamics.analysis.power` — Power analysis (variance from
  spectra, band-limited power).
- :py:mod:`~osl_dynamics.analysis.prediction` — Decoding/prediction using
  inferred dynamics.
- :py:mod:`~osl_dynamics.analysis.spectral` — Spectral decomposition of
  state/mode covariances into power maps and coherence networks.
- :py:mod:`~osl_dynamics.analysis.static` — Static (time-averaged) power
  and functional connectivity.
- :py:mod:`~osl_dynamics.analysis.statistics` — Statistical testing
  (permutation tests, group comparisons).

Tutorials
---------
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

Python example scripts
----------------------
- `Spectra <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/spectra>`_
- `Static analysis <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/static>`_
- `Statistics <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/statistics>`_
- `MEG analysis <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/meg_analysis>`_
- `fMRI analysis <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/fmri>`_
- `Decoding <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/decoding>`_
- `Plotting <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/plotting>`_
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
