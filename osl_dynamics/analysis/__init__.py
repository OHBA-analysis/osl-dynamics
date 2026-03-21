"""Post-hoc analysis of inferred state/mode dynamics.

This subpackage provides tools for analysing the output of trained models
(HMM states, DyNeMo modes, etc.) and for static (time-averaged) analysis.

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
