"""Statistical significance testing.

Tests are built on a General Linear Model (GLM) fitted with ordinary least
squares at every target (e.g. every parcel, state or time point), with
non-parametric permutation tests to control the family-wise error rate
across targets.

Modules
-------
- :py:mod:`~osl_dynamics.stats.inference` — Ready-made tests:
  :func:`evoked_response`, :func:`paired_diff`, :func:`group_diff` and
  :func:`regression`.
- :py:mod:`~osl_dynamics.stats.design` — :class:`DesignConfig` and
  :class:`Design` for specifying regressors and contrasts.
- :py:mod:`~osl_dynamics.stats.glm` — :class:`GLM` fitting.
- :py:mod:`~osl_dynamics.stats.permutation` — :class:`MaxStatPermutation`
  (family-wise error control) and :class:`Permutation` (uncorrected).
- :py:mod:`~osl_dynamics.stats.correction` — :func:`bonferroni`,
  :func:`holm` and :func:`fdr` corrections for p-values.
- :py:mod:`~osl_dynamics.stats.results` — :class:`TestResult` and
  :class:`GLMResult` returned by the tests.

Tutorials
---------
- :doc:`Group Contrast </tutorials_build/7-1_group_contrast>`
- :doc:`Network Response </tutorials_build/7-2_group_network_response>`

Python example scripts
----------------------
- `Statistics <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/statistics>`_
"""

from osl_dynamics.stats.correction import bonferroni, fdr, holm
from osl_dynamics.stats.design import Design, DesignConfig
from osl_dynamics.stats.glm import GLM
from osl_dynamics.stats.inference import (
    evoked_response,
    group_diff,
    paired_diff,
    regression,
)
from osl_dynamics.stats.permutation import MaxStatPermutation, Permutation
from osl_dynamics.stats.results import GLMResult, TestResult
