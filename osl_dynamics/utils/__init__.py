"""Utility and helper functions.

Modules
-------
- :py:mod:`~osl_dynamics.utils.array_ops` — Array operations (covariance ↔
  correlation conversions, matrix decompositions, etc.).
- :py:mod:`~osl_dynamics.utils.filenames` — :py:class:`OSLFilenames` for
  managing pipeline file paths.
- :py:mod:`~osl_dynamics.utils.logger` — Logging utilities for pipeline
  scripts.
- :py:mod:`~osl_dynamics.utils.misc` — Miscellaneous helpers (random seeds,
  FSL setup, YAML loading).
- :py:mod:`~osl_dynamics.utils.model` — Model I/O utilities (saving/loading
  configs and weights).
- :py:mod:`~osl_dynamics.utils.plotting` — Plotting functions (brain
  surfaces, power maps, PSDs, state time courses, networks).
- :py:mod:`~osl_dynamics.utils.sklearn_wrappers` — Scikit-learn compatible
  wrappers for osl-dynamics models.
- :py:mod:`~osl_dynamics.utils.topoplots` — Sensor-space topographic plots.
- :py:mod:`~osl_dynamics.utils.workbench` — HCP Workbench integration for
  cortical surface visualisation.
"""

from osl_dynamics.utils import (
    array_ops,
    filenames,
    logger,
    misc,
    model,
    plotting,
    sklearn_wrappers,
    topoplots,
    workbench,
)
from osl_dynamics.utils.misc import set_random_seed, setup_fsl
