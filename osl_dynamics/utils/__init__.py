"""Utility and helper functions.

Modules
-------
- :py:mod:`~osl_dynamics.utils.array_ops` — Array operations (covariance ↔
  correlation conversions, matrix decompositions, etc.).
- :py:mod:`~osl_dynamics.utils.logger` — Logging utilities for pipeline
  scripts.
- :py:mod:`~osl_dynamics.utils.misc` — Miscellaneous helpers (random seeds,
  FSL setup, YAML loading).
- :py:mod:`~osl_dynamics.utils.model` — Model I/O utilities (saving/loading
  configs and weights).
- :py:mod:`~osl_dynamics.utils.plotting` — Plotting functions (brain
  surfaces, power maps, PSDs, state time courses, networks).
- :py:mod:`~osl_dynamics.utils.progress` — Progress bars, including for
  functions applied in parallel.
- :py:mod:`~osl_dynamics.utils.sklearn_wrappers` — Scikit-learn compatible
  wrappers for osl-dynamics models.
- :py:mod:`~osl_dynamics.utils.topoplots` — Sensor-space topographic plots.
- :py:mod:`~osl_dynamics.utils.workbench` — HCP Workbench integration for
  cortical surface visualisation.
"""

import importlib

from osl_dynamics.utils.misc import set_random_seed, setup_fsl

# Submodules are imported on first use because some (e.g. plotting) are slow
# to import
_submodules = [
    "array_ops",
    "logger",
    "misc",
    "model",
    "plotting",
    "progress",
    "sklearn_wrappers",
    "topoplots",
    "workbench",
]


def __getattr__(name):
    if name in _submodules:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + _submodules)
