"""General Linear Model (GLM) and permutation testing.

Modules
-------
- :py:mod:`~osl_dynamics.glm.base` — :py:class:`DesignConfig` for specifying
  regressors/contrasts and :py:class:`GLM` for fitting the model.
- :py:mod:`~osl_dynamics.glm.ols` — Ordinary least squares implementation.
- :py:mod:`~osl_dynamics.glm.permutation` — :py:class:`MaxStatPermutation`
  for non-parametric group-level inference with family-wise error correction.

Tutorials
---------
- :doc:`Group Contrast </tutorials_build/7-1_group_contrast>`
- :doc:`Network Response </tutorials_build/7-2_group_network_response>`

Python example scripts
----------------------
- `Statistics <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/statistics>`_
"""

from osl_dynamics.glm.base import DesignConfig, GLM
from osl_dynamics.glm.permutation import MaxStatPermutation


def __getattr__(name):
    # Imported on first use because osl_dynamics.utils.plotting is slow to
    # import
    if name == "plot_design_matrix":
        from osl_dynamics.utils.plotting import plot_design_matrix

        return plot_design_matrix
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
