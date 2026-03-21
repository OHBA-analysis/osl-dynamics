"""General Linear Model (GLM) and permutation testing.

Modules
-------
- ``base.py`` — :py:class:`DesignConfig` for specifying regressors/contrasts
  and :py:class:`GLM` for fitting the model.
- ``ols.py`` — Ordinary least squares implementation.
- ``permutation.py`` — :py:class:`MaxStatPermutation` for non-parametric
  group-level inference with family-wise error correction.
"""

from osl_dynamics.glm.base import DesignConfig, GLM
from osl_dynamics.glm.permutation import MaxStatPermutation
