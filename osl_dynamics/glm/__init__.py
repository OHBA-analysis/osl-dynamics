"""General Linear Model (GLM) and permutation testing.

Tutorials
---------
- :doc:`Group Contrast </tutorials_build/7-1_group_contrast>`
- :doc:`Network Response </tutorials_build/7-2_group_network_response>`

Python example scripts
----------------------
- `Statistics <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/statistics>`_

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
