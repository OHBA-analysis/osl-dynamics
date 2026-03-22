"""TensorFlow layers, callbacks, and utilities used by the models.

This subpackage contains the building blocks that the model classes in
:py:mod:`osl_dynamics.models` use internally. Most users will not need to
interact with this subpackage directly.

Modules
-------
- :py:mod:`~osl_dynamics.inference.layers` — Custom Keras layers (RNN
  inference networks, softmax layers, sampling layers, observation model
  layers, etc.).
- :py:mod:`~osl_dynamics.inference.callbacks` — Training callbacks (KL
  annealing, transition probability updates, EMA).
- :py:mod:`~osl_dynamics.inference.initializers` — Weight initialisers for
  observation model parameters.
- :py:mod:`~osl_dynamics.inference.metrics` — Loss metrics and evaluation
  functions.
- :py:mod:`~osl_dynamics.inference.modes` — Utilities for manipulating
  inferred mode/state time courses (reordering, matching, correlation).
- :py:mod:`~osl_dynamics.inference.optimizers` — Custom optimisers.
- :py:mod:`~osl_dynamics.inference.regularizers` — Regularisers for
  observation model parameters.
- :py:mod:`~osl_dynamics.inference.tf_ops` — TensorFlow utility operations.
"""

from osl_dynamics.inference import metrics
from osl_dynamics.inference import modes
from osl_dynamics.inference import tf_ops
