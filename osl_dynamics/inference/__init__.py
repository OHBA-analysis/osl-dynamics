"""TensorFlow layers, callbacks, and utilities used by the models.

This subpackage contains the building blocks that the model classes in
:py:mod:`osl_dynamics.models` use internally. Most users will not need to
interact with this subpackage directly.

Modules
-------
- ``layers.py`` — Custom Keras layers (RNN inference networks, softmax
  layers, sampling layers, observation model layers, etc.).
- ``callbacks.py`` — Training callbacks (KL annealing, transition
  probability updates, EMA).
- ``initializers.py`` — Weight initialisers for observation model parameters.
- ``metrics.py`` — Loss metrics and evaluation functions.
- ``modes.py`` — Utilities for manipulating inferred mode/state time courses
  (reordering, matching, correlation).
- ``optimizers.py`` — Custom optimisers.
- ``regularizers.py`` — Regularisers for observation model parameters.
- ``tf_ops.py`` — TensorFlow utility operations.
"""

from osl_dynamics.inference import metrics
from osl_dynamics.inference import modes
from osl_dynamics.inference import tf_ops
