"""Custom TensorFlow optimizers."""

from packaging import version

import tensorflow as tf

if version.parse(tf.__version__) < version.parse("2.12"):
    from keras.optimizers.optimizer_v2.optimizer_v2 import OptimizerV2 as Optimizer
elif version.parse(tf.__version__) < version.parse("2.13"):
    from keras.optimizers.legacy.optimizer_v2 import OptimizerV2 as Optimizer
else:
    from keras.optimizers import Optimizer


class ExponentialMovingAverage(Optimizer):
    """Optimizer for applying a exponential moving average update.

    Parameters
    ----------
    decay : float
        Decay for the exponential moving average, which will be
        calculated as :code:`(1-decay) * old + decay * new`.
    """

    def __init__(self, learning_rate, decay=0.1):
        super().__init__(learning_rate, name="EMAOptimizer")
        self.decay = tf.Variable(decay, trainable=False, name="ema_decay")

    def update_step(self, gradient, variable, learning_rate):
        value = (1.0 - self.decay) * variable + self.decay * gradient
        self.assign(variable, value)


class MarkovStateModelOptimizer(Optimizer):
    """Optimizer for a model containing a hidden state Markov chain.

    Parameters
    ----------
    base_optimizer : tf.keras.optimizers.Optimizer
        A TensorFlow optimizer for all other trainable model variables.
    ema_optimizer : osl_dynamics.inference.optimizers.ExponentialMovingAverage
        Exponential moving average optimizer.
    ema_variable : list
        List of trainable variables to update with the EMA optimizer.
    learning_rate : float
        Learning rate for the base optimizer.
    """

    def __init__(self, base_optimizer, ema_optimizer, ema_variables, learning_rate):
        super().__init__(learning_rate, name="MarkovStateModelOptimizer")
        self.base_optimizer = base_optimizer
        self.ema_optimizer = ema_optimizer
        self.ema_variable_ids = [id(v) for v in ema_variables]

    def apply_gradients(self, grads_and_vars, **kwargs):
        # Update base optimizer learning rate
        self.base_optimizer.learning_rate.assign(self.learning_rate)

        # Split variables
        base_grads, base_vars = [], []
        ema_grads, ema_vars = [], []
        for g, v in grads_and_vars:
            if id(v) in self.ema_variable_ids:
                ema_grads.append(g)
                ema_vars.append(v)
            else:
                base_grads.append(g)
                base_vars.append(v)

        # Apply gradients with the base optimizer
        if base_grads and base_vars:
            self.base_optimizer.apply_gradients(zip(base_grads, base_vars))

        # Apply gradients with the EMA optimizer
        if ema_grads and ema_vars:
            self.ema_optimizer.apply_gradients(zip(ema_grads, ema_vars))
