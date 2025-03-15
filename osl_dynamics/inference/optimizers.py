"""Custom TensorFlow optimizers."""

import tensorflow as tf


class ExponentialMovingAverage(tf.keras.Optimizer):
    """Optimizer for applying a exponential moving average update.

    Parameters
    ----------
    decay : float
        Decay for the exponential moving average, which will be
        calculated as :code:`(1-decay) * old + decay * new`.
    """

    def __init__(self, learning_rate, decay=0.1):
        super().__init__(learning_rate, name="EMAOptimizer")
        self.decay = decay

    def update_step(self, gradient, variable, learning_rate):
        learning_rate = tf.cast(learning_rate, variable.dtype)
        gradient = tf.cast(gradient, variable.dtype)
        self.assign(variable, (1.0 - self.decay) * variable + self.decay * gradient)


class MarkovStateModelOptimizer(tf.keras.Optimizer):
    """Optimizer for a model containing a hidden state Markov chain.

    Parameters
    ----------
    ema_optimizer : osl_dynamics.inference.optimizers.ExponentialMovingAverage
        Exponential moving average optimizer for the transition
        probability matrix.
    base_optimizer : tf.keras.optimizers.Optimizer
        A TensorFlow optimizer for all other trainable model variables.
    learning_rate : float
        Learning rate for the base optimizer.
    """

    def __init__(self, ema_optimizer, base_optimizer, learning_rate):
        super().__init__(learning_rate, name="MarkovStateModelOptimizer")

        # Moving average optimizer for the transition probability matrix
        self.ema_optimizer = ema_optimizer

        # Optimizer for all other trainable variables
        self.base_optimizer = base_optimizer

    def build(self, var_list):
        self.base_optimizer.build(
            [v for v in var_list if "hid_state_inf" not in v.name]
        )

    def update_step(self, gradient, variable, learning_rate):
        if "hid_state_inf" in variable.name:
            self.ema_optimizer.update_step(gradient, variable, learning_rate)
        else:
            self.base_optimizer.update_step(gradient, variable, learning_rate)
