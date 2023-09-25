"""Custom TensorFlow optimizers.

"""

import tensorflow as tf
from keras.optimizers.optimizer_v2 import optimizer_v2


class ExponentialMovingAverage(optimizer_v2.OptimizerV2):
    """Optimizer for applying a exponential moving average update.

    Parameters
    ----------
    decay : float
        Decay for the exponential moving average, which will be
        calculated as :code:`(1-decay) * old + decay * new`.
    """

    def __init__(self, decay=0.1):
        super().__init__(name="EMAOptimizer")
        self.decay = decay

    @tf.function
    def _resource_apply_dense(self, grad, var):
        # grad should be the new value for var
        grad = tf.cast(grad, var.dtype)
        return var.assign((1.0 - self.decay) * var + self.decay * grad)


class MarkovStateModelOptimizer(optimizer_v2.OptimizerV2):
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
        super().__init__(name="MarkovStateModelOptimizer")

        # Set learning rate for this optimizer (needed to avoid and error)
        self._set_hyper("learning_rate", learning_rate)

        # Moving average optimizer for the transition probability matrix
        self.ema_optimizer = ema_optimizer

        # Optimizer for all other trainable variables
        self.base_optimizer = base_optimizer
        self.base_optimizer._set_hyper("learning_rate", self.learning_rate)

    def _create_slots(self, var_list):
        self.base_optimizer._create_slots(var_list)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        return self.base_optimizer._prepare_local(
            var_device,
            var_dtype,
            apply_state,
        )

    def _resource_apply_dense(self, grad, var, **kwargs):
        if "hid_state_inf_kernel" in var.name:
            # This is a HiddenMarkovStateInferenceLayer, use a moving
            # average to update the transition probability matrix
            updated_var = self.ema_optimizer._resource_apply_dense(grad, var)
        else:
            # This is a normal TensorFlow variable,
            # use the base optimizer to update this variable
            updated_var = self.base_optimizer._resource_apply_dense(
                grad,
                var,
                **kwargs,
            )
        return updated_var

    def _resource_apply_sparse(self, *args, **kwargs):
        return self.base_optimizer._resource_apply_sparse(*args, **kwargs)
