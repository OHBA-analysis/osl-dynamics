"""Custom Tensorflow losses.

"""

import tensorflow as tf
from tensorflow.keras.losses import Loss


class ModelOutputLoss(Loss):
    """Class to use a model output as a loss function.

    Parameters
    ----------
    kl_annealing_factor : tf.Variable
        Factor to multiply the model output by. Optional, default is 1.0.

    Returns
    -------
    float
        Loss value.
    """

    def __init__(self, kl_annealing_factor: tf.Variable = None, **kwargs):
        if kl_annealing_factor is None:
            self.kl_annealing_factor = tf.Variable(1.0)
        else:
            self.kl_annealing_factor = kl_annealing_factor
        super().__init__(**kwargs)

    def call(self, target, model_output):
        return self.kl_annealing_factor * model_output

    def get_config(self):
        config = super().get_config()
        return {**config, "kl_annealing_factor": self.kl_annealing_factor}
