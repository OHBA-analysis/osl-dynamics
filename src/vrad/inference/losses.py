"""Custom Tensorflow losses.

"""

import tensorflow as tf
from tensorflow.keras.losses import Loss


class ModelOutputLoss(Loss):
    """Class to use a model output as a loss function.

    Parameters
    ----------
    annealing_factor : tf.Variable
        Factor to multiply the model output by. Optional, default is 1.0.

    Returns
    -------
    float
        Loss value.
    """

    def __init__(self, annealing_factor: tf.Variable = None, **kwargs):
        if annealing_factor is None:
            self.annealing_factor = tf.Variable(1.0)
        else:
            self.annealing_factor = annealing_factor
        super().__init__(**kwargs)

    def call(self, target, model_output):
        return self.annealing_factor * model_output

    def get_config(self):
        config = super().get_config()
        return {**config, "annealing_factor": self.annealing_factor}
