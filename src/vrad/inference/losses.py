"""Custom Tensorflow losses.

"""

from tensorflow import Variable
from tensorflow.keras.losses import Loss


class LogLikelihoodLoss(Loss):
    """Negative log-likelihood loss."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, ll_loss):
        """Returns the first output of the model, which is the negative log-likelihood.
        """
        return ll_loss


class KullbackLeiblerLoss(Loss):
    """Kullback-Leibler (KL) loss."""

    def __init__(self, annealing_factor, **kwargs):
        self.annealing_factor = annealing_factor
        super().__init__(**kwargs)

    def call(self, y_true, kl_loss):
        """Returns the second output of the model, which is the KL loss."""
        return self.annealing_factor * kl_loss

    def get_config(self):
        config = super().get_config()
        return {**config, "annealing_factor": self.annealing_factor}
