"""Custom Tensorflow losses.

"""

from tensorflow.keras.losses import Loss


class LogLikelihoodLoss(Loss):
    """Negative log-likelihood loss."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, ll_loss):
        """Negative log-likelihood loss.

        The negative log-likelihood is the first output of the model.

        Returns
        -------
        float
            Negative log-likelihood.
        """
        return ll_loss


class KullbackLeiblerLoss(Loss):
    """Kullback-Leibler (KL) loss."""

    def __init__(self, annealing_factor, **kwargs):
        self.annealing_factor = annealing_factor
        super().__init__(**kwargs)

    def call(self, y_true, kl_loss):
        """KL divergence loss.
        
        The KL divergence is the second output of the model.

        Returns
        -------
        float
            KL divergence.
        """
        return self.annealing_factor * kl_loss

    def get_config(self):
        config = super().get_config()
        return {**config, "annealing_factor": self.annealing_factor}
