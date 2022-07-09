"""Custom TensorFlow regularizers.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers


class InverseWishart(regularizers.Regularizer):
    """Inverse Wishart regularizer.

    Parameters
    ----------
    nu : int
        Degrees of freedom. Must be greater than (n_channels - 1).
    psi : np.ndarray
        Scale matrix. All elements must be positive.
        Shape must be (n_channels, n_channels).
    """

    def __init__(self, nu, psi, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        self.psi = psi
        self.n_channels = psi.shape[-1]

        if not self.nu > self.n_channels - 1:
            raise ValueError("nu must be greater than (n_channels - 1).")

        if np.any(self.psi < 0):
            raise ValueError("psi must be positive.")

    def __call__(self, cov):
        log_det_cov = tf.reduce_sum(tf.linalg.logdet(cov))
        inv_cov = tf.linalg.inv(cov)
        reg = ((self.nu + self.n_channels + 1) / 2) * log_det_cov + tf.reduce_sum(
            tf.multiply(
                tf.linalg.diag_part(inv_cov),
                tf.expand_dims(self.psi, axis=0),
            )
        ) / 2
        return reg


class MultivariateNormal(regularizers.Regularizer):
    """Multivariate normal regularizer.

    Parameters
    ----------
    sigma : np.ndarray
        1D numpy array of variances for each channel.
        Shape must be (n_channels,).
    """

    def __init__(self, sigma, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.regularization_strength = 1 / (2 * sigma)

    def __call__(self, vectors):
        reg = tf.reduce_sum(
            tf.multiply(
                tf.math.square(vectors),
                tf.expand_dims(self.regularization_strength, axis=0),
            )
        )
        return reg
