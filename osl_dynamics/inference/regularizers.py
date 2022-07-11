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
        Scale matrix. Must be a symmetric positive definite matrix.
        Shape must be (n_channels, n_channels).
    """

    def __init__(self, nu, psi, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        self.psi = psi
        self.n_channels = psi.shape[-1]

        # Validation
        if not self.nu > self.n_channels - 1:
            raise ValueError("nu must be greater than (n_channels - 1).")

        if self.psi.ndim != 2:
            raise ValueError("psi must be a 2D array.")

        if not np.allclose(self.psi, self.psi.T):
            raise ValueError("psi must be symmetric.")

        try:
            np.linalg.cholesky(self.psi)
        except:
            raise ValueError(
                "Cholesky decomposition of psi failed. psi must be positive definite."
            )

    def __call__(self, cov):
        log_det_cov = tf.linalg.logdet(cov)
        inv_cov = tf.linalg.inv(cov)
        reg = tf.reduce_sum(
            ((self.nu + self.n_channels + 1) / 2) * log_det_cov
            + (1 / 2) * tf.linalg.trace(tf.matmul(tf.expand_dims(self.psi, 0), inv_cov))
        )
        return reg


class MultivariateNormal(regularizers.Regularizer):
    """Multivariate normal regularizer.

    Parameters
    ----------
    mu : np.ndarray
        1D numpy array of the mean of the prior.
        Shape must be (n_channels,).
    sigma : np.ndarray
        2D numpy array of covariance matrix of the prior.
        Shape must be (n_channels, n_channels).
    """

    def __init__(self, mu, sigma, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma

        # Validation
        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D array.")

        if self.sigma.ndim != 2:
            raise ValueError("sigma must be a 2D array.")

        if not np.allclose(self.sigma, self.sigma.T):
            raise ValueError("sigma must be symmetric.")

        try:
            np.linalg.cholesky(self.sigma)
        except:
            raise ValueError(
                "Cholesky decomposition of sigma failed. sigma must be positive definite."
            )

        self.inv_sigma = tf.linalg.inv(self.sigma)

    def __call__(self, vectors):
        vectors = vectors - tf.expand_dims(self.mu, 0)

        reg = (1 / 2) * tf.reduce_sum(
            tf.matmul(
                tf.expand_dims(vectors, -2),
                tf.matmul(
                    tf.expand_dims(self.inv_sigma, 0), tf.expand_dims(vectors, -1)
                ),
            )
        )
        return reg
