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
    n_batches : int
        Number of batches in the data.
    """

    def __init__(self, nu, psi, n_batches, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        self.psi = psi
        self.n_batches = n_batches
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
    n_batches : int
        Number of batches in the data.
    """

    def __init__(self, mu, sigma, n_batches, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma
        self.n_batches = n_batches

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


class MarginalInverseWishart(regularizers.Regularizer):
    """Inverse Wishart regularizer on correlaton matrices.

    It is assumed that the scale matrix of the inverse Wishart distribution
    is diagonal. Hence the marginal distribution on the correlation matrix is
    independent of the scale matrix.

    Parameters
    ----------
    nu : int
        Degrees of freedom. Must be greater than (n_channels - 1).
    n_channels : int
        Number of channels of the correlation matrices.
    n_batches : int
        Number of batches in the data.
    """

    def __init__(self, nu, n_channels, n_batches, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        self.n_channels = n_channels
        self.n_batches = n_batches
        # Validation
        if not self.nu > self.n_channels - 1:
            raise ValueError("nu must be greater than (n_channels - 1).")

    def __call__(self, corr):
        log_det_corr = tf.linalg.logdet(corr)
        inv_corr = tf.linalg.inv(corr)
        reg = tf.reduce_sum(
            ((self.nu + self.n_channels + 1) / 2) * log_det_corr
        ) + tf.reduce_sum((self.nu / 2) * tf.math.log(tf.linalg.diag_part(inv_corr)))
        return reg


class LogNormal(regularizers.Regularizer):
    """Log normal regularizer on the standard deviations.

    Parameters
    ----------
    mu : np.ndarray
        Mu parameters of the log normal distribution.
        Shape is (n_channels,).
    sigma : np.ndarray
        Sigma parameters of the log normal distribution.
        Shape is (n_channels,). All entries must be positive.
    n_batches : int
        Number of batches in the data.
    """

    def __init__(self, mu, sigma, n_batches, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma
        self.n_batches = n_batches

        # Validation
        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D array.")

        if self.sigma.ndim != 1:
            raise ValueError("sigma must be a 1D array.")

        if self.mu.shape[0] != self.sigma.shape[0]:
            raise ValueError("mu and sigma must have the same length.")

        if np.any(self.sigma < 0):
            raise ValueError("Entries of sigma must be positive.")

    def __call__(self, std):
        log_std = tf.math.log(std)
        reg = tf.reduce_sum(
            log_std
            + tf.multiply(
                tf.math.square(log_std - tf.expand_dims(self.mu, 0)),
                1 / (2 * tf.math.square(tf.expand_dims(self.sigma, 0))),
            )
        )
        return reg
