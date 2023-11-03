"""Custom TensorFlow regularizers.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow_probability as tfp

from osl_dynamics.inference.layers import add_epsilon

tfb = tfp.bijectors


class InverseWishart(regularizers.Regularizer):
    """Inverse Wishart regularizer.

    Parameters
    ----------
    nu : int
        Degrees of freedom. Must be greater than (n_channels - 1).
    psi : np.ndarray
        Scale matrix. Must be a symmetric positive definite matrix.
        Shape must be (n_channels, n_channels).
    epsilon : float
        Error added to the diagonal of the covariances.
    """

    def __init__(self, nu, psi, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        self.psi = psi
        self.epsilon = epsilon
        self.n_channels = psi.shape[-1]
        self.bijector = tfb.Chain(
            [tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()],
        )

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

    def __call__(self, flattened_cholesky_factors):
        covariances = add_epsilon(
            self.bijector(flattened_cholesky_factors), self.epsilon, diag=True
        )
        log_det_cov = tf.linalg.logdet(covariances)
        inv_cov = tf.linalg.inv(covariances)
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
        1D array of the mean of the prior. Shape must be (n_channels,).
    sigma : np.ndarray
        2D array of covariance matrix of the prior.
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
                "Cholesky decomposition of sigma failed. "
                + "sigma must be positive definite."
            )

        self.inv_sigma = tf.linalg.inv(self.sigma)

    def __call__(self, vectors):
        vectors = vectors - tf.expand_dims(self.mu, 0)

        reg = (1 / 2) * tf.reduce_sum(
            tf.matmul(
                tf.expand_dims(vectors, -2),
                tf.matmul(
                    tf.expand_dims(self.inv_sigma, 0),
                    tf.expand_dims(vectors, -1),
                ),
            )
        )
        return reg


class MarginalInverseWishart(regularizers.Regularizer):
    """Inverse Wishart regularizer on correlaton matrices.

    Parameters
    ----------
    nu : int
        Degrees of freedom. Must be greater than (n_channels - 1).
    epsilon : float
        Error added to the correlations.
    n_channels : int
        Number of channels of the correlation matrices.

    Note
    ----
    It is assumed that the scale matrix of the inverse Wishart distribution
    is diagonal. Hence, the marginal distribution on the correlation matrix is
    independent of the scale matrix.
    """

    def __init__(self, nu, epsilon, n_channels, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        self.epsilon = epsilon
        self.n_channels = n_channels
        self.bijector = tfb.Chain(
            [tfb.CholeskyOuterProduct(), tfb.CorrelationCholesky()]
        )

        # Validation
        if not self.nu > self.n_channels - 1:
            raise ValueError("nu must be greater than (n_channels - 1).")

    def __call__(self, flattened_cholesky_factor):
        correlations = add_epsilon(
            self.bijector(flattened_cholesky_factor), self.epsilon, diag=True
        )
        log_det_corr = tf.linalg.logdet(correlations)
        inv_corr = tf.linalg.inv(correlations)
        reg = tf.reduce_sum(
            ((self.nu + self.n_channels + 1) / 2) * log_det_corr
        ) + tf.reduce_sum((self.nu / 2) * tf.math.log(tf.linalg.diag_part(inv_corr)))
        return reg


class LogNormal(regularizers.Regularizer):
    """Log-Normal regularizer on the standard deviations.

    Parameters
    ----------
    mu : np.ndarray
        Mu parameters of the log normal distribution. Shape is (n_channels,).
    sigma : np.ndarray
        Sigma parameters of the log normal distribution.
        Shape is (n_channels,). All entries must be positive.
    epsilon : float
        Error added to the standard deviations.
    """

    def __init__(self, mu, sigma, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon
        self.bijector = tfb.Softplus()

        # Validation
        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D array.")

        if self.sigma.ndim != 1:
            raise ValueError("sigma must be a 1D array.")

        if self.mu.shape[0] != self.sigma.shape[0]:
            raise ValueError("mu and sigma must have the same length.")

        if np.any(self.sigma < 0):
            raise ValueError("Entries of sigma must be positive.")

    def __call__(self, diagonals):
        std = add_epsilon(self.bijector(diagonals), self.epsilon)
        log_std = tf.math.log(std)
        reg = tf.reduce_sum(
            log_std
            + tf.multiply(
                tf.math.square(log_std - tf.expand_dims(self.mu, 0)),
                1 / (2 * tf.math.square(tf.expand_dims(self.sigma, 0))),
            )
        )
        return reg
