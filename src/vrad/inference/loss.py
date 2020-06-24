import functools
from typing import List

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.backend import expand_dims


def normal_kl_divergence(
    mean_a: tf.Tensor, log_std_a: tf.Tensor, mean_b: tf.Tensor, log_std_b: tf.Tensor
):
    r"""Calculate the KL divergence for two univariate normal distributions

    Taking the means and standard deviations of two univariate normal distributions

    Parameters
    ----------
    mean_a : tf.Tensor
        Mean of first distribution
    log_std_a : tf.Tensor
        Standard deviation of first distribution
    mean_b : tf.Tensor
        Mean of second distribution
    log_std_b : tf.Tensor
        Standard deviation of second distribution

    Returns
    -------
    kl_divergence : tf.Tensor
        KL divergence of univariate normals parameterised by their means
        and standard deviations

    Notes
    -----
    .. math::
        \text{KL} = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 +\
         (\mu_1 - \mu_2)^2}{2 \sigma_2^2} - \frac{1}{2}

    Examples
    --------
    >>> mean_1, mean_2 = tf.zeros(2)
    >>> log_std_1, log_std_2 = tf.zeros(2)
    >>> print(normal_kl_divergence(mean_1, mean_2, log_std_1, log_std_2))
    tf.Tensor(0.0, shape=(), dtype=float32)
    """
    std_a = tf.exp(log_std_a)
    std_b = tf.exp(log_std_b)

    term_1 = log_std_b - log_std_a

    numerator = 0.5 * (std_a ** 2 + (mean_a - mean_b) ** 2)

    term_2 = numerator / std_b ** 2

    kl_divergence = term_1 + term_2 - 0.5

    return kl_divergence


class LogLikelihoodLayer(Layer):
    """Computes log likelihood."""

    def __init__(self, n_states, n_channels, alpha_xform, **kwargs):
        super(LogLikelihoodLayer, self).__init__(**kwargs)
        self.n_states = n_states
        self.n_channels = n_channels
        self.alpha_xform = alpha_xform

    def build(self, input_shape):
        # only learn alpha_scaling if softmax is being used
        learn = self.alpha_xform == "softmax"
        self.alpha_scaling = self.add_weight(
            "alpha_scaling",
            shape=self.n_states,
            dtype=K.floatx(),
            initializer=tf.keras.initializers.Ones(),
            trainable=learn,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        """
        The Log-Likelihood is given by:
        c − (0.5*m)*log|Σ| − 0.5*∑(x - mu)^T sigma^-1 (x - mu)
        where:
        - c is some constant
        - Σ is the covariance matrix
        - mu is the mean (in this case, zero),
        - x are the observations, i.e. Y_portioned here.
        - m is the number of observations = channels

        Dimensions:
        - cov_arg_inv: [batches, mini_batch_length, n_channels, n_channels]
        - Y_portioned: [batches, mini_batch_length, n_channels]
        - alpha: [batch_size, mini_batch_length, n_states, 1, 1]
        - mean basis functions: [1, 1, n_states, n_channels]
        - covariance basis functions: [1,1, n_states, n_channels, n_channels]
        """
        n_states = self.n_states
        n_channels = self.n_channels
        Y_portioned, theta_ast, mean_matrix, covariance_matrix = inputs

        if self.alpha_xform == "softplus":
            alpha_ext = K.expand_dims(tf.keras.activations.softplus(theta_ast), axis=-1)
        elif self.alpha_xform == "softmax":
            alpha_ext = K.expand_dims(
                tf.keras.activations.softmax(theta_ast, axis=2), axis=-1
            )

        mean_ext = tf.reshape(mean_matrix, (1, 1, n_states, n_channels))

        # Do the multiplicative sum over the n_states dimension:
        mn_arg = tf.reduce_sum(tf.multiply(alpha_ext, mean_ext), 2)

        if self.alpha_xform == "softplus":
            alpha_ext = K.expand_dims(
                K.expand_dims(tf.keras.activations.softplus(theta_ast), axis=-1),
                axis=-1,
            )
        elif self.alpha_xform == "softmax":
            alpha_ext = tf.keras.activations.softmax(theta_ast, axis=2)
            alpha_ext = tf.multiply(
                alpha_ext, tf.keras.activations.softplus(self.alpha_scaling)
            )
            alpha_ext = K.expand_dims(K.expand_dims(alpha_ext, axis=-1), axis=-1)

        covariance_ext = tf.reshape(
            covariance_matrix, (1, 1, n_states, n_channels, n_channels)
        )

        # Do the multiplicative sum over the n_states dimension:
        cov_arg = tf.reduce_sum(tf.multiply(alpha_ext, covariance_ext), 2)

        # Add a tiny bit of diagonal to the covariance to ensure invertability
        cov_arg += 1e-8 * tf.eye(n_channels, n_channels)

        inv_cov_arg = tf.linalg.inv(cov_arg)
        log_det = -0.5 * tf.linalg.logdet(cov_arg)

        # Y_portioned is [batches, mini_batch_length,n_channels],
        # but we need it to be [batches, mini_batch_length,1,n_channels],
        # so that it can muliply the [n_channels x n_channels] covariance
        # as a [1 x n_channels] vector (at each tpt)
        Y_exp_dims = tf.expand_dims(Y_portioned, axis=2)
        mn_exp_dims = tf.expand_dims(mn_arg, axis=2)

        tmp = tf.subtract(Y_exp_dims, mn_exp_dims)
        attempt = -0.5 * tf.matmul(
            tf.matmul(tmp, inv_cov_arg), tf.transpose(tmp, perm=[0, 1, 3, 2])
        )
        LL = log_det + tf.squeeze(tf.squeeze(attempt, axis=3), axis=2)
        LL = -tf.reduce_sum(LL)

        return expand_dims(LL)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])

    def get_config(self):
        config = super(LogLikelihoodLayer, self).get_config()
        config.update(
            {
                "n_states": self.n_states,
                "n_channels": self.n_channels,
                "alpha_xform": self.alpha_xform,
            }
        )
        return config


class KLDivergenceLayer(Layer):
    """Computes KL Divergence."""

    def __init__(self, n_states, n_channels, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)
        self.n_states = n_states
        self.n_channels = n_channels

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        inference_mu, inference_sigma, model_mu, log_sigma_theta_j = inputs

        # model_mu needs shifting forward by one tpt
        shifted_model_mu = tf.roll(model_mu, shift=1, axis=1)

        prior = tfp.distributions.Normal(
            loc=shifted_model_mu, scale=tf.exp(log_sigma_theta_j)
        )
        posterior = tfp.distributions.Normal(
            loc=inference_mu, scale=tf.exp(inference_sigma)
        )

        kl_loss = tf.reduce_sum(tfp.distributions.kl_divergence(posterior, prior))

        return expand_dims(kl_loss)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])

    def get_config(self):
        config = super(KLDivergenceLayer, self).get_config()
        config.update(
            {"n_states": self.n_states, "n_channels": self.n_channels,}
        )
        return config
