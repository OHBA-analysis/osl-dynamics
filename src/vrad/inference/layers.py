from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.backend import expand_dims, stop_gradient
from vrad.inference.functions import (
    cholesky_factor,
    cholesky_factor_to_full_matrix,
    is_symmetric,
)
from vrad.inference.initializers import (
    CholeskyCovariancesInitializer,
    Identity3D,
    MeansInitializer,
)


class ReparameterizationLayer(Layer):
    """Resample a normal distribution.

    The reparameterization trick is used to provide a differentiable random sample.
    By optimizing the values which define the distribution (i.e. mu and sigma), a model
    can be trained while still maintaining its stochasticity.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        chans = K.int_shape(z_mean)[2]
        epsilon = K.random_normal(shape=(batch, dim, chans))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape


class MultivariateNormalLayer(Layer):
    """Parameterises multiple multivariate Gaussians and in terms of their means
       and covariances. Means and covariances are outputted.
    """

    def __init__(
        self,
        n_gaussians,
        dim,
        learn_means=True,
        learn_covariances=True,
        initial_means=None,
        initial_covariances=None,
        **kwargs
    ):
        super(MultivariateNormalLayer, self).__init__(**kwargs)
        self.n_gaussians = n_gaussians
        self.dim = dim
        self.learn_means = learn_means
        self.learn_covariances = learn_covariances
        self.initial_means = initial_means
        self.burnin = tf.Variable(False, trainable=False)

        if initial_covariances is not None:
            # If the matrix is symmetric we assume it's the full covariance matrix
            # WARNING: diagonal matrices are assumed to be the full covariance matrix
            if is_symmetric(initial_covariances):
                self.initial_cholesky_covariances = cholesky_factor(initial_covariances)

            # Otherwise, we assume the cholesky factor has already been calculated
            else:
                self.initial_cholesky_covariances = initial_covariances

        else:
            self.initial_cholesky_covariances = None

    def build(self, input_shape):
        # Initializer for means
        if self.initial_means is None:
            self.means_initializer = tf.keras.initializers.Zeros
        else:
            self.means_initializer = MeansInitializer(self.initial_means)

        # Create weights the means
        self.means = self.add_weight(
            "means",
            shape=(self.n_gaussians, self.dim),
            dtype=tf.float32,
            initializer=self.means_initializer,
            trainable=self.learn_means,
        )

        # Initializer for covariances
        if self.initial_cholesky_covariances is None:
            self.cholesky_covariances_initializer = Identity3D
        else:
            self.cholesky_covariances_initializer = CholeskyCovariancesInitializer(
                self.initial_cholesky_covariances
            )

        # Create weights for the cholesky decomposition of the covariances
        self.cholesky_covariances = self.add_weight(
            "cholesky_covariances",
            shape=(self.n_gaussians, self.dim, self.dim),
            dtype=tf.float32,
            initializer=self.cholesky_covariances_initializer,
            trainable=self.learn_covariances,
        )

        self.built = True

    def call(self, inputs, **kwargs):
        def no_grad():
            return stop_gradient(
                cholesky_factor_to_full_matrix(self.cholesky_covariances)
            )

        def with_grad():
            return cholesky_factor_to_full_matrix(self.cholesky_covariances)

        # If we are in the burn-in phase call no_grad, otherwise call with_grad
        self.covariances = tf.cond(self.burnin, no_grad, with_grad,)

        return self.means, self.covariances

    def set_means(self, means):
        """Overrides the means weights"""
        weights = self.get_weights()
        weights[0] = means
        self.set_weight(weights)

    def set_covariances(self, covariances):
        """Overrides the covariances weights"""
        # Calculate the cholesky factor
        cholesky_covariances = cholesky_factor(covariances)

        # Set layer weights
        weights = self.get_weights()
        weights[1] = cholesky_covariances
        self.set_weights(weights)

    def get_means(self):
        """Returns the means"""
        return self.means

    def get_covariances(self):
        """Returns the covariances"""
        covariances = cholesky_factor_to_full_matrix(self.cholesky_covariances)
        return covariances

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape([self.n_gaussians, self.dim]),
            tf.TensorShape([self.n_gaussians, self.dim, self.dim]),
        ]

    def get_config(self):
        config = super(MultivariateNormalLayer, self).get_config()
        config.update(
            {
                "n_gaussian": self.n_gaussians,
                "dim": self.dim,
                "learn_means": self.learn_means,
                "learn_covariances": self.learn_covariances,
                "initial_means": self.initial_means,
                "initial_cholesky_covariances": self.initial_cholesky_covariances,
            }
        )
        return config


class TrainableVariablesLayer(Layer):
    """Generic trainable variables layer.

       Sets up trainable parameters/weights tensor of a certain shape.
       Parameters/weights are outputted.
    """

    def __init__(self, shape, initial_values=None, trainable=True, **kwargs):
        super(TrainableVariablesLayer, self).__init__(**kwargs)
        self.shape = shape
        self.initial_values = initial_values
        self.trainable = trainable

    def _variables_initializer(self, shape, dtype=tf.float32):
        values = self.initial_values
        return values

    def build(self, input_shape):
        # Set initialiser for means
        if self.initial_values is None:
            self.values_initializer = tf.keras.initializers.Zeros
        else:
            self.values_initializer = self._variables_initializer

        self.values = self.add_weight(
            "values",
            shape=self.shape,
            dtype=K.floatx(),
            initializer=self.values_initializer,
            trainable=self.trainable,
        )

        self.built = True

    def call(self, inputs, **kwargs):
        return self.values

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.shape)

    def get_config(self):
        config = super(TrainableVariablesLayer, self).get_config()
        config.update(
            {"shape": self.shape, "trainable": self.trainable,}
        )
        return config


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
