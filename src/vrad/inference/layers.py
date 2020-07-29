"""Tensorflow layers used in the inference and generative model.

"""

from typing import Union

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax, softplus
from tensorflow.python.keras.backend import stop_gradient
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


class ReparameterizationLayer(layers.Layer):
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
        z_mean_shape, z_log_var_shape = input_shape
        return z_mean_shape

    def get_config(self):
        config = super().get_config()
        return config


class MultivariateNormalLayer(layers.Layer):
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
        super().__init__(**kwargs)
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
            self.means_initializer = tf.keras.initializers.Zeros()
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
            self.cholesky_covariances_initializer = Identity3D()
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
        self.covariances = tf.cond(self.burnin, no_grad, with_grad)

        return [self.means, self.covariances]

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape([self.n_gaussians, self.dim]),
            tf.TensorShape([self.n_gaussians, self.dim, self.dim]),
        ]

    def get_config(self):
        config = super().get_config()
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


class TrainableVariablesLayer(layers.Layer):
    """Generic trainable variables layer.

       Sets up trainable parameters/weights tensor of a certain shape.
       Parameters/weights are outputted.
    """

    def __init__(self, shape, initial_values=None, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.initial_values = initial_values
        self.trainable = trainable

    def build(self, input_shape):

        # If no initial values have been passed, initialise with zeros
        if self.initial_values is None:
            self.values_initializer = tf.keras.initializers.Zeros()

        # Otherwise, initialise with the variables passed
        else:

            def variables_initializer(shape, dtype=None):
                return self.initial_values

            self.values_initializer = variables_initializer

        # Create traininable weights
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
        config = super().get_config()
        config.update({"shape": self.shape, "trainable": self.trainable})
        return config


class MixMeansCovsLayer(layers.Layer):
    """Computes a probabilistic mixture of means and covariances."""

    def __init__(
        self, n_states, n_channels, alpha_xform, learn_alpha_scaling, **kwargs
    ):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.n_channels = n_channels
        self.alpha_xform = alpha_xform
        self.learn_alpha_scaling = learn_alpha_scaling

    def build(self, input_shape):
        self.alpha_scaling_initializer = tf.keras.initializers.Ones()
        self.alpha_scaling = self.add_weight(
            "alpha_scaling",
            shape=self.n_states,
            dtype=K.floatx(),
            initializer=self.alpha_scaling_initializer,
            trainable=self.learn_alpha_scaling,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        """Computes m_t = Sum_j alpha_jt mu_j and C_t = Sum_j alpha_jt D_j."""
        # Unpack the inputs:
        # - theta_t.shape = (None, sequence_length, n_states)
        # - mu.shape      = (n_states, n_channels)
        # - D.shape       = (n_states, n_channels, n_channels)
        theta_t, mu, D = inputs

        if self.alpha_xform == "softplus":
            alpha_t = softplus(theta_t)
        elif self.alpha_xform == "softmax":
            alpha_t = softmax(theta_t, axis=2)
            alpha_t = tf.multiply(alpha_t, softplus(self.alpha_scaling))

        # Reshape alpha_t and mu for multiplication
        alpha_t = K.expand_dims(alpha_t, axis=-1)
        mu = tf.reshape(mu, (1, 1, self.n_states, self.n_channels))

        # Calculate the mean: m_t = Sum_j alpha_jt mu_j
        m_t = tf.reduce_sum(tf.multiply(alpha_t, mu), 2)

        # Reshape alpha_t and D for multiplication
        alpha_t = K.expand_dims(alpha_t, axis=-1)
        D = tf.reshape(D, (1, 1, self.n_states, self.n_channels, self.n_channels))

        # Calculate the covariance: C_t = Sum_j alpha_jt D_j
        C_t = tf.reduce_sum(tf.multiply(alpha_t, D), 2)

        return [m_t, C_t]

    def compute_output_shape(self, input_shape):
        theta_t_shape, mu_shape, D_shape = input_shape
        return [mu_shape, D_shape]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_states": self.n_states,
                "n_channels": self.n_channels,
                "alpha_xform": self.alpha_xform,
                "learn_alpha_scaling": self.learn_alpha_scaling,
            }
        )
        return config


class LogLikelihoodLayer(layers.Layer):
    """Computes log likelihood."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        """Computes the log likelihood:
           c - 0.5 * log(|sigma|) - 0.5 * [(x - mu)^T sigma^-1 (x - mu)]
           where:
           - x is a single observation
           - mu is the mean vector
           - sigma is the covariance matrix
           - c is a constant
           This method returns the negative of the log likelihood.
        """
        x, mu, sigma = inputs

        x = tf.expand_dims(x, axis=2)
        mu = tf.expand_dims(mu, axis=2)

        # Calculate second term: -0.5 * log(|sigma|)
        second_term = -0.5 * tf.linalg.logdet(sigma)

        # Calculate third term: -0.5 * [(x - mu)^T sigma^-1 (x - mu)]
        inv_sigma = tf.linalg.inv(sigma + 1e-8 * tf.eye(sigma.shape[-1]))
        x_minus_mu = tf.subtract(x, mu)
        x_minus_mu_T = tf.transpose(x_minus_mu, perm=[0, 1, 3, 2])
        third_term = -0.5 * tf.matmul(tf.matmul(x_minus_mu, inv_sigma), x_minus_mu_T)
        third_term = tf.squeeze(tf.squeeze(third_term, axis=3), axis=2)

        # Calculate the log likelihood
        # We ignore the first term which is a constant
        ll_loss = tf.reduce_sum(second_term + third_term)

        # We return the negative of the log likelihood
        nll_loss = -ll_loss

        return K.expand_dims(nll_loss)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])

    def get_config(self):
        config = super().get_config()
        return config


class KLDivergenceLayer(layers.Layer):
    """Computes KL Divergence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        inference_mu, inference_log_sigma, model_mu, model_log_sigma = inputs

        # model_mu needs shifting forward by one time point
        shifted_model_mu = tf.roll(model_mu, shift=1, axis=1)

        prior = tfp.distributions.Normal(
            loc=shifted_model_mu, scale=tf.exp(model_log_sigma)
        )

        posterior = tfp.distributions.Normal(
            loc=inference_mu, scale=tf.exp(inference_log_sigma)
        )

        kl_loss = tf.reduce_sum(tfp.distributions.kl_divergence(posterior, prior))

        return K.expand_dims(kl_loss)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])

    def get_config(self):
        config = super().get_config()
        return config


class InferenceRNNLayers(layers.Layer):
    """RNN layers for the inference network."""

    def __init__(self, n_layers, n_units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.n_units = n_units
        self.layers = []
        for n in range(n_layers):
            self.layers.append(
                layers.Bidirectional(
                    layer=layers.LSTM(n_units, return_sequences=True, stateful=False)
                )
            )
            self.layers.append(layers.LayerNormalization())
            self.layers.append(layers.Dropout(dropout_rate))

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        # we multiply self.n_units by 2 because we're using a bidirectional RNN
        return tf.TensorShape(input_shape.as_list()[:-1] + [2 * self.n_units])

    def get_config(self):
        config = super().get_config()
        config.update({"n_units": self.n_units})
        return config


class ModelRNNLayers(layers.Layer):
    """RNN layers for the generative model."""

    def __init__(self, n_layers, n_units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.n_units = n_units
        self.layers = []
        for n in range(n_layers):
            self.layers.append(
                layers.LSTM(n_units, return_sequences=True, stateful=False)
            )
            self.layers.append(layers.LayerNormalization())
            self.layers.append(layers.Dropout(dropout_rate))

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape.as_list()[:-1] + [self.n_units])

    def get_config(self):
        config = super().get_config()
        config.update({"n_units": self.n_units})
        return config
