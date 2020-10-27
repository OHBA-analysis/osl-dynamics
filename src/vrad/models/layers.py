"""Tensorflow layers used in the inference and generative model.

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import activations, layers
from tensorflow.python.keras.backend import stop_gradient
from vrad.inference.functions import (
    cholesky_factor,
    cholesky_factor_to_full_matrix,
    is_symmetric,
    trace_normalize,
)
from vrad.inference.initializers import (
    CholeskyCovariancesInitializer,
    Identity3D,
    MeansInitializer,
)


class DummyLayer(layers.Layer):
    """Dummy layer.

    Returns the inputs without modification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class TrainableVariablesLayer(layers.Layer):
    """Generic trainable variables layer.

    Sets up trainable parameters/weights tensor of a certain shape.
    Parameters/weights are outputted.
    """

    def __init__(self, n_units, activation=None, initial_values=None, **kwargs):
        super().__init__(**kwargs)
        self.n_units = n_units
        self.initial_values = initial_values
        self.activation = activations.get(activation)

    def build(self, input_shape):

        # If no initial values have been passed, initialise with zeros
        if self.initial_values is None:
            self.values_initializer = tf.keras.initializers.Zeros()

        # Otherwise, initialise with the variables passed
        else:

            def variables_initializer(shape, dtype=None):
                return self.initial_values

            self.values_initializer = variables_initializer

        # Create trainable weights
        self.values = self.add_weight(
            "values",
            shape=(self.n_units,),
            dtype=K.floatx(),
            initializer=self.values_initializer,
            trainable=True,
        )

        self.built = True

    def call(self, inputs, **kwargs):
        return self.activation(self.values)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_units": self.n_units,
                "activation": activations.serialize(self.activation),
            }
        )
        return config


class SampleNormalDistributionLayer(layers.Layer):
    """Layer for sampling a normal distribution.

    This layer accepts the mean and (log of) the standard deviation and
    outputs samples from a normal distribution.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        mu, log_sigma = inputs
        if training:
            N = tfp.distributions.Normal(loc=mu, scale=tf.exp(log_sigma))
            return N.sample()
        else:
            return mu

    def compute_output_shape(self, input_shape):
        mu_shape, log_sigma_shape = input_shape
        return mu_shape


class StateMixingFactorsLayer(layers.Layer):
    """Layer for calculating the mixing ratio of the states.

    This layer accepts the logits theta_t and outputs alpha_t.
    """

    def __init__(self, alpha_xform, lasso_alpha_regularization, **kwargs):
        super().__init__(**kwargs)
        self.alpha_xform = alpha_xform
        self.lasso_alpha_regularization = lasso_alpha_regularization

    def call(self, theta_t, **kwargs):

        # Calculate alpha_t from theta_t
        if self.alpha_xform == "softplus":
            alpha_t = activations.softplus(theta_t)
        elif self.alpha_xform == "relu":
            alpha_t = activations.relu(theta_t)
        elif self.alpha_xform == "softmax":
            alpha_t = activations.softmax(theta_t, axis=2)
        elif self.alpha_xform == "categorical":
            gumbel_softmax_distribution = tfp.distributions.RelaxedOneHotCategorical(
                temperature=0.5, probs=activations.softmax(theta_t, axis=2)
            )
            alpha_t = gumbel_softmax_distribution.sample()

        # Regularisation on alpha_t
        if self.lasso_alpha_regularization:
            self.add_loss(tf.reduce_sum(alpha_t))

        return alpha_t

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha_xform": self.alpha_xform,
                "lasso_alpha_regularization": self.lasso_alpha_regularization,
            }
        )
        return config


class MeansCovsLayer(layers.Layer):
    """Layer to learn the mean and covariance of each state.

    Outputs the mean vector and covariance matrix of each state.
    """

    def __init__(
        self,
        n_gaussians,
        dim,
        learn_means,
        learn_covariances,
        normalize_covariances,
        initial_means,
        initial_covariances,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_gaussians = n_gaussians
        self.dim = dim
        self.learn_means = learn_means
        self.learn_covariances = learn_covariances
        self.normalize_covariances = normalize_covariances
        self.initial_means = initial_means

        if initial_covariances is not None:
            # Normalise the covariances if required
            if normalize_covariances:
                initial_covariances = trace_normalize(initial_covariances)

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
        # Calculate the covariance matrix from the cholesky factor
        self.covariances = cholesky_factor_to_full_matrix(self.cholesky_covariances)

        # Normalise the covariance
        if self.normalize_covariances:
            self.covariances = trace_normalize(self.covariances)

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
                "normalize_covariances": self.normalize_covariances,
            }
        )
        return config


class MixMeansCovsLayer(layers.Layer):
    """Computes a probabilistic mixture of means and covariances."""

    def __init__(self, n_states, n_channels, learn_alpha_scaling, **kwargs):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.n_channels = n_channels
        self.learn_alpha_scaling = learn_alpha_scaling

    def build(self, input_shape):
        # Initialise such that softplus(alpha_scaling) = 1
        self.alpha_scaling_initializer = tf.keras.initializers.Constant(
            np.log(np.exp(1.0) - 1.0)
        )
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
        # - alpha_t.shape = (None, sequence_length, n_states)
        # - mu.shape      = (n_states, n_channels)
        # - D.shape       = (n_states, n_channels, n_channels)
        alpha_t, mu, D = inputs

        # Rescale the state mixing factors
        alpha_t = tf.multiply(alpha_t, activations.softplus(self.alpha_scaling))

        # Reshape alpha_t and mu for multiplication
        alpha_t = K.expand_dims(alpha_t, axis=-1)
        mu = tf.reshape(mu, (1, 1, self.n_states, self.n_channels))

        # Calculate the mean: m_t = Sum_j alpha_jt mu_j
        m_t = tf.reduce_sum(tf.multiply(alpha_t, mu), axis=2)

        # Reshape alpha_t and D for multiplication
        alpha_t = K.expand_dims(alpha_t, axis=-1)
        D = tf.reshape(D, (1, 1, self.n_states, self.n_channels, self.n_channels))

        # Calculate the covariance: C_t = Sum_j alpha_jt D_j
        C_t = tf.reduce_sum(tf.multiply(alpha_t, D), axis=2)

        return [m_t, C_t]

    def compute_output_shape(self, input_shape):
        alpha_t_shape, mu_shape, D_shape = input_shape
        return [mu_shape, D_shape]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_states": self.n_states,
                "n_channels": self.n_channels,
                "learn_alpha_scaling": self.learn_alpha_scaling,
            }
        )
        return config


class LogLikelihoodLayer(layers.Layer):
    """Computes the negative log likelihood."""

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


class KLDivergenceLayer(layers.Layer):
    """Computes KL Divergence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        inference_mu, inference_log_sigma, model_mu, model_log_sigma = inputs

        # model_mu needs be shifted one time point to the right
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


class InferenceRNNLayers(layers.Layer):
    """RNN layers for the inference network."""

    def __init__(
        self, n_layers, n_units, dropout_rate, NormalizationLayer=None, **kwargs
    ):
        super().__init__(**kwargs)
        if NormalizationLayer is None:
            NormalizationLayer = layers.LayerNormalization
        self.n_units = n_units
        self.layers = []
        for n in range(n_layers):
            self.layers.append(
                layers.Bidirectional(
                    layer=layers.LSTM(n_units, return_sequences=True, stateful=False)
                )
            )
            self.layers.append(NormalizationLayer())
            self.layers.append(layers.Dropout(dropout_rate))

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
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

    def __init__(
        self, n_layers, n_units, dropout_rate, NormalizationLayer=None, **kwargs
    ):
        super().__init__(**kwargs)
        if NormalizationLayer is None:
            NormalizationLayer = layers.LayerNormalization
        self.n_units = n_units
        self.layers = []
        for n in range(n_layers):
            self.layers.append(
                layers.LSTM(n_units, return_sequences=True, stateful=False)
            )
            self.layers.append(NormalizationLayer())
            self.layers.append(layers.Dropout(dropout_rate))

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape.as_list()[:-1] + [self.n_units])

    def get_config(self):
        config = super().get_config()
        config.update({"n_units": self.n_units})
        return config
