"""Custom Tensorflow layers used in the inference network and generative model.

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import activations, constraints, layers
from vrad.inference.functions import (
    cholesky_factor,
    cholesky_factor_to_full_matrix,
    is_symmetric,
    trace_normalize,
)
from vrad.inference.initializers import WeightInitializer


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

    Setup trainable parameter/weight tensors of a certain shape.
    Parameters/weights are outputted by the layer.

    Parameters
    ----------
    n_units : int
        Number of units/neurons in the layer.
    activation : str
        Activation function to apply to the output of the layer.
    initial_values : np.ndarray
        Initial values for the trainable parameters/weights.
    """

    def __init__(
        self,
        n_units: int,
        activation: str = None,
        initial_values: np.ndarray = None,
        **kwargs
    ):
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
            dtype=tf.float32,
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
    """Layer for sampling from a normal distribution.

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


class SampleDirichletDistributionLayer(layers.Layer):
    """Layer for sampling from a Dirichlet distribution.

    This layer accepts the parameters of a Dirichlet distribution and
    outputs a sample.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):

        # Parameters of the Dirichlet distribution must be positive
        concentration = activations.softplus(inputs)

        if training:
            D = tfp.distributions.Dirichlet(concentration)
            return D.sample()
        else:
            sum_concentration = tf.reduce_sum(concentration, axis=-1)
            sum_concentration = tf.expand_dims(sum_concentration, axis=-1)
            return tf.divide(concentration, sum_concentration)

    def compute_output_shape(self, input_shape):
        return input_shape


class StateMixingFactorLayer(layers.Layer):
    """Layer for calculating the mixing ratio of the states.

    This layer accepts the logits theta_t and outputs alpha_t.

    Parameters
    ----------
    alpha_xform : str
        The functional form used to convert from theta_t to alpha_t.
    alpha_temperature : float
        Temperature parameter for the softmax or Gumbel-Softmax.
    """

    def __init__(self, alpha_xform: str, alpha_temperature: float, **kwargs):
        super().__init__(**kwargs)
        self.alpha_xform = alpha_xform
        self.alpha_temperature = alpha_temperature

    def call(self, theta_t, **kwargs):

        # Calculate alpha_t from theta_t
        if self.alpha_xform == "softplus":
            alpha_t = activations.softplus(theta_t)
        elif self.alpha_xform == "relu":
            alpha_t = activations.relu(theta_t)
        elif self.alpha_xform == "softmax":
            alpha_t = activations.softmax(theta_t / self.alpha_temperature, axis=2)
        elif self.alpha_xform == "gumbel-softmax":
            gumbel_softmax_distribution = tfp.distributions.RelaxedOneHotCategorical(
                temperature=self.alpha_temperature,
                logits=theta_t,
            )
            alpha_t = gumbel_softmax_distribution.sample()

        return alpha_t

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha_xform": self.alpha_xform,
                "alpha_temperature": self.alpha_temperature,
            }
        )
        return config


class MeansCovsLayer(layers.Layer):
    """Layer to learn the mean and covariance of each state.

    Outputs the mean vector and covariance matrix of each state.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    learn_means : bool
        Should we learn the means?
    learn_covariances : bool
        Should we learn the covariances?
    initial_means : np.ndarray
        Initial values for the mean of each state.
    initial_covariances : np.ndarray
        Initial values for the covariance of each state. Must be dtype float32
    """

    def __init__(
        self,
        n_states: int,
        n_channels: int,
        learn_means: bool,
        learn_covariances: bool,
        normalize_covariances: bool,
        initial_means: np.ndarray,
        initial_covariances: np.ndarray,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.n_channels = n_channels
        self.learn_means = learn_means
        self.learn_covariances = learn_covariances
        self.normalize_covariances = normalize_covariances

        # Initialisation of means
        if initial_means is None:
            self.initial_means = np.zeros([n_states, n_channels], dtype=np.float32)
        else:
            self.initial_means = initial_means

        self.means_initializer = WeightInitializer(self.initial_means)

        # Initialisation of covariances
        if initial_covariances is None:
            self.initial_covariances = np.stack(
                [np.eye(n_channels, dtype=np.float32)] * n_states
            )
            self.initial_cholesky_covariances = cholesky_factor(
                self.initial_covariances
            )
        else:
            # Ensure data is float32
            initial_covariances = initial_covariances.astype("float32")

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

        self.flattened_cholesky_covariances_initializer = WeightInitializer(
            tfp.math.fill_triangular_inverse(self.initial_cholesky_covariances)
        )

    def build(self, input_shape):

        # Create weights the means
        self.means = self.add_weight(
            "means",
            shape=(self.n_states, self.n_channels),
            dtype=tf.float32,
            initializer=self.means_initializer,
            trainable=self.learn_means,
        )

        # Create weights for the cholesky decomposition of the covariances
        self.flattened_cholesky_covariances = self.add_weight(
            "flattened_cholesky_covariances",
            shape=(self.n_states, self.n_channels * (self.n_channels + 1) // 2),
            dtype=tf.float32,
            initializer=self.flattened_cholesky_covariances_initializer,
            trainable=self.learn_covariances,
        )

        self.built = True

    def call(self, inputs, **kwargs):
        # Calculate the covariance matrix from the cholesky factor
        cholesky_covariances = tfp.math.fill_triangular(
            self.flattened_cholesky_covariances
        )
        self.covariances = cholesky_factor_to_full_matrix(cholesky_covariances)

        # Normalise the covariance
        if self.normalize_covariances:
            self.covariances = trace_normalize(self.covariances)

        return [self.means, self.covariances]

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape([self.n_states, self.n_channels]),
            tf.TensorShape([self.n_states, self.n_channels, self.n_channels]),
        ]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_states": self.n_states,
                "n_channels": self.n_channels,
                "learn_means": self.learn_means,
                "learn_covariances": self.learn_covariances,
                "normalize_covariances": self.normalize_covariances,
            }
        )
        return config


class MixMeansCovsLayer(layers.Layer):
    """Compute a probabilistic mixture of means and covariances.

    The mixture is calculated as  m_t = Sum_j alpha_jt mu_j and
    C_t = Sum_j alpha_jt D_j.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    learn_alpha_scaling : bool
        Should we learn an alpha scaling?
    """

    def __init__(
        self, n_states: int, n_channels: int, learn_alpha_scaling: bool, **kwargs
    ):
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
            dtype=tf.float32,
            initializer=self.alpha_scaling_initializer,
            trainable=self.learn_alpha_scaling,
        )
        self.built = True

    def call(self, inputs, **kwargs):

        # Unpack the inputs:
        # - alpha_t.shape = (None, sequence_length, n_states)
        # - mu.shape      = (n_states, n_channels)
        # - D.shape       = (n_states, n_channels, n_channels)
        alpha_t, mu, D = inputs

        # Rescale the state mixing factors
        alpha_t = tf.multiply(alpha_t, activations.softplus(self.alpha_scaling))

        # Reshape alpha_t and mu for multiplication
        alpha_t = tf.expand_dims(alpha_t, axis=-1)
        mu = tf.reshape(mu, (1, 1, self.n_states, self.n_channels))

        # Calculate the mean: m_t = Sum_j alpha_jt mu_j
        m_t = tf.reduce_sum(tf.multiply(alpha_t, mu), axis=2)

        # Reshape alpha_t and D for multiplication
        alpha_t = tf.expand_dims(alpha_t, axis=-1)
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
    """Layer to calculate the negative log likelihood.

    The negative log-likelihood is calculated assuming a multivariate normal
    probability density.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        x, mu, sigma = inputs

        # Calculate the log-likelihood
        mvn = tfp.distributions.MultivariateNormalTriL(
            loc=mu,
            scale_tril=tf.linalg.cholesky(sigma + 1e-6 * tf.eye(sigma.shape[-1])),
        )
        ll_loss = mvn.log_prob(x)

        # Sum over time dimension and average over the batch dimension
        ll_loss = tf.reduce_sum(ll_loss, axis=1)
        ll_loss = tf.reduce_mean(ll_loss, axis=0)

        # We return the negative of the log likelihood
        nll_loss = -ll_loss

        return tf.expand_dims(nll_loss, axis=-1)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])


class NormalKLDivergenceLayer(layers.Layer):
    """Layer to calculate a KL divergence between two Normal distributions.

    The KL divergence between the posterior and prior is calculated.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        inference_mu, inference_log_sigma, model_mu, model_log_sigma = inputs

        # The Model RNN predicts one time step into the future compared to the
        # inference RNN. We clip its last value, and first value of the inference RNN.
        model_mu = model_mu[:, :-1]
        model_sigma = tf.exp(model_log_sigma)[:, :-1]

        inference_mu = inference_mu[:, 1:]
        inference_sigma = tf.exp(inference_log_sigma)[:, 1:]

        # Calculate the KL divergence between the posterior and prior
        prior = tfp.distributions.Normal(loc=model_mu, scale=model_sigma)
        posterior = tfp.distributions.Normal(loc=inference_mu, scale=inference_sigma)
        kl_loss = tfp.distributions.kl_divergence(posterior, prior)

        # Sum the KL loss for each state and time point and average over batches
        kl_loss = tf.reduce_sum(kl_loss, axis=2)
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss = tf.reduce_mean(kl_loss, axis=0)

        return tf.expand_dims(kl_loss, axis=-1)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])


class DirichletKLDivergenceLayer(layers.Layer):
    """Layer to calculate a KL divergence between two Dirichlet distributions.

    The KL divergence between the posterior and prior is calculated.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        inference_theta, model_theta = inputs

        # The Model RNN predicts one time step into the future compared to the
        # inference RNN. We clip its last value, and first value of the inference RNN.
        model_theta = model_theta[:, :-1]
        inference_theta = inference_theta[:, 1:]

        # Parameters of the Dirichlet distribution must be positive
        inference_concentration = activations.softplus(inference_theta)
        model_concentration = activations.softplus(model_theta)

        # Calculate the KL divergence between the posterior and prior
        prior = tfp.distributions.Dirichlet(model_concentration)
        posterior = tfp.distributions.Dirichlet(inference_concentration)
        kl_loss = tfp.distributions.kl_divergence(posterior, prior)

        # Sum the KL loss for each time point and average over batches
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss = tf.reduce_mean(kl_loss, axis=0)

        return tf.expand_dims(kl_loss, axis=-1)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])


class InferenceRNNLayers(layers.Layer):
    """RNN inference network.

    Parameters
    ----------
    rnn_type : str
        Either 'lstm' or 'gru'. Defaults to GRU.
    normalization_type : str
        Either 'layer', 'batch' or None.
    n_layers : int
        Number of layers.
    n_units : int
        Number of units/neurons per layer.
    dropout_rate : float
        Dropout rate for the output of each layer.
    """

    def __init__(
        self,
        rnn_type: str,
        normalization_type: str,
        n_layers: int,
        n_units: int,
        dropout_rate: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_units = n_units

        # Choice of RNN
        if rnn_type == "lstm":
            RNNLayer = layers.LSTM
        else:
            RNNLayer = layers.GRU

        # Choice of normalisation layer
        if normalization_type == "layer":
            NormalizationLayer = layers.LayerNormalization
        elif normalization_type == "batch":
            NormalizationLayer = layers.BatchNormalization
        else:
            NormalizationLayer = DummyLayer

        self.layers = []
        for n in range(n_layers):
            self.layers.append(
                layers.Bidirectional(
                    layer=RNNLayer(n_units, return_sequences=True, stateful=False)
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
    """RNN generative model.

    Parameters
    ----------
    rnn_type : str
        Either 'lstm' or 'gru'. Defaults to GRU.
    normalization_type : str
        Either 'layer', 'batch' or None.
    n_layers : int
        Number of layers.
    n_units : int
        Number of units/neurons per layer.
    dropout_rate : float
        Dropout rate for the output of each layer.
    """

    def __init__(
        self,
        rnn_type: str,
        normalization_type: str,
        n_layers: int,
        n_units: int,
        dropout_rate: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_units = n_units

        # Choice of RNN
        if rnn_type == "lstm":
            RNNLayer = layers.LSTM
        else:
            RNNLayer = layers.GRU

        # Choice of normalisation layer
        if normalization_type == "layer":
            NormalizationLayer = layers.LayerNormalization
        elif normalization_type == "batch":
            NormalizationLayer = layers.BatchNormalization
        else:
            NormalizationLayer = DummyLayer

        self.layers = []
        for n in range(n_layers):
            self.layers.append(RNNLayer(n_units, return_sequences=True, stateful=False))
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


class MARParametersLayer(layers.Layer):
    """Layer to learn parameters of a multivariate autoregressive (MAR) model.

    Outputs the MAR parameters:
    - Matrix of MAR coefficients for each state and lag.
    - Covariance matrix for each state.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    n_lags : int
        Number of lags.
    initial_coeffs : np.ndarray
        Initial values for the MAR coefficients.
    initial_cov : np.ndarray
        Initial values for the covariances.
    learn_coeffs : bool
        Should we learn the MAR coefficients?
    learn_cov : bool
        Should we learn the covariances?
    """

    def __init__(
        self,
        n_states: int,
        n_channels: int,
        n_lags: int,
        initial_coeffs: np.ndarray,
        initial_cov: np.ndarray,
        learn_coeffs: bool,
        learn_cov: bool,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.n_channels = n_channels
        self.n_lags = n_lags
        self.learn_coeffs = learn_coeffs
        self.learn_cov = learn_cov

        # Initialisation for MAR coefficients
        if initial_coeffs is None:
            self.initial_coeffs = np.zeros(
                [n_states, n_lags, n_channels, n_channels], dtype=np.float32
            )
        else:
            self.initial_coeffs = initial_coeffs
        self.coeffs_initializer = WeightInitializer(self.initial_coeffs)

        # Initialisation for covariances
        if initial_cov is None:
            self.initial_cov = np.ones([n_states, n_channels], dtype=np.float32)
        else:
            self.initial_cov = initial_cov
        self.cov_initializer = WeightInitializer(self.initial_cov)

    def build(self, input_shape):

        # Create weights for the MAR coefficients
        self.coeffs = self.add_weight(
            "coeffs",
            shape=(self.n_states, self.n_lags, self.n_channels, self.n_channels),
            dtype=tf.float32,
            initializer=self.coeffs_initializer,
            trainable=self.learn_coeffs,
        )

        # Create weights for the MAR covariance
        self.cov = self.add_weight(
            "cov",
            shape=(self.n_states, self.n_channels),
            dtype=tf.float32,
            initializer=self.cov_initializer,
            trainable=self.learn_cov,
            constraint=constraints.NonNeg(),
        )

        self.built = True

    def call(self, inputs, **kwargs):
        return [self.coeffs, tf.linalg.diag(self.cov)]

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape(
                [self.n_states, self.n_lags, self.n_channels, self.n_channels]
            ),
            tf.TensorShape([self.n_states, self.n_channels, self.n_channels]),
        ]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_states": self.n_states,
                "n_channels": self.n_channels,
                "n_lags": self.n_lags,
                "learn_coeffs": self.learn_coeffs,
                "learn_cov": self.learn_cov,
            }
        )
        return config


class MARMeanCovLayer(layers.Layer):
    def __init__(
        self,
        n_states: int,
        n_channels: int,
        sequence_length: int,
        n_lags: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_lags = n_lags

    def call(self, inputs, **kwargs):

        # Input data:
        # - data_t.shape = (None, sequence_length, n_channels)
        # - alpha_jt.shape = (None, sequence_length, n_states)
        # - coeffs_jl.shape = (n_states, n_lags, n_channels, n_channels)
        # - cov_j.shape = (n_states, n_channels, n_channels)
        data_t, alpha_jt, coeffs_jl, cov_j = inputs

        # Data for the log-likelihood calculation
        clipped_data_t = tf.roll(data_t, shift=-1, axis=1)
        clipped_data_t = clipped_data_t[:, self.n_lags : -1]

        # Reshape data and coeffs for multiplication
        # data_t -> (None, sequence_length, 1, 1, n_channels, 1)
        data_t = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(data_t, axis=2), axis=3), axis=-1
        )
        # coeffs -> (1, 1, n_states, n_lags, n_channels, n_channels)
        coeffs_jl = tf.expand_dims(tf.expand_dims(coeffs_jl, axis=0), axis=0)

        # Multiply the data by the coefficients for each state and lag
        # lagged_data_jlt.shape = (None, sequence_length, n_states, n_lags, n_channels)
        lagged_data_jlt = tf.squeeze(tf.matmul(coeffs_jl, data_t), axis=-1)

        # Calculate the mean for each state: mu_jt = Sum_l coeffs_j data_{t-l}
        # mu_jt.shape = (None, sequence_length - n_lags, n_states, n_channels)
        mu_jt = lagged_data_jlt[:, :, :, 0]
        for lag in range(1, self.n_lags):
            mu_jt = tf.add(
                mu_jt, tf.roll(lagged_data_jlt[:, :, :, lag], shift=lag, axis=1)
            )
        mu_jt = mu_jt[:, self.n_lags : -1]

        # Remove alpha_jt value we don't have all lags for and
        # reshape for multiplication with mu_jt
        # alpha_jt -> (None, sequence_length - n_lags, n_states, 1)
        alpha_jt = tf.expand_dims(alpha_jt[:, self.n_lags : -1], axis=-1)

        # Calculate the mean at each point in time: mu_t = Sum_j alpha_jt mu_jt
        mu_t = tf.reduce_sum(tf.multiply(alpha_jt, mu_jt), axis=2)

        # Reshape cov_j and alpha_jt for multiplication
        # alpha_jt -> (None, sequence_length - n_lags, n_states, 1, 1)
        alpha_jt = tf.expand_dims(alpha_jt, axis=-1)

        # cov_j -> (1, 1, n_states, n_channels, n_channels)
        cov_j = tf.expand_dims(tf.expand_dims(cov_j, axis=0), axis=0)

        # Calculate the covariance at each time point: sigma_t = Sum_j alpha^2_jt cov_j
        alpha2_jt = tf.square(alpha_jt)
        sigma_t = tf.reduce_sum(tf.multiply(alpha2_jt, cov_j), axis=2)

        return clipped_data_t, mu_t, sigma_t

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape([None, self.sequence_length - self.n_lags, self.n_channels]),
            tf.TensorShape([None, self.sequence_length - self.n_lags, self.n_channels]),
            tf.TensorShape(
                [
                    None,
                    self.sequence_length - self.n_lags,
                    self.n_channels,
                    self.n_channels,
                ]
            ),
        ]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_states": self.n_states,
                "n_channels": self.n_channels,
                "sequence_length": self.sequence_length,
                "n_lags": self.n_lags,
            }
        )
        return config
