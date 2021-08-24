"""Custom Tensorflow layers used in the inference network and generative model.

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import activations, layers
from vrad.inference.functions import (
    cholesky_factor,
    cholesky_factor_to_full_matrix,
    is_symmetric,
    trace_normalize,
)
from vrad.inference.initializers import WeightInitializer


def NormalizationLayer(norm_type: str, *args, **kwargs):
    """Returns a normalization layer.

    Parameters
    ----------
    norm_type : str
        Type of normalization layer. Either 'layer', 'batch' or None.
    """
    if norm_type == "layer":
        return layers.LayerNormalization(*args, **kwargs)
    elif norm_type == "batch":
        return layers.BatchNormalization(*args, **kwargs)
    elif norm_type is None:
        return DummyLayer(*args, **kwargs)
    else:
        raise NotImplementedError(norm_type)


def RNNLayer(rnn_type: str, *args, **kwargs):
    """Returns an RNN layer.

    Parameters
    ----------
    rnn_type : str
        Type of RNN. Either 'lstm' or 'gru'.
    """
    if rnn_type == "lstm":
        return layers.LSTM(*args, **kwargs)
    elif rnn_type == "gru":
        return layers.GRU(*args, **kwargs)
    else:
        raise NotImplementedError(rnn_type)


class DummyLayer(layers.Layer):
    """Dummy layer.

    Returns the inputs without modification.
    """

    def call(self, inputs, **kwargs):
        return inputs


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

    def call(self, inputs, training=None, **kwargs):
        mu, sigma = inputs
        if training:
            N = tfp.distributions.Normal(loc=mu, scale=sigma)
            return N.sample()
        else:
            return mu


class SampleDirichletDistributionLayer(layers.Layer):
    """Layer for sampling from a Dirichlet distribution.

    This layer accepts the parameters of a Dirichlet distribution and
    outputs a sample.
    """

    def call(self, concentration, training=None, **kwargs):
        if training:
            D = tfp.distributions.Dirichlet(concentration)
            return D.sample()
        else:
            sum_concentration = tf.reduce_sum(concentration, axis=-1)
            sum_concentration = tf.expand_dims(sum_concentration, axis=-1)
            return tf.divide(concentration, sum_concentration)


class ThetaActivationLayer(layers.Layer):
    """Layer for applying an activation function to theta.

    This layer accepts theta and outputs alpha.

    Parameters
    ----------
    alpha_xform : str
        The functional form of the activation used to convert from theta to alpha.
    initial_alpha_temperature : float
        Temperature parameter for the softmax or Gumbel-Softmax.
    learn_alpha_temperature : bool
        Should we learn the alpha temperature?
    """

    def __init__(
        self,
        alpha_xform: str,
        initial_alpha_temperature: float,
        learn_alpha_temperature: bool,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha_xform = alpha_xform
        self.initial_alpha_temperature = initial_alpha_temperature
        self.learn_alpha_temperature = learn_alpha_temperature

    def build(self, input_shape):

        # Initialiser for the a learnable alpha temperature
        def alpha_temperature_initializer(shape, dtype=None):
            return self.initial_alpha_temperature

        self.alpha_temperature_initializer = alpha_temperature_initializer

        # Create trainable alpha temperature
        self.alpha_temperature = self.add_weight(
            "alpha_temperature",
            shape=(),
            dtype=tf.float32,
            initializer=self.alpha_temperature_initializer,
            trainable=self.learn_alpha_temperature,
        )

        self.built = True

    def call(self, theta, **kwargs):

        # Calculate alpha from theta
        if self.alpha_xform == "softplus":
            alpha = activations.softplus(theta)
        elif self.alpha_xform == "softmax":
            alpha = activations.softmax(theta / self.alpha_temperature, axis=2)
        elif self.alpha_xform == "gumbel-softmax":
            gumbel_softmax_distribution = tfp.distributions.RelaxedOneHotCategorical(
                temperature=self.alpha_temperature,
                logits=theta,
            )
            alpha = gumbel_softmax_distribution.sample()

        return alpha

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha_xform": self.alpha_xform,
                "initial_alpha_temperature": self.initial_alpha_temperature,
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
        # - alpha.shape = (None, sequence_length, n_states)
        # - mu.shape    = (n_states, n_channels)
        # - D.shape     = (n_states, n_channels, n_channels)
        alpha, mu, D = inputs

        # Rescale the state mixing factors
        alpha = tf.multiply(alpha, activations.softplus(self.alpha_scaling))

        # Reshape alpha and mu for multiplication
        alpha = tf.expand_dims(alpha, axis=-1)
        mu = tf.reshape(mu, (1, 1, self.n_states, self.n_channels))

        # Calculate the mean: m_t = Sum_j alpha_jt mu_j
        m = tf.reduce_sum(tf.multiply(alpha, mu), axis=2)

        # Reshape alpha and D for multiplication
        alpha = tf.expand_dims(alpha, axis=-1)
        D = tf.reshape(D, (1, 1, self.n_states, self.n_channels, self.n_channels))

        # Calculate the covariance: C_t = Sum_j alpha_jt D_j
        C = tf.reduce_sum(tf.multiply(alpha, D), axis=2)

        return [m, C]

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

    def call(self, inputs):
        x, mu, sigma = inputs

        # Calculate the log-likelihood
        mvn = tfp.distributions.MultivariateNormalTriL(
            loc=mu,
            scale_tril=tf.linalg.cholesky(sigma),
            allow_nan_stats=False,
        )
        ll_loss = mvn.log_prob(x)

        # Sum over time dimension and average over the batch dimension
        ll_loss = tf.reduce_sum(ll_loss, axis=1)
        ll_loss = tf.reduce_mean(ll_loss, axis=0)

        # We return the negative of the log likelihood
        nll_loss = -ll_loss

        return tf.expand_dims(nll_loss, axis=-1)


class NormalKLDivergenceLayer(layers.Layer):
    """Layer to calculate a KL divergence between two Normal distributions."""

    def call(self, inputs, **kwargs):
        inference_mu, inference_sigma, model_mu, model_sigma = inputs

        # The Model RNN predicts one time step into the future compared to the
        # inference RNN. We clip its last value, and first value of the inference RNN.
        model_mu = model_mu[:, :-1]
        model_sigma = model_sigma[:, :-1]

        inference_mu = inference_mu[:, 1:]
        inference_sigma = inference_sigma[:, 1:]

        # Calculate the KL divergence between the posterior and prior
        prior = tfp.distributions.Normal(loc=model_mu, scale=model_sigma)
        posterior = tfp.distributions.Normal(loc=inference_mu, scale=inference_sigma)
        kl_loss = tfp.distributions.kl_divergence(
            posterior, prior, allow_nan_stats=False
        )

        # Sum the KL loss for each state and time point and average over batches
        kl_loss = tf.reduce_sum(kl_loss, axis=2)
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss = tf.reduce_mean(kl_loss, axis=0)

        return tf.expand_dims(kl_loss, axis=-1)


class DirichletKLDivergenceLayer(layers.Layer):
    """Layer to calculate a KL divergence between two Dirichlet distributions."""

    def call(self, inputs, **kwargs):
        inference_concentration, model_concentration = inputs

        # The Model RNN predicts one time step into the future compared to the
        # inference RNN. We clip its last value, and first value of the inference RNN.
        model_concentration = model_concentration[:, :-1]
        inference_concentration = inference_concentration[:, 1:]

        # Calculate the KL divergence between the posterior and prior
        prior = tfp.distributions.Dirichlet(model_concentration)
        posterior = tfp.distributions.Dirichlet(inference_concentration)
        kl_loss = tfp.distributions.kl_divergence(posterior, prior)

        # Sum the KL loss for each time point and average over batches
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss = tf.reduce_mean(kl_loss, axis=0)

        return tf.expand_dims(kl_loss, axis=-1)


class InferenceRNNLayers(layers.Layer):
    """RNN inference network.

    Parameters
    ----------
    rnn_type : str
        Either 'lstm' or 'gru'. Defaults to GRU.
    norm_type : str
        Either 'layer', 'batch' or None.
    act_type : 'str'
        Activation type, e.g. 'relu', 'elu', etc.
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
        norm_type: str,
        act_type: str,
        n_layers: int,
        n_units: int,
        dropout_rate: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_units = n_units

        self.layers = []
        for n in range(n_layers):
            self.layers.append(
                layers.Bidirectional(
                    layer=RNNLayer(
                        rnn_type, n_units, return_sequences=True, stateful=False
                    )
                )
            )
            self.layers.append(NormalizationLayer(norm_type))
            self.layers.append(layers.Activation(act_type))
            self.layers.append(layers.Dropout(dropout_rate))

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs

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
    norm_type : str
        Either 'layer', 'batch' or None.
    act_type : 'str'
        Activation type, e.g. 'relu', 'elu', etc.
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
        norm_type: str,
        act_type: str,
        n_layers: int,
        n_units: int,
        dropout_rate: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_units = n_units

        self.layers = []
        for n in range(n_layers):
            self.layers.append(
                RNNLayer(rnn_type, n_units, return_sequences=True, stateful=False)
            )
            self.layers.append(NormalizationLayer(norm_type))
            self.layers.append(layers.Activation(act_type))
            self.layers.append(layers.Dropout(dropout_rate))

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"n_units": self.n_units})
        return config


class CoeffsCovsLayer(layers.Layer):
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
    diag_covs : bool
        Are the covariances diagonal? Optional, default is False.
    """

    def __init__(
        self,
        n_states: int,
        n_channels: int,
        n_lags: int,
        initial_coeffs: np.ndarray,
        initial_covs: np.ndarray,
        learn_coeffs: bool,
        learn_covs: bool,
        diag_covs: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.n_channels = n_channels
        self.n_lags = n_lags
        self.learn_coeffs = learn_coeffs
        self.learn_covs = learn_covs
        self.diag_covs = diag_covs

        # Initialisation for MAR coefficients
        if initial_coeffs is None:
            self.initial_coeffs = np.zeros(
                [n_states, n_lags, n_channels, n_channels], dtype=np.float32
            )
        else:
            self.initial_coeffs = initial_coeffs
        self.coeffs_initializer = WeightInitializer(self.initial_coeffs)

        # Initialisation for covariances
        if self.diag_covs:
            if initial_covs is None:
                self.initial_covs = np.ones([n_states, n_channels], dtype=np.float32)
            else:
                self.initial_covs = initial_covs
            self.initial_covs = np.log(np.exp(self.initial_covs) - 1.0)
            self.diagonal_covs_initializer = WeightInitializer(self.initial_covs)
        else:
            if initial_covs is None:
                self.initial_covs = np.stack(
                    [np.eye(n_channels, dtype=np.float32)] * n_states
                )
                self.initial_cholesky_covs = cholesky_factor(self.initial_covs)
            else:
                if initial_covs.ndim == 2:
                    raise ValueError(
                        "Please pass covariances with shape (n_states, n_channels, "
                        + "n_channels) or use diag_covs=True."
                    )
                initial_covs = initial_covs.astype("float32")
                if is_symmetric(initial_covs):
                    self.initial_cholesky_covs = cholesky_factor(initial_covs)
                else:
                    self.initial_cholesky_covs = initial_covs
            self.flattened_cholesky_covs_initializer = WeightInitializer(
                tfp.math.fill_triangular_inverse(self.initial_cholesky_covs)
            )

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
        if self.diag_covs:
            self.diagonal_covs = self.add_weight(
                "diagonal_covs",
                shape=(self.n_states, self.n_channels),
                dtype=tf.float32,
                initializer=self.diagonal_covs_initializer,
                trainable=self.learn_covs,
            )
        else:
            self.flattened_cholesky_covs = self.add_weight(
                "flattened_cholesky_covs",
                shape=(self.n_states, self.n_channels * (self.n_channels + 1) // 2),
                dtype=tf.float32,
                initializer=self.flattened_cholesky_covs_initializer,
                trainable=self.learn_covs,
            )

        self.built = True

    def call(self, inputs, **kwargs):
        if self.diag_covs:
            # Ensure covariances contain variances that are positive
            covs = activations.softplus(self.diagonal_covs)
            covs = tf.linalg.diag(covs)
        else:
            # Calculate the covariance matrix from the cholesky factor
            cholesky_covs = tfp.math.fill_triangular(self.flattened_cholesky_covs)
            covs = cholesky_factor_to_full_matrix(cholesky_covs)

        return [self.coeffs, covs]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_states": self.n_states,
                "n_channels": self.n_channels,
                "n_lags": self.n_lags,
                "learn_coeffs": self.learn_coeffs,
                "learn_covs": self.learn_covs,
                "diag_covs": diag_covs,
            }
        )
        return config


class MixCoeffsCovsLayer(layers.Layer):
    """Mixes the MAR coefficients and covariances."""

    def call(self, inputs, **kwargs):

        # Input data:
        # - alpha_jt.shape = (None, sequence_length, n_states)
        # - coeffs_jl.shape = (n_states, n_lags, n_channels, n_channels)
        # - cov_j.shape = (n_states, n_channels, n_channels)
        alpha_jt, coeffs_jl, cov_j = inputs

        # Reshape alpha_jt and coeffs_jl for multiplication
        # alpha_jt -> (None, sequence_length, n_states, 1, 1, 1)
        # coeffs_jl -> (1, 1, n_states, n_lags, n_channels, n_channels)
        alpha_jt = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(alpha_jt, axis=-1), axis=-1), axis=-1
        )
        coeffs_jl = tf.expand_dims(tf.expand_dims(coeffs_jl, axis=0), axis=0)

        # Calculate coefficients for each lag at each time point:
        # - coeffs_lt = Sum_j alpha_jt coeffs_jl
        # - coeffs_lt.shape = (None, sequence_length, n_lags, n_channels, n_channels)
        coeffs_lt = tf.reduce_sum(tf.multiply(alpha_jt, coeffs_jl), axis=2)

        # Reshape alpha_jt and cov_j for multiplication
        # alpha_jt -> (None, sequence_length, n_states, 1, 1)
        # cov_j -> (1, 1, n_states, n_channels, n_channels)
        alpha_jt = tf.squeeze(alpha_jt, axis=-1)
        cov_j = tf.expand_dims(tf.expand_dims(cov_j, axis=0), axis=0)

        # Calcalute covariance at each time point:
        # - cov_t = Sum_j alpha^2_jt cov_j
        # - cov_t.shape = (None, sequence_length, n_channels, n_channels)
        alpha2_jt = tf.square(alpha_jt)
        cov_t = tf.reduce_sum(tf.multiply(alpha2_jt, cov_j), axis=2)

        return coeffs_lt, cov_t


class MARMeansCovsLayer(layers.Layer):
    """Calculates the time-vaying mean and covariance for observing MAR data.

    Parameters
    ----------
    n_lags : int
        Number of lags.
    """

    def __init__(self, n_lags: int, **kwargs):
        super().__init__(**kwargs)
        self.n_lags = n_lags

    def call(self, inputs, **kwargs):

        # Input data:
        # - data.shape = (None, sequence_length, n_channels)
        # - coeffs_lt.shape = (None, sequence_length, n_lags, n_channels, n_channels)
        # - covs_t.shape = (None, sequence_length, n_channels, n_channels)
        data, coeffs_lt, covs_t = inputs

        # Reshape data for multiplication
        # x -> (None, sequence_length, n_channels, 1)
        x_t = tf.expand_dims(data, axis=-1)

        # mu_t.shape = (None, sequence_length, n_channels)
        mu_t = tf.squeeze(tf.matmul(coeffs_lt[:, :, 0], x_t), axis=-1)
        for lag in range(1, self.n_lags):
            # Make sure we multiply the data by the coefficients at the right
            # time point
            coeffs_lt = tf.roll(coeffs_lt, shift=-lag, axis=1)
            coeffs_x_lt = tf.squeeze(tf.matmul(coeffs_lt[:, :, lag], x_t), axis=-1)
            mu_t = tf.add(mu_t, tf.roll(coeffs_x_lt, shift=lag, axis=1))

        # The mean mu_t defines the probability distribution function
        # for the next data point, so we need to roll the data back
        # one time step
        x_t = tf.roll(data, shift=-1, axis=1)

        # Covariances for the probability distribution function
        sigma_t = covs_t

        # Remove data points we don't have all the lags for
        x_t = x_t[:, self.n_lags : -1]
        mu_t = mu_t[:, self.n_lags : -1]
        sigma_t = sigma_t[:, self.n_lags : -1]

        return x_t, mu_t, sigma_t

    def get_config(self):
        config = super().get_config()
        config.update({"n_lags": self.n_lags})
        return config


class VectorQuantizerLayer(layers.Layer):
    """Layer to perform vector quantization.

    Parameters
    ----------
    n_vectors : int
        Number of vectors.
    vector_dim : int
        Dimensionality of the vectors.
    beta : float
        Weighting term for the commitment loss. Optional, default is 0.25.
    """

    def __init__(self, n_vectors: int, vector_dim: int, beta: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.n_vectors = n_vectors
        self.vector_dim = vector_dim
        self.beta = beta

        # Initializer for the quantised vectors
        self.quantized_vectors_initializer = tf.random_uniform_initializer()

    def build(self, input_shape):

        # Trainable weights for the quantised vectors
        self.quantized_vectors = self.add_weight(
            "quantized_vectors",
            shape=(self.vector_dim, self.n_vectors),
            dtype=tf.float32,
            initializer=self.quantized_vectors_initializer,
            trainable=True,
        )

        self.built = True

    def call(self, inputs):
        input_shape = tf.shape(inputs)

        # Flatten the inputs keeping vector_dim intact
        flattened = tf.reshape(inputs, [-1, self.vector_dim])

        # Quantization
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.n_vectors)
        quantized = tf.matmul(encodings, self.quantized_vectors, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - inputs) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator
        quantized = inputs + tf.stop_gradient(quantized - inputs)

        return quantized

    def get_code_indices(self, flattened_inputs):

        # Calculate L2-normalized distance between the inputs and the codes
        similarity = tf.matmul(flattened_inputs, self.quantized_vectors)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.quantized_vectors ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances
        encoding_indices = tf.argmin(distances, axis=1)

        return encoding_indices
