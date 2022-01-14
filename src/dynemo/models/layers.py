"""Custom Tensorflow layers used in the inference network and generative model.

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import activations, layers, regularizers
from dynemo.inference.initializers import WeightInitializer, CopyTensorInitializer

tfb = tfp.bijectors


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


class SampleNormalDistributionLayer(layers.Layer):
    """Layer for sampling from a normal distribution.

    This layer accepts the mean and the standard deviation and
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


class MeansCovsLayer(layers.Layer):
    """Layer to learn the mean and covariance of each mode.

    Outputs the mean vector and covariance matrix of each mode.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    learn_means : bool
        Should we learn the means?
    learn_covariances : bool
        Should we learn the covariances?
    initial_means : np.ndarray
        Initial values for the mean of each mode.
    initial_covariances : np.ndarray
        Initial values for the covariance of each mode. Must be dtype float32
    """

    def __init__(
        self,
        n_modes: int,
        n_channels: int,
        learn_means: bool,
        learn_covariances: bool,
        normalize_covariances: bool,
        initial_means: np.ndarray,
        initial_covariances: np.ndarray,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_modes = n_modes
        self.n_channels = n_channels
        self.learn_means = learn_means
        self.learn_covariances = learn_covariances
        self.normalize_covariances = normalize_covariances

        # Bijector used to transform covariance matrices to a learnable vector
        self.bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])

        # Initialisation of means
        if initial_means is None:
            self.initial_means = np.zeros([n_modes, n_channels], dtype=np.float32)
        else:
            self.initial_means = initial_means.astype("float32")

        self.means_initializer = WeightInitializer(self.initial_means)

        # Initialisation of covariances
        if initial_covariances is None:
            self.initial_covariances = np.stack(
                [np.eye(n_channels, dtype=np.float32)] * n_modes
            )
        else:
            self.initial_covariances = initial_covariances.astype("float32")

        if normalize_covariances:
            normalization = (
                tf.reduce_sum(tf.linalg.diag_part(self.initial_covariances), axis=1)[
                    ..., tf.newaxis, tf.newaxis
                ]
                / n_channels
            )
            self.initial_covariances = self.initial_covariances / normalization

        self.initial_flattened_cholesky_covariances = self.bijector.inverse(
            self.initial_covariances
        )

        self.flattened_cholesky_covariances_initializer = WeightInitializer(
            self.initial_flattened_cholesky_covariances
        )

    def build(self, input_shape):

        # Create weights the means
        self.means = self.add_weight(
            "means",
            shape=(self.n_modes, self.n_channels),
            dtype=tf.float32,
            initializer=self.means_initializer,
            trainable=self.learn_means,
        )

        # Create weights for the cholesky decomposition of the covariances
        self.flattened_cholesky_covariances = self.add_weight(
            "flattened_cholesky_covariances",
            shape=(self.n_modes, self.n_channels * (self.n_channels + 1) // 2),
            dtype=tf.float32,
            initializer=self.flattened_cholesky_covariances_initializer,
            trainable=self.learn_covariances,
        )

        self.built = True

    def call(self, inputs, **kwargs):

        # Calculate the covariance matrix from the cholesky factor
        self.covariances = self.bijector(self.flattened_cholesky_covariances)

        # Normalise the covariance by the trace
        if self.normalize_covariances:
            normalization = (
                tf.reduce_sum(tf.linalg.diag_part(self.covariances), axis=1)[
                    ..., tf.newaxis, tf.newaxis
                ]
                / self.n_channels
            )
            self.covariances = self.covariances / normalization

        return [self.means, self.covariances]


class MixMeansCovsLayer(layers.Layer):
    """Compute a probabilistic mixture of means and covariances.

    The mixture is calculated as  m_t = Sum_j alpha_jt mu_j and
    C_t = Sum_j alpha_jt D_j.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    learn_alpha_scaling : bool
        Should we learn an alpha scaling?
    """

    def __init__(
        self, n_modes: int, n_channels: int, learn_alpha_scaling: bool, **kwargs
    ):
        super().__init__(**kwargs)
        self.n_modes = n_modes
        self.n_channels = n_channels
        self.learn_alpha_scaling = learn_alpha_scaling

    def build(self, input_shape):

        # Initialise such that softplus(alpha_scaling) = 1
        self.alpha_scaling_initializer = tf.keras.initializers.Constant(
            np.log(np.exp(1.0) - 1.0)
        )
        self.alpha_scaling = self.add_weight(
            "alpha_scaling",
            shape=self.n_modes,
            dtype=tf.float32,
            initializer=self.alpha_scaling_initializer,
            trainable=self.learn_alpha_scaling,
        )
        self.built = True

    def call(self, inputs, **kwargs):

        # Unpack the inputs:
        # - alpha.shape = (None, sequence_length, n_modes)
        # - mu.shape    = (n_modes, n_channels)
        # - D.shape     = (n_modes, n_channels, n_channels)
        alpha, mu, D = inputs

        # Rescale the mode mixing factors
        alpha = tf.multiply(alpha, activations.softplus(self.alpha_scaling))

        # Reshape alpha and mu for multiplication
        alpha = tf.expand_dims(alpha, axis=-1)
        mu = tf.reshape(mu, (1, 1, self.n_modes, self.n_channels))

        # Calculate the mean: m_t = Sum_j alpha_jt mu_j
        m = tf.reduce_sum(tf.multiply(alpha, mu), axis=2)

        # Reshape alpha and D for multiplication
        alpha = tf.expand_dims(alpha, axis=-1)
        D = tf.reshape(D, (1, 1, self.n_modes, self.n_channels, self.n_channels))

        # Calculate the covariance: C_t = Sum_j alpha_jt D_j
        C = tf.reduce_sum(tf.multiply(alpha, D), axis=2)

        return [m, C]


class LogLikelihoodLayer(layers.Layer):
    """Layer to calculate the negative log likelihood.

    The negative log-likelihood is calculated assuming a multivariate normal
    probability density.

    Parameters
    ----------
    diag_only : bool
        Are the covariances passed just the diagonal? Optional, default is False.
    clip : int
        Number of data points to clip from the means and data.
        Optional, default is no clipping.
    """

    def __init__(self, diag_only: bool = False, clip: int = None, **kwargs):
        super().__init__(**kwargs)
        self.diag_only = diag_only
        self.clip = clip

    def call(self, inputs):
        x, mu, sigma = inputs

        # Clip data, means and covariances
        # This is neccessary if mu and sigma are one time step in the future
        if self.clip is not None:
            x = x[:, self.clip :]
            mu = mu[:, : -self.clip]
            if not self.diag_only:
                sigma = sigma[:, : -self.clip]

        # Calculate the log-likelihood
        if self.diag_only:
            mvn = tfp.distributions.MultivariateNormalDiag(
                loc=mu,
                scale_diag=sigma,
                allow_nan_stats=False,
            )
        else:
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

        # Sum the KL loss for each mode and time point and average over batches
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


class CoeffsCovsLayer(layers.Layer):
    """Layer to learn parameters of a multivariate autoregressive (MAR) model.

    Outputs the MAR parameters:
    - Matrix of MAR coefficients for each mode and lag.
    - Covariance matrix for each mode.

    Parameters
    ----------
    n_modes : int
        Number of modes.
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
        Are the covariances diagonal?
    """

    def __init__(
        self,
        n_modes: int,
        n_channels: int,
        n_lags: int,
        initial_coeffs: np.ndarray,
        initial_covs: np.ndarray,
        learn_coeffs: bool,
        learn_covs: bool,
        diag_covs: bool,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_modes = n_modes
        self.n_channels = n_channels
        self.n_lags = n_lags
        self.learn_coeffs = learn_coeffs
        self.learn_covs = learn_covs
        self.diag_covs = diag_covs

        # Initialisation for MAR coefficients
        if initial_coeffs is None:
            self.initial_coeffs = np.zeros(
                [n_modes, n_lags, n_channels, n_channels], dtype=np.float32
            )
        else:
            self.initial_coeffs = initial_coeffs
        self.coeffs_initializer = WeightInitializer(self.initial_coeffs)

        # Initialisation for covariances
        if self.diag_covs:
            if initial_covs is None:
                self.initial_covs = np.ones([n_modes, n_channels], dtype=np.float32)
            else:
                self.initial_covs = initial_covs

            self.initial_covs = np.log(np.exp(self.initial_covs) - 1.0)

            self.diagonal_covs_initializer = WeightInitializer(self.initial_covs)

        else:
            # Bijector used to transform covariance matrices to a learnable vector
            self.bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])

            if initial_covs is None:
                self.initial_covs = np.stack(
                    [np.eye(n_channels, dtype=np.float32)] * n_modes
                )

            else:
                if initial_covs.ndim == 2:
                    raise ValueError(
                        "Please pass covariances with shape (n_modes, n_channels, "
                        + "n_channels) or use diag_covs=True."
                    )
                self.initial_covs = initial_covs.astype("float32")

            self.initial_flattened_cholesky_covs = self.bijector.inverse(
                self.initial_covs
            )

            self.flattened_cholesky_covs_initializer = WeightInitializer(
                self.initial_flattened_cholesky_covs
            )

    def build(self, input_shape):

        # Create weights for the MAR coefficients
        self.coeffs = self.add_weight(
            "coeffs",
            shape=(self.n_modes, self.n_lags, self.n_channels, self.n_channels),
            dtype=tf.float32,
            initializer=self.coeffs_initializer,
            trainable=self.learn_coeffs,
        )

        # Create weights for the MAR covariances
        if self.diag_covs:
            self.diagonal_covs = self.add_weight(
                "diagonal_covs",
                shape=(self.n_modes, self.n_channels),
                dtype=tf.float32,
                initializer=self.diagonal_covs_initializer,
                trainable=self.learn_covs,
            )
        else:
            self.flattened_cholesky_covs = self.add_weight(
                "flattened_cholesky_covs",
                shape=(self.n_modes, self.n_channels * (self.n_channels + 1) // 2),
                dtype=tf.float32,
                initializer=self.flattened_cholesky_covs_initializer,
                trainable=self.learn_covs,
            )

        self.built = True

    def call(self, inputs, **kwargs):
        if self.diag_covs:
            # Ensure covariances contain variances that are positive
            self.covs = activations.softplus(self.diagonal_covs)
        else:
            # Calculate the covariance matrix from the cholesky factor
            self.covs = self.bijector(self.flattened_cholesky_covs)

        return [self.coeffs, self.covs]


class MixCoeffsCovsLayer(layers.Layer):
    """Mixes the MAR coefficients and covariances.

    Parameters
    ----------
    diag_only : bool
        Is the covariance matrix diagonal?
    """

    def __init__(self, diag_only: bool, **kwargs):
        super().__init__(**kwargs)
        self.diag_only = diag_only

    def call(self, inputs, **kwargs):

        # Input data:
        # - alpha_jt.shape = (None, sequence_length, n_modes)
        # - coeffs_jl.shape = (n_modes, n_lags, n_channels, n_channels)
        # - cov_j.shape = (n_modes, n_channels) if diag_only
        #   else (n_modes, n_channels, n_channels)
        alpha_jt, coeffs_jl, cov_j = inputs

        # Reshape alpha_jt and coeffs_jl for multiplication
        # alpha_jt -> (None, sequence_length, n_modes, 1, 1, 1)
        # coeffs_jl -> (1, 1, n_modes, n_lags, n_channels, n_channels)
        alpha_jt = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(alpha_jt, axis=-1), axis=-1), axis=-1
        )
        coeffs_jl = tf.expand_dims(tf.expand_dims(coeffs_jl, axis=0), axis=0)

        # Calculate coefficients for each lag at each time point:
        # - coeffs_lt = Sum_j alpha_jt coeffs_jl
        # - coeffs_lt.shape = (None, sequence_length, n_lags, n_channels, n_channels)
        coeffs_lt = tf.reduce_sum(tf.multiply(alpha_jt, coeffs_jl), axis=2)

        # Reshape alpha_jt and cov_j for multiplication
        # if diag_only:
        #     alpha_jt -> (None, sequence_length, n_modes, 1)
        #     cov_j -> (1, 1, n_modes, n_channels)
        # else:
        #     alpha_jt -> (None, sequence_length, n_modes, 1, 1)
        #     cov_j -> (1, 1, n_modes, n_channels, n_channels)
        if self.diag_only:
            alpha_jt = tf.squeeze(tf.squeeze(alpha_jt, axis=-1), axis=-1)
        else:
            alpha_jt = tf.squeeze(alpha_jt, axis=-1)
        cov_j = tf.expand_dims(tf.expand_dims(cov_j, axis=0), axis=0)

        # Calcalute covariance at each time point:
        # - cov_t = Sum_j alpha_jt cov_j
        cov_t = tf.reduce_sum(tf.multiply(alpha_jt, cov_j), axis=2)

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
        # - covs_t.shape = (None, sequence_length, n_channels, n_channels) if diag_only
        #   else (None, sequence_length, n_channels)
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


class VectorQuantizerLayer(layers.Layer):
    """Layer to perform vector quantization.

    Parameters
    ----------
    n_embeddings : int
        Number of vectors.
    embedding_dim : int
        Dimensionality of the vectors.
    beta : float
        Weighting term for the commitment loss.
    initial_embeddings : np.ndarray
        Initial values for the quantized vectors.
    learn_embeddings : bool
        Should we learn the quantized vectors?
    gamma : float
        Decay for the exponential moving average. Optional.
    epsilon : float
        Small error for numerical stability. Optional.
    """

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        beta: float,
        initial_embeddings: np.ndarray,
        learn_embeddings: bool,
        gamma: float = 0.99,
        epsilon: float = 1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1.")
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_embeddings = initial_embeddings
        self.learn_embeddings = learn_embeddings

        # Tensor for quantised vectors
        if self.initial_embeddings is None:
            self.embeddings_initializer = tf.random_uniform_initializer()
        else:
            self.embeddings_initializer = WeightInitializer(self.initial_embeddings)
        self.embeddings = tf.Variable(
            initial_value=self.embeddings_initializer(
                shape=(embedding_dim, n_embeddings),
                dtype=tf.float32,
            ),
            trainable=False,
            name="embeddings",
        )

        # Tensor for cluster size exponential moving average
        self.cluster_size_ema_initializer = tf.zeros_initializer()
        self.cluster_size_ema = tf.Variable(
            initial_value=self.cluster_size_ema_initializer(
                shape=(n_embeddings,),
                dtype=tf.float32,
            ),
            trainable=False,
            name="cluster_size_ema",
        )

        # Tensor for quantised vector exponential moving average
        self.embeddings_ema_initializer = CopyTensorInitializer(self.embeddings)
        self.embeddings_ema = tf.Variable(
            initial_value=self.embeddings_ema_initializer(
                shape=(embedding_dim, n_embeddings),
                dtype=tf.float32,
            ),
            trainable=False,
            name="embeddings_ema",
        )

    def call(self, inputs, training=None):

        # Flatten the inputs keeping embedding_dim intact
        flattened_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        # Calculate L2 distance between the inputs and the embeddings
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0, keepdims=True)
            - 2 * tf.matmul(flattened_inputs, self.embeddings)
        )

        # Derive the indices for minimum distances
        embedding_indices = tf.argmin(distances, axis=1)

        # One hot vectors containing classifications of each vector
        one_hot_embeddings = tf.one_hot(embedding_indices, self.n_embeddings)

        # Quantization
        embedding_indices = tf.reshape(embedding_indices, tf.shape(inputs)[:-1])
        quantized = tf.nn.embedding_lookup(
            tf.transpose(self.embeddings), embedding_indices
        )

        if training and self.learn_embeddings:
            # Update codebook

            # Calculate exponential moving average for the cluster size
            one_hot_embeddings_sum = tf.reduce_sum(one_hot_embeddings, axis=0)
            self.cluster_size_ema.assign(
                self.cluster_size_ema * self.gamma
                + one_hot_embeddings_sum * (1 - self.gamma)
            )

            # Calculate exponential moving average for the quantised vectors
            embeddings_sum = tf.matmul(
                flattened_inputs, one_hot_embeddings, transpose_a=True
            )
            self.embeddings_ema.assign(
                self.embeddings_ema * self.gamma + embeddings_sum * (1 - self.gamma)
            )
            n = tf.reduce_sum(self.cluster_size_ema)
            cluster_size_ema = (
                (self.cluster_size_ema + self.epsilon)
                / (n + self.n_embeddings * self.epsilon)
                * n
            )

            # Calculate the new codebook quantised vectors
            normalized_embeddings = self.embeddings_ema / tf.reshape(
                cluster_size_ema, [1, -1]
            )
            self.embeddings.assign(normalized_embeddings)

        # Add the commitment loss to the total loss
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - inputs) ** 2
        )
        self.add_loss(commitment_loss)

        # Straight-through estimator
        quantized = inputs + tf.stop_gradient(quantized - inputs)

        return quantized


class WaveNetLayer(layers.Layer):
    """Layer for generating data using a WaveNet architecture.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    n_filters : int
        Number of filters for each convolution.
    n_layers : int
        Number of dilated causal convolution layers in each residual block.
    """

    def __init__(self, n_channels: int, n_filters: int, n_layers: int, **kwargs):
        super().__init__(**kwargs)

        self.n_channels = n_channels
        self.n_filters = n_filters
        self.n_layers = n_layers

        self.causal_conv_layer = layers.Conv1D(
            filters=n_filters,
            kernel_size=2,
            dilation_rate=1,
            padding="causal",
            use_bias=False,
            kernel_regularizer=regularizers.l2(),
        )
        self.residual_block_layers = []
        for i in range(1, n_layers):
            self.residual_block_layers.append(
                WaveNetResidualBlockLayer(filters=n_filters, dilation_rate=2 ** i)
            )
        self.dense_layers = [
            layers.Conv1D(
                filters=n_channels,
                kernel_size=1,
                padding="same",
                kernel_regularizer=regularizers.l2(),
                bias_regularizer=regularizers.l2(),
            ),
            layers.Conv1D(
                filters=n_channels,
                kernel_size=1,
                padding="same",
                kernel_regularizer=regularizers.l2(),
                bias_regularizer=regularizers.l2(),
            ),
        ]

    def call(self, inputs, training=None, **kwargs):
        x, alpha = inputs
        alpha = tf.roll(alpha, shift=-1, axis=1)
        out = self.causal_conv_layer(x)
        skips = []
        for layer in self.residual_block_layers:
            out, skip = layer([out, alpha])
            skips.append(skip)
        out = tf.add_n(skips)
        for layer in self.dense_layers:
            out = activations.selu(out)
            out = layer(out)
        return out


class WaveNetResidualBlockLayer(layers.Layer):
    """Layer for a residual block in WaveNet.

    Parameters
    ----------
    filters : int
        Number of filters for the convolutions.
    dilation_rate : int
        Dilation rate for the causal convolutions.
    """

    def __init__(self, filters: int, dilation_rate: int, **kwargs):
        super().__init__(**kwargs)
        self.h_transform_layer = layers.Conv1D(filters, kernel_size=1, padding="same")
        self.x_filter_layer = layers.Conv1D(
            filters,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding="causal",
            use_bias=False,
            kernel_regularizer=regularizers.l2(),
        )
        self.y_filter_layer = layers.Conv1D(
            filters,
            kernel_size=1,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(),
        )
        self.x_gate_layer = layers.Conv1D(
            filters,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding="causal",
            use_bias=False,
            kernel_regularizer=regularizers.l2(),
        )
        self.y_gate_layer = layers.Conv1D(
            filters,
            kernel_size=1,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(),
        )
        self.res_layer = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            kernel_regularizer=regularizers.l2(),
            bias_regularizer=regularizers.l2(),
        )
        self.skip_layer = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            kernel_regularizer=regularizers.l2(),
            bias_regularizer=regularizers.l2(),
        )

    def call(self, inputs, training=None, **kwargs):
        x, h = inputs
        y = self.h_transform_layer(h)
        x_filter = self.x_filter_layer(x)
        y_filter = self.y_filter_layer(y)
        x_gate = self.x_gate_layer(x)
        y_gate = self.y_gate_layer(y)
        z = tf.tanh(x_filter + y_filter) * tf.sigmoid(x_gate + y_gate)
        residual = self.res_layer(z)
        skip = self.skip_layer(z)
        return x + residual, skip


class StdDevLayer(layers.Layer):
    """Layer to learn standard deviations.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    learn_std_dev : bool
        Should we learn the standard deviation.
    initial_std_dev : int
        Initial values for the standard deviation.
    """

    def __init__(
        self,
        n_channels: int,
        learn_std_dev: bool,
        initial_std_dev: np.ndarray = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.learn_std_dev = learn_std_dev

        # Initialisation of standard deviation
        if initial_std_dev is None:
            self.initial_std_dev = np.ones(n_channels, dtype=np.float32)
        else:
            self.initial_std_dev = initial_std_dev.astype("float32")
        self.std_dev_initializer = WeightInitializer(self.initial_std_dev)

    def build(self, input_shape):
        self.std_dev = self.add_weight(
            "std_dev",
            shape=(self.n_channels),
            dtype=tf.float32,
            initializer=self.std_dev_initializer,
            trainable=self.learn_std_dev,
        )
        self.built = True

    def call(self, inputs):
        return activations.softplus(self.std_dev)


class MeanSquaredErrorLayer(layers.Layer):
    """Layer for calculating the mean squared error.

    Parameters
    ----------
    clip : int
        Number of data points to clip from each input.
        Optional, default is no clipping.
    """

    def __init__(self, clip: int = None, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip

    def call(self, inputs, training=None, **kwargs):
        training_data, generated_data = inputs

        if self.clip is not None:
            training_data = training_data[:, self.clip :]
            generated_data = generated_data[:, : -self.clip]

        se = tf.math.squared_difference(training_data, generated_data)
        mse = tf.reduce_mean(se)  # mean over batches, time points and channels

        return tf.expand_dims(mse, axis=-1)


class MeansStdsFcsLayer(layers.Layer):
    """Layer to learn the means, diagonal standard deviation matrices,
    and functional connectivities of modes.

    Outputs the mean vector, the standard deviations, and correlation (fc) matrix
    of each mode.

    Paramters
    ---------
    n_modes: int
        Number of modes.
    n_channels: int
        Number of channels.
    learn_means : bool
        Should we learn the means?
    learn_stds : bool
        Should we learn the standard deviations?
    learn_fcs : bool
        Should we learn the functional connectivities?
    initial_means : np.ndarray
        Initial value of the mean each of mode. Optional.
    initial_stds: np.ndarray
        Initial value of the standard deviation of each model. Optional.
    initial_fcs : np.ndarray
        Initial values of the functional connectivity of each mode. Optional.
    """

    def __init__(
        self,
        n_modes: int,
        n_channels: int,
        learn_means: bool,
        learn_stds: bool,
        learn_fcs: bool,
        initial_means: np.ndarray,
        initial_stds: np.ndarray,
        initial_fcs: np.ndarray,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_modes = n_modes
        self.n_channels = n_channels
        self.learn_means = learn_means
        self.learn_stds = learn_stds
        self.learn_fcs = learn_fcs

        self.bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])

        # Initialisation of means
        if initial_means is None:
            self.initial_means = np.zeros([n_modes, n_channels], dtype=np.float32)
        else:
            self.initial_means = initial_means

        self.means_initializer = WeightInitializer(self.initial_means)

        # Initialisation of standard deviations
        if initial_stds is None:
            self.initial_stds = np.ones([n_modes, n_channels], dtype=np.float32)
        else:
            self.initial_stds = initial_stds

        self.stds_initializer = WeightInitializer(self.initial_stds)

        # Initialisation of functional connectiviy matrices
        if initial_fcs is None:
            self.initial_fcs = np.stack(
                [np.eye(n_channels, dtype=np.float32)] * n_modes
            )
        else:
            self.initial_fcs = initial_fcs

        self.initial_flattened_cholesky_fcs = self.bijector.inverse(self.initial_fcs)

        self.flattened_cholesky_fcs_initializer = WeightInitializer(
            self.initial_flattened_cholesky_fcs
        )

    def build(self, input_shape):

        # Create weights for the means
        self.means = self.add_weight(
            "means",
            shape=(self.n_modes, self.n_channels),
            dtype=tf.float32,
            initializer=self.means_initializer,
            trainable=self.learn_means,
        )

        # Create weights for the diagonal entries
        self.stds = self.add_weight(
            "stds",
            shape=(self.n_modes, self.n_channels),
            dtype=tf.float32,
            initializer=self.stds_initializer,
            trainable=self.learn_stds,
        )

        # Create weights for the lower triangular entries
        self.flattened_cholesky_fcs = self.add_weight(
            "flattened_cholesky_fcs",
            shape=(self.n_modes, self.n_channels * (self.n_channels + 1) // 2),
            dtype=tf.float32,
            initializer=self.flattened_cholesky_fcs_initializer,
            trainable=self.learn_fcs,
        )

        self.built = True

    def call(self, inputs, **kwargs):

        # Make sure standard deviations are positive
        stds = activations.softplus(self.stds)

        # The L2 norm of cholesky vectors for penalisation in the loss
        L2norm_cholesky = tf.reduce_sum(tf.norm(self.flattened_cholesky_fcs, axis=1))
        L2norm_cholesky = tf.expand_dims(L2norm_cholesky, axis=0)
        
        # Calculate functional connectivity matrix from flattened vector
        fcs = self.bijector(self.flattened_cholesky_fcs)

        # # Normalise so that FCs are correlation matrices
        # sd = tf.expand_dims(tf.sqrt(tf.linalg.diag_part(fcs)), axis=-1)
        # fcs /= tf.matmul(sd, sd, transpose_b=True)

        return [self.means, stds, fcs, L2norm_cholesky]


class MixMeansStdsFcsLayer(layers.Layer):
    """Compute a probabilistic mixture of means, standard deviations and functional
    connectivies.

    The mixture is calculated as

    m_t       = Sum_j alpha_jt mu_j
    diag(G_t) = Sum_j beta_jt E_j
    F_t       = Sum_j gamma_jt D_j
    C_t       = G_t F_t G_t

    Parameters
    ----------
    n_modes: int
        number of modes.
    n_channels: int
        number of channels.
    fix_std : bool
        Do we want to fix the standard deviation time course?
    """

    def __init__(self, n_modes: int, n_channels: int, fix_std: bool, **kwargs):
        super().__init__(**kwargs)
        self.n_modes = n_modes
        self.n_channels = n_channels
        self.fix_std = fix_std

    def call(self, inputs, **kwargs):
        # Unpack the inputs:
        # - alpha.shape = (None, sequence_length, n_modes)
        # - beta.shape = (None, sequence_length, n_modes)
        # - gamma.shape = (None, sequence_length, n_modes)
        # - mu.shape = (n_modes, n_channels)
        # - E.shape = (n_modes, n_channels)
        # - D.shape = (n_modes, n_channels, n_channels)
        alpha, beta, gamma, mu, E, D = inputs

        # Reshape alpha and mu for multiplication
        alpha = tf.expand_dims(alpha, axis=-1)
        mu = tf.reshape(mu, (1, 1, self.n_modes, self.n_channels))

        # Calculate the mixed mean
        m = tf.reduce_sum(tf.multiply(alpha, mu), axis=2)

        # Reshape beta and E for multiplication
        beta = tf.expand_dims(beta, axis=-1)
        E = tf.reshape(E, (1, 1, self.n_modes, self.n_channels))

        # Calculate the mixed diagonal entries
        G = tf.reduce_sum(tf.multiply(beta, E), axis=2)
        G = tf.linalg.diag(G)

        # Reshape gamma and D for multiplication
        gamma = tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1)
        D = tf.reshape(D, (1, 1, self.n_modes, self.n_channels, self.n_channels))
        F = tf.reduce_sum(tf.multiply(gamma, D), axis=2)

        # Normalise F so that it is a valid correlation matrix
        sd = tf.expand_dims(tf.sqrt(tf.linalg.diag_part(F)), axis=-1)
        F /= tf.matmul(sd, sd, transpose_b=True)

        # Construct the covariance matrices given by C = GFG
        C = tf.matmul(G, tf.matmul(F, G))

        return [m, C]


class Sum(layers.Layer):
    """Layer to sum a set of tensors.

    This layer is a wrapper for tensorflow.add_n.
    """

    def call(self, inputs, **kwargs):
        return tf.add_n(inputs)


class FillConstant(layers.Layer):
    """Layer to create tensor with the same shape of the input,
    but filled with a constant.
    """

    def __init__(self, constant, **kwargs):
        super().__init__(**kwargs)
        self.constant = constant

    def call(self, inputs, **kwargs):
        return tf.fill(tf.shape(inputs), self.constant)
