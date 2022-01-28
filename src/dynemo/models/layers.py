"""Custom Tensorflow layers used in the inference network and generative model.

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import activations, layers, regularizers
from dynemo.inference.initializers import WeightInitializer

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


class ConcatenateLayer(layers.Layer):
    """Concatenates a set of tensors.

    Wrapper for tf.concat().

    Parameters
    ----------
    axis : int
        Axis to concatenate along.
    """

    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.concat(inputs, axis=self.axis)


class MatMulLayer(layers.Layer):
    """Multiplies a set of matrices.

    Wrapper for tf.matmul().
    """

    def call(self, inputs, **kwargs):
        # If [A, B, C] is passed, we return matmul(A, matmul(B, C))
        out = inputs[-1]
        for tensor in inputs[len(inputs) - 2 :: -1]:
            out = tf.matmul(tensor, out)
        return out


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


class ThetaActivationLayer(layers.Layer):
    """Layer for applying an activation function to theta.

    This layer accepts theta and outputs alpha.

    Parameters
    ----------
    xform : str
        The functional form of the activation used to convert from theta to alpha.
    initial_temperature : float
        Temperature parameter for the softmax or Gumbel-Softmax.
    learn_temperature : bool
        Should we learn the alpha temperature?
    """

    def __init__(
        self,
        xform: str,
        initial_temperature: float,
        learn_temperature: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.xform = xform
        self.initial_temperature = initial_temperature
        self.learn_temperature = learn_temperature

        # Initialiser for the a learnable alpha temperature
        self.temperature_initializer = WeightInitializer(self.initial_temperature)

    def build(self, input_shape):
        self.temperature = self.add_weight(
            "temperature",
            shape=(),
            dtype=tf.float32,
            initializer=self.temperature_initializer,
            trainable=self.learn_temperature,
        )
        self.built = True

    def call(self, theta, **kwargs):
        if self.xform == "softplus":
            alpha = activations.softplus(theta)

        elif self.xform == "softmax":
            alpha = activations.softmax(theta / self.temperature, axis=2)

        elif self.xform == "gumbel-softmax":
            gumbel_softmax_distribution = tfp.distributions.RelaxedOneHotCategorical(
                temperature=self.temperature,
                logits=theta,
            )
            alpha = gumbel_softmax_distribution.sample()

        return alpha


class MeanVectorsLayer(layers.Layer):
    """Layer to learn a set of mean vectors.

    The vectors are free parameters.

    Parameters
    ----------
    n : int
        Number of vectors.
    m : int
        Number of elements.
    learn : bool
        Should we learn the vectors?
    initial_value : np.ndarray
        Initial value for the vectors.
    """

    def __init__(
        self,
        n: int,
        m: int,
        learn: bool,
        initial_value: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n
        self.m = m
        self.learn = learn
        if initial_value is None:
            self.initial_value = np.zeros([n, m], dtype=np.float32)
        else:
            self.initial_value = initial_value.astype("float32")
        self.vectors_initializer = WeightInitializer(self.initial_value)

    def build(self, input_shape):
        self.vectors = self.add_weight(
            "vectors",
            shape=(self.n, self.m),
            dtype=tf.float32,
            initializer=self.vectors_initializer,
            trainable=self.learn,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        return self.vectors


class CovarianceMatricesLayer(layers.Layer):
    """Layer to learn a set of covariance matrices.

    A cholesky factor is learnt and used to calculate a covariance matrix as
    C = LL^T, where L is the cholesky factor. The cholesky factor is learnt as
    a vector of free parameters.

    Parameters
    ----------
    n : int
        Number of matrices.
    m : int
        Number of rows/columns.
    learn : bool
        Should the matrices be learnable?
    initial_value : np.ndarray
        Initial values for the matrices.
    """

    def __init__(
        self,
        n: int,
        m: int,
        learn: bool,
        initial_value: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n
        self.m = m
        self.learn = learn

        # Bijector used to transform learnable vectors to covariance matrices
        self.bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])

        # Initialisation of matrices
        if initial_value is None:
            self.initial_value = np.stack([np.eye(m, dtype=np.float32)] * n)
        else:
            self.initial_value = initial_value.astype("float32")
        self.initial_flattened_cholesky_factors = self.bijector.inverse(
            self.initial_value
        )
        self.flattened_cholesky_factors_initializer = WeightInitializer(
            self.initial_flattened_cholesky_factors
        )

    def build(self, input_shape):
        self.flattened_cholesky_factors = self.add_weight(
            "flattened_cholesky_factors",
            shape=(self.n, self.m * (self.m + 1) // 2),
            dtype=tf.float32,
            initializer=self.flattened_cholesky_factors_initializer,
            trainable=self.learn,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        return self.bijector(self.flattened_cholesky_factors)


class CorrelationMatricesLayer(layers.Layer):
    """Layer to learn a set of correlation matrices.

    A cholesky factor is learnt as a vector of free parameters and used to
    calculate a correlation matrix.

    Parameters
    ----------
    n : int
        Number of matrices.
    m : int
        Number of rows/columns.
    learn : bool
        Should the matrices be learnable?
    initial_value : np.ndarray
        Initial values for the matrices.
    """

    def __init__(
        self,
        n: int,
        m: int,
        learn: bool,
        initial_value: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n
        self.m = m
        self.learn = learn

        # Bijector used to transform learnable vectors to correlation matrices
        self.bijector = tfb.Chain(
            [tfb.CholeskyOuterProduct(), tfb.CorrelationCholesky()]
        )

        # Initialisation of matrices
        if initial_value is None:
            self.initial_value = np.stack([np.eye(m, dtype=np.float32)] * n)
        else:
            self.initial_value = initial_value.astype("float32")
        self.initial_flattened_cholesky_factors = self.bijector.inverse(
            self.initial_value
        )
        self.flattened_cholesky_factors_initializer = WeightInitializer(
            self.initial_flattened_cholesky_factors
        )

    def build(self, input_shape):
        self.flattened_cholesky_factors = self.add_weight(
            "flattened_cholesky_factors",
            shape=(self.n, self.m * (self.m - 1) // 2),
            dtype=tf.float32,
            initializer=self.flattened_cholesky_factors_initializer,
            trainable=self.learn,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        return self.bijector(self.flattened_cholesky_factors)


class DiagonalMatricesLayer(layers.Layer):
    """Layer to learn a set of diagonal matrices.

    The diagonal is forced to be positive using a softplus transformation.

    Parameters
    ----------
    n : int
        Number of matrices.
    m : int
        Number of rows/columns.
    learn : bool
        Should the matrices be learnable?
    initial_value : np.ndarray
        Initial values for the matrices.
    """

    def __init__(
        self,
        n: int,
        m: int,
        learn: bool,
        initial_value: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n
        self.m = m
        self.learn = learn

        # Softplus transformation to ensure diagonal is positive
        self.bijector = tfb.Softplus()

        # Initialisation for the diagonals
        if initial_value is None:
            self.initial_value = np.ones([n, m], dtype=np.float32)
        else:
            self.initial_value = initial_value.astype("float32")
        self.initial_diagonals = self.bijector.inverse(self.initial_value)
        self.diagonals_initializer = WeightInitializer(self.initial_diagonals)

    def build(self, input_shape):
        self.diagonals = self.add_weight(
            "diagonals",
            shape=(self.n, self.m),
            dtype=tf.float32,
            initializer=self.diagonals_initializer,
            trainable=self.learn,
        )
        self.built = True

    def call(self, alpha, **kwargs):
        D = self.bijector(self.diagonals)
        D = tf.linalg.diag(D)
        return D


class MixVectorsLayer(layers.Layer):
    """Mix a set of vectors.

    The mixture is calculated as m_t = Sum_j alpha_jt mu_j,
    where mu_j are the vectors and alpha_jt are mixing coefficients.
    """

    def call(self, inputs, **kwargs):

        # Unpack the inputs:
        # - alpha.shape = (None, sequence_length, n_modes)
        # - mu.shape    = (n_modes, n_channels)
        alpha, mu = inputs

        # Calculate the mixture: m_t = Sum_j alpha_jt mu_j
        alpha = tf.expand_dims(alpha, axis=-1)
        mu = tf.expand_dims(tf.expand_dims(mu, axis=0), axis=0)
        m = tf.reduce_sum(tf.multiply(alpha, mu), axis=2)

        return m


class MixMatricesLayer(layers.Layer):
    """Layer to mix matrices.

    The mixture is calculated as C_t = Sum_j alpha_jt D_j,
    where D_j are the matrices and alpha_jt are mixing coefficients.
    """

    def call(self, inputs, **kwargs):

        # Unpack the inputs:
        # - alpha.shape = (None, sequence_length, n_modes)
        # - D.shape     = (n_modes, n_channels, n_channels)
        alpha, D = inputs

        # Calculate the mixture: C_t = Sum_j alpha_jt D_j
        alpha = tf.expand_dims(tf.expand_dims(alpha, axis=-1), axis=-1)
        D = tf.expand_dims(tf.expand_dims(D, axis=0), axis=0)
        C = tf.reduce_sum(tf.multiply(alpha, D), axis=2)

        return C


class LogLikelihoodLossLayer(layers.Layer):
    """Layer to calculate the negative log likelihood.

    The negative log-likelihood is calculated assuming a multivariate normal
    probability density and its value is added to the loss function.

    Parameters
    ----------
    diag_cov : bool
        Are the covariances diagonal?
    """

    def __init__(self, diag_cov: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.diag_cov = diag_cov

    def call(self, inputs):
        x, mu, sigma = inputs

        # Multivariate normal distribution
        if self.diag_cov:
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

        # Calculate the log-likelihood
        ll_loss = mvn.log_prob(x)

        # Sum over time dimension and average over the batch dimension
        ll_loss = tf.reduce_sum(ll_loss, axis=1)
        ll_loss = tf.reduce_mean(ll_loss, axis=0)

        # Add the negative log-likelihood to the loss
        nll_loss = -ll_loss
        self.add_loss(nll_loss)
        self.add_metric(nll_loss, name=self.name)

        return nll_loss


class KLDivergenceLayer(layers.Layer):
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

        return kl_loss


class KLLossLayer(layers.Layer):
    """Layer to calculate the KL loss.

    This layer sums KL divergences if multiple values as passed, applies an
    annealing factor and adds the value to the loss function.

    Parameters
    ----------
    do_annealing : bool
        Should we perform KL annealing?
    """

    def __init__(self, do_annealing: bool, **kwargs):
        super().__init__(**kwargs)
        if do_annealing:
            self.annealing_factor = tf.Variable(0.0, trainable=False)
        else:
            self.annealing_factor = tf.Variable(1.0, trainable=False)

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            # Sum KL divergences
            inputs = tf.add_n(inputs)

        # KL annealing
        kl_loss = tf.multiply(inputs, self.annealing_factor)

        # Add to loss
        self.add_loss(kl_loss)
        self.add_metric(kl_loss, name=self.name)

        return kl_loss


class InferenceRNNLayer(layers.Layer):
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
    drop_rate : float
        Dropout rate for the output of each layer.
    """

    def __init__(
        self,
        rnn_type: str,
        norm_type: str,
        act_type: str,
        n_layers: int,
        n_units: int,
        drop_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

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
            self.layers.append(layers.Dropout(drop_rate))

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs


class ModelRNNLayer(layers.Layer):
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
    drop_rate : float
        Dropout rate for the output of each layer.
    """

    def __init__(
        self,
        rnn_type: str,
        norm_type: str,
        act_type: str,
        n_layers: int,
        n_units: int,
        drop_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layers = []
        for n in range(n_layers):
            self.layers.append(
                RNNLayer(rnn_type, n_units, return_sequences=True, stateful=False)
            )
            self.layers.append(NormalizationLayer(norm_type))
            self.layers.append(layers.Activation(act_type))
            self.layers.append(layers.Dropout(drop_rate))

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs


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

    def call(self, inputs, **kwargs):
        out = self.causal_conv_layer(inputs)
        skips = []
        for layer in self.residual_block_layers:
            out, skip = layer(out)
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
        self.filter_layer = layers.Conv1D(
            filters,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding="causal",
            use_bias=False,
            kernel_regularizer=regularizers.l2(),
        )
        self.gate_layer = layers.Conv1D(
            filters,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding="causal",
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

    def call(self, inputs, **kwargs):
        filter_ = self.filter_layer(inputs)
        gate = self.gate_layer(inputs)
        z = tf.tanh(filter_) * tf.sigmoid(gate)
        residual = self.res_layer(z)
        skip = self.skip_layer(z)
        return inputs + residual, skip


class MultiLayerPerceptronLayer(layers.Layer):
    """Multi-layer perceptron.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    n_units : int
        Number of units/neurons per layer.
    norm_type : str
        Either 'layer', 'batch' or None.
    act_type : 'str'
        Activation type, e.g. 'relu', 'elu', etc.
    drop_rate : float
        Dropout rate for the output of each layer.
    """

    def __init__(
        self,
        n_layers: int,
        n_units: int,
        norm_type: str,
        act_type: str,
        drop_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layers = []
        for n in range(n_layers):
            self.layers.append(layers.Dense(n_units))
            self.layers.append(NormalizationLayer(norm_type))
            self.layers.append(layers.Activation(act_type))
            self.layers.append(layers.Dropout(drop_rate))

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs
