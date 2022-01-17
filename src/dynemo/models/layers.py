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
        **kwargs,
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
        **kwargs,
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


class VectorsLayer(layers.Layer):
    """Layer to learn vectors.

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


class MatricesLayer(layers.Layer):
    """Layer to learn a set of matrices.

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
    diag_only : bool
        Should we learn diagonal matrices?
    """

    def __init__(
        self,
        n: int,
        m: int,
        learn: bool,
        initial_value: np.ndarray,
        diag_only: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n = n
        self.m = m
        self.diag_only = diag_only
        self.learn = learn

        if diag_only:
            # Learn a vector for the diagonal of each matrix

            # Initialisation of matrices
            if initial_value is None:
                self.initial_value = np.ones([n, m], dtype=np.float32)
            else:
                self.initial_value = initial_value.astype("float32")
            self.matrices_initializer = WeightIniitalizer(self.initial_value)

        else:
            # Learn the cholesky factor of the full matrix

            # Bijector used to transform covariance matrices to a learnable vector
            self.bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])

            # Initialisation of matrices
            if initial_value is None:
                self.initial_value = np.stack([np.eye(m, dtype=np.float32)] * n)
            else:
                self.initial_value = initial_value.astype("float32")
            self.initial_flattened_cholesky_matrices = self.bijector.inverse(
                self.initial_value
            )
            self.flattened_cholesky_matrices_initializer = WeightInitializer(
                self.initial_flattened_cholesky_matrices
            )

    def build(self, input_shape):

        if self.diag_only:
            # Create weights for the diagonal of the matrices
            self.matrices = self.add_weight(
                "matrices",
                shape=(self.n, self.m),
                dtype=tf.float32,
                initializer=self.matrices_initializer,
                trainable=self.learn,
            )

        else:
            # Create weights for the cholesky decomposition of the matrices
            self.flattened_cholesky_matrices = self.add_weight(
                "flattened_cholesky_matrices",
                shape=(self.n, self.m * (self.m + 1) // 2),
                dtype=tf.float32,
                initializer=self.flattened_cholesky_matrices_initializer,
                trainable=self.learn,
            )

        self.built = True

    def call(self, alpha, **kwargs):

        if self.diag_only:
            # Make sure diagonal is positive and make a m by m tensor
            D = activations.softplus(self.matrices)
            D = tf.matrix_diagonal(D)

        else:
            # Calculate the covariance matrix from the cholesky factor
            D = self.bijector(self.flattened_cholesky_matrices)

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

        # Reshape for multiplication
        alpha = tf.expand_dims(alpha, axis=-1)
        mu = tf.expand_dims(tf.expand_dims(mu, axis=0), axis=0)

        # Calculate the mean: m_t = Sum_j alpha_jt mu_j
        m = tf.reduce_sum(tf.multiply(alpha, mu), axis=2)

        return m


class MixMatricesLayer(layers.Layer):
    """Layer to mix matrices.

    The mixture is calculated as C_t = Sum_j alpha_jt D_j,
    where D_j are the matrices and alpha_jt are mixing coefficients.

    Parameters
    ----------
    shift : bool
        Do the mixing coefficients correspond to one time step in the future?
        If so, we roll the mixed matrices.
    """

    def __init__(self, shift: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift

    def call(self, inputs, **kwargs):

        # Unpack the inputs:
        # - alpha.shape = (None, sequence_length, n_modes)
        # - D.shape     = (n_modes, n_channels, n_channels)
        alpha, D = inputs

        # Reshape alpha and D for multiplication
        alpha = tf.expand_dims(tf.expand_dims(alpha, axis=-1), axis=-1)
        D = tf.expand_dims(tf.expand_dims(D, axis=0), axis=0)

        # Calculate the covariance: C_t = Sum_j alpha_jt D_j
        C = tf.reduce_sum(tf.multiply(alpha, D), axis=2)

        if self.shift:
            # The means are predicted one time step in the future so we need to
            # make sure the covariances correspond to the correct time point
            C = tf.roll(C, shift=-1, axis=1)

        return C


class LogLikelihoodLossLayer(layers.Layer):
    """Layer to calculate the negative log likelihood.

    The negative log-likelihood is calculated assuming a multivariate normal
    probability density and its value is added to the loss function.

    Parameters
    ----------
    clip : int
        Number of data points to clip from the means and data.
        Optional, default is no clipping.
    """

    def __init__(self, clip: int = None, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip

    def call(self, inputs):
        x, mu, sigma = inputs

        # Clip data, means and covariances
        # This is neccessary if mu and sigma are one time step in the future
        if self.clip is not None:
            x = x[:, self.clip :]
            mu = mu[:, : -self.clip]
            sigma = sigma[:, : -self.clip]

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
        **kwargs,
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
        **kwargs,
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


class MeansStdsFcsLayer(layers.Layer):
    """Layer to learn the means, diagonal standard deviation matrices,
    and functional connectivities of modes.

    Outputs the mean vector, the standard deviations, and correlation (fc) matrix
    of each mode.

    Parameters
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
    regularize_fcs : bool
        Should we add an L2 loss for the cholesky factors of the functional
        connectivities.
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
        regularize_fcs: bool,
        **kwargs,
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

        # Regulariser for the cholesky factor of the functional connectivities
        if regularize_fcs:
            self.flattened_cholesky_fcs_regularizer = regularizers.l2()
        else:
            self.flattened_cholesky_fcs_regularizer = None

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
            regularizer=self.flattened_cholesky_fcs_regularizer,
        )

        self.built = True

    def call(self, inputs, **kwargs):

        # Make sure standard deviations are positive
        stds = activations.softplus(self.stds)

        # Calculate functional connectivity matrix from flattened vector
        fcs = self.bijector(self.flattened_cholesky_fcs)

        # Normalise so that FCs are correlation matrices
        # sd = tf.expand_dims(tf.sqrt(tf.linalg.diag_part(fcs)), axis=-1)
        # fcs /= tf.matmul(sd, sd, transpose_b=True)

        return [self.means, stds, fcs]


class MixMeansStdsFcsLayer(layers.Layer):
    """Compute a probabilistic mixture of means, standard deviations and functional
    connectivies.

    The mixture is calculated as:

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


class FillConstantLayer(layers.Layer):
    """Layer to create tensor with the same shape of the input,
    but filled with a constant.

    Parameters
    ----------
    constant : float
        Value to full tensor with.
    """

    def __init__(self, constant: float, **kwargs):
        super().__init__(**kwargs)
        self.constant = constant

    def call(self, inputs, **kwargs):
        return tf.fill(tf.shape(inputs), self.constant)
