"""Custom Tensorflow layers.

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import activations, layers
from osl_dynamics.inference.initializers import WeightInitializer

tfb = tfp.bijectors


def NormalizationLayer(norm_type, *args, **kwargs):
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


def RNNLayer(rnn_type, *args, **kwargs):
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


class TFRangeLayer(layers.Layer):
    """Wrapper for tf.range.

    Parameters
    ----------
    limit : int
        Upper limit for range.
    """

    def __init__(self, limit, **kwargs):
        super().__init__(**kwargs)
        self.limit = limit

    def call(self, inputs):
        return tf.range(self.limit)


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


class SampleGumbelSoftmaxDistributionLayer(layers.Layer):
    """Layer for sampling from a Gumbel-Softmax distribution."""

    def call(self, inputs, **kwargs):
        gs = tfp.distributions.RelaxedOneHotCategorical(temperature=0.05, logits=inputs)
        return gs.sample()


class SoftmaxLayer(layers.Layer):
    """Layer for applying a softmax activation function.

    Parameters
    ----------
    initial_temperature : float
        Temperature parameter.
    learn_temperature : bool
        Should we learn the alpha temperature?
    """

    def __init__(
        self,
        initial_temperature,
        learn_temperature,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.initial_temperature = initial_temperature
        self.learn_temperature = learn_temperature
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

    def call(self, inputs, **kwargs):
        return activations.softmax(inputs / self.temperature, axis=2)


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
        n,
        m,
        learn,
        initial_value,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n
        self.m = m
        self.learn = learn
        if initial_value is None:
            self.initial_value = np.zeros([n, m], dtype=np.float32)
        else:
            if initial_value.ndim != 2:
                raise ValueError(
                    "a (n_modes, n_channels) array must be passed for initial_means."
                )
            if initial_value.shape[-1] != m:
                raise ValueError(
                    f"mismatch between the number of channels ({m}) and number of "
                    + f"elements in initial_means ({initial_value.shape[-1]})."
                )
            if initial_value.shape[0] != n:
                raise ValueError(
                    f"mismatch bettwen the number of modes and vectors in "
                    + f"initial_means ({initial_value.shape[0]})."
                )
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
        n,
        m,
        learn,
        initial_value,
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
            if initial_value.ndim != 3:
                raise ValueError(
                    "a (n_modes, n_channels, n_channels) array must be passed for "
                    + "initial_covariances."
                )
            if initial_value.shape[-1] != m:
                raise ValueError(
                    f"mismatch between the number of channels ({m}) and number of "
                    + "rows/columns in initial_covariances "
                    + f"({initial_value.shape[-1]})."
                )
            if initial_value.shape[0] != n:
                raise ValueError(
                    f"mismatch bettwen the number of modes and matrices in "
                    + f"initial_covariances ({initial_value.shape[0]})."
                )
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
        n,
        m,
        learn,
        initial_value,
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
            if initial_value.ndim != 3:
                raise ValueError(
                    "a (n_modes, n_channels, n_channels) array must be passed for "
                    + "initial_fcs."
                )
            if initial_value.shape[-1] != m:
                raise ValueError(
                    f"mismatch between the number of channels ({m}) and number of "
                    + "rows/columns in initial_fcs "
                    + f"({initial_value.shape[-1]})."
                )
            if initial_value.shape[0] != n:
                raise ValueError(
                    f"mismatch bettwen the number of modes and matrices in "
                    + f"initial_fcs ({initial_value.shape[0]})."
                )
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
        n,
        m,
        learn,
        initial_value,
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
            if initial_value.ndim != 2:
                raise ValueError(
                    "a (n_modes, n_channels) array must be passed for initial_value."
                )
            if initial_value.shape[-1] != m:
                raise ValueError(
                    f"mismatch between the number of channels ({m}) and number of "
                    + f"elements in initial_value ({initial_value.shape[-1]})."
                )
            if initial_value.shape[0] != n:
                raise ValueError(
                    f"mismatch bettwen the number of modes and vectors in "
                    + f"initial_value ({initial_value.shape[0]})."
                )
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


class MatrixLayer(layers.Layer):
    """Layer to learn a matrix.

    Parameters
    ----------
    m : int
        Number of rows/columns.
    constraint : str
        Either 'covariance' or 'diagonal'.
    learn : bool
        Should the matrix be trainable?
    initial_value : np.ndarray
        Initial value for the matrix.
    """

    def __init__(self, m, constraint, learn, initial_value, **kwargs):

        super().__init__(**kwargs)
        self.m = m
        self.constraint = constraint
        self.learn = learn

        self.initial_value = initial_value
        if initial_value is not None:
            if initial_value.ndim != 2:
                raise ValueError("A 2D numpy array must be passed for initial_value.")
            if initial_value.shape[0] != m:
                raise ValueError(
                    "Number of rows/columns in initial_value does not match m."
                )
            initial_value = initial_value[np.newaxis, ...]

        if constraint == "covariance":
            self.matrix_layer = CovarianceMatricesLayer(1, m, learn, initial_value)
        elif constraint == "diagonal":
            self.matrix_layer = DiagonalMatricesLayer(1, m, learn, initial_value)
        else:
            raise ValueError("Please use constraint='diagonal' or 'covariance.'")

    def call(self, inputs, **kwargs):
        return self.matrix_layer(inputs)[0]


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


class MixVectorsMatricesLayer(layers.Layer):
    """Layer to mix vectors and matrices.

    The mixture is calculated as C_u_t.
    """

    def call(self, inputs, **kwargs):

        # Unpack the inputs:
        m, C = inputs
        m = tf.expand_dims(m, axis=-1)
        C_m = tf.concat([C, m], axis=3)

        return C_m


class LogLikelihoodLossLayer(layers.Layer):
    """Layer to calculate the negative log likelihood.

    The negative log-likelihood is calculated assuming a multivariate normal
    probability density and its value is added to the loss function.
    """

    def call(self, inputs):
        x, mu, sigma = inputs

        # Multivariate normal distribution
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

        return tf.expand_dims(nll_loss, axis=-1)


class AdversarialLogLikelihoodLossLayer(layers.Layer):
    """Layer to calculate the negative log likelihood.

    The negative log-likelihood is calculated assuming a multivariate normal
    probability density and its value is added to the loss function.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    """

    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.__name__ = self._name  # needed to fix error

    def call(self, y_true, y_pred):
        sigma = y_pred[:, :, :, : self.n_channels]
        mu = y_pred[:, :, :, self.n_channels]

        # Multivariate normal distribution
        mvn = tfp.distributions.MultivariateNormalTriL(
            loc=mu,
            scale_tril=tf.linalg.cholesky(sigma),
            allow_nan_stats=False,
        )

        # Calculate the log-likelihood
        ll_loss = mvn.log_prob(y_true)

        # Sum over time dimension and average over the batch dimension
        ll_loss = tf.reduce_sum(ll_loss, axis=1)
        ll_loss = tf.reduce_mean(ll_loss, axis=0)

        return -ll_loss


class KLDivergenceLayer(layers.Layer):
    """Layer to calculate a KL divergence between two Normal distributions.

    Parameters
    ----------
    clip_start : int
        Index to clip the sequences inputted to this layer.
    """

    def __init__(self, clip_start=0, **kwargs):
        super().__init__(**kwargs)
        self.clip_start = clip_start

    def call(self, inputs, **kwargs):
        inference_mu, inference_sigma, model_mu, model_sigma = inputs

        # The model network predicts one time step into the future compared to
        # the inference network. We clip the sequences to ensure we are comparing
        # the correct time points.
        model_mu = model_mu[:, self.clip_start : -1]
        model_sigma = model_sigma[:, self.clip_start : -1]

        inference_mu = inference_mu[:, self.clip_start + 1 :]
        inference_sigma = inference_sigma[:, self.clip_start + 1 :]

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

    def __init__(self, do_annealing, **kwargs):
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

        return tf.expand_dims(kl_loss, axis=-1)


class InferenceRNNLayer(layers.Layer):
    """RNN inference network.

    Parameters
    ----------
    rnn_type : str
        Either 'lstm' or 'gru'.
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
        rnn_type,
        norm_type,
        act_type,
        n_layers,
        n_units,
        drop_rate,
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
        Either 'lstm' or 'gru'.
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
        rnn_type,
        norm_type,
        act_type,
        n_layers,
        n_units,
        drop_rate,
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


class CategoricalKLDivergenceLayer(layers.Layer):
    """Layer to calculate a KL divergence between two categorical distributions.

    Parameters
    ----------
    clip_start : int
        Index to clip the sequences inputted to this layer.
    """

    def __init__(self, clip_start=0, **kwargs):
        super().__init__(**kwargs)
        self.clip_start = clip_start

    def call(self, inputs, **kwargs):
        inference_logits, model_logits = inputs

        # The model network predicts one time step into the future compared to
        # the inference network. We clip the sequences to ensure we are comparing
        # the correct time points.
        model_logits = model_logits[:, self.clip_start : -1]
        inference_logits = inference_logits[:, self.clip_start + 1 :]

        # Calculate the KL divergence between the posterior and prior
        prior = tfp.distributions.Categorical(logits=model_logits)
        posterior = tfp.distributions.Categorical(logits=inference_logits)
        kl_loss = tfp.distributions.kl_divergence(
            posterior, prior, allow_nan_stats=False
        )

        # Sum the KL loss for each time point and average over batches
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss = tf.reduce_mean(kl_loss, axis=0)

        return kl_loss


class CategoricalLogLikelihoodLossLayer(layers.Layer):
    """Layer to calculate the log-likelihood loss assuming a categorical model.

    Parameters
    ----------
    n_states : int
        Number of states
    """

    def __init__(self, n_states, **kwargs):
        super().__init__(**kwargs)
        self.n_states = n_states

    def call(self, inputs, **kwargs):
        x, mu, sigma, probs = inputs

        # Log-likelihood for each state
        ll_loss = tf.zeros(shape=tf.shape(x)[:-1])
        for i in range(self.n_states):
            mvn = tfp.distributions.MultivariateNormalTriL(
                loc=mu[i],
                scale_tril=tf.linalg.cholesky(sigma[i]),
                allow_nan_stats=False,
            )
            ll_loss += probs[:, :, i] * mvn.log_prob(x)

        # Sum over time dimension and average over the batch dimension
        ll_loss = tf.reduce_sum(ll_loss, axis=1)
        ll_loss = tf.reduce_mean(ll_loss, axis=0)

        # Add the negative log-likelihood to the loss
        nll_loss = -ll_loss
        self.add_loss(nll_loss)
        self.add_metric(nll_loss, name=self.name)

        return tf.expand_dims(nll_loss, axis=-1)
