"""Custom Tensorflow layers.

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import activations, layers, initializers

import osl_dynamics.inference.initializers as osld_initializers

tfb = tfp.bijectors


@tf.function
def add_epsilon(A, epsilon, diag=False):
    """Adds epsilon the the diagonal of batches of square matrices
    or all elements of matrices.

    Parameters
    ----------
    A : tf.Tensor
        Batches of square matrices or vectors.
        Shape is (..., N, N) or (..., N).
    epsilon : float
        Small error added to the diagonal of the matrices or every element
        of the vectors.
    diag : bool
        Do we want to add epsilon to the diagonal only?
    """
    epsilon = tf.cast(epsilon, dtype=tf.float32)
    A_shape = tf.shape(A)
    if diag:
        # Add epsilon to the diagonal only
        I = tf.eye(A_shape[-1])
    else:
        # Add epsilon to all elements
        I = 1.0
    return A + epsilon * I


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


class AddRegularizationLossLayer(layers.Layer):
    """Adds a regularization loss.

    Can be used as a wrapper for a keras regularizer. Inputs are used to
    calculate the regularization loss and returned without modification.

    Parameters
    ----------
    reg : str
        Type of regularization.
    strength : float
        Strength of regularization. The regularization is multiplied
        by the strength before being added to the loss
    """

    def __init__(self, reg, strength, **kwargs):
        super().__init__(**kwargs)
        self.reg = tf.keras.regularizers.get(reg)
        self.strength = strength

    def call(self, inputs, **kwargs):
        reg_loss = self.reg(inputs)
        self.add_loss(self.strength * reg_loss)
        return inputs


class ConcatenateLayer(layers.Layer):
    """Concatenates a set of tensors.

    Wrapper for tf.concat().

    Parameters
    ----------
    axis : int
        Axis to concatenate along.
    """

    def __init__(self, axis, **kwargs):
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


class ZeroLayer(layers.Layer):
    """Layer that outputs tensor of zeros.

    Wrapper for tf.zeros(). Note, the inputs to this layer are not used.

    Parameters
    ----------
    shape : tuple
        Shape of the zero tensor.
    """

    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def call(self, inputs):
        return tf.zeros(self.shape)


class InverseCholeskyLayer(layers.Layer):
    """Layer for getting Cholesky vectors from postive definite symmetric matrices.

    Parameters
    ----------
    epsilon : float
        Small error added to the diagonal of the matrices.
    """

    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])

    def call(self, inputs):
        inputs = add_epsilon(inputs, self.epsilon, diag=True)
        return self.bijector.inverse(inputs)


class SampleGammaDistributionLayer(layers.Layer):
    """Layer for sampling from a gamma distribution.

    This layer accepts the shape and rate
    and outputs samples from a gamma distribution.

    Parameters
    ----------
    epsilon : float
        Error to add to the shape and rate for numerical stability.
    """

    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs, training=None, **kwargs):
        alpha, beta = inputs
        alpha = add_epsilon(alpha, self.epsilon)
        beta = add_epsilon(beta, self.epsilon)
        if training:
            N = tfp.distributions.Gamma(
                concentration=alpha, rate=beta, allow_nan_stats=False
            )
            return N.sample()
        else:
            mode = (alpha - 1) / beta
            return tf.maximum(mode, 0)


class SampleNormalDistributionLayer(layers.Layer):
    """Layer for sampling from a normal distribution.

    This layer accepts the mean and the standard deviation and
    outputs samples from a normal distribution.

    Parameters
    ----------
    epsilon : float
        Error to add to the standard deviations for numerical stability.
    """

    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs, training=None, **kwargs):
        mu, sigma = inputs
        sigma = add_epsilon(sigma, self.epsilon)
        if training:
            N = tfp.distributions.Normal(loc=mu, scale=sigma)
            return N.sample()
        else:
            return mu


class SampleGumbelSoftmaxDistributionLayer(layers.Layer):
    """Layer for sampling from a Gumbel-Softmax distribution.

    Parameters
    ----------
    temperature : float
        Temperature for the Gumbel-Softmax distribution.
    """

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs, **kwargs):
        gs = tfp.distributions.RelaxedOneHotCategorical(
            temperature=self.temperature, logits=inputs
        )
        return gs.sample()


class SampleOneHotCategoricalDistributionLayer(layers.Layer):
    """Layer for sampling from a Categorical distribution."""

    def call(self, inputs, **kwargs):
        cat = tfp.distributions.OneHotCategorical(logits=inputs, dtype=tf.float32)
        return cat.sample()


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
        self.layers = [
            LearnableTensorLayer(
                shape=(),
                learn=learn_temperature,
                initial_value=initial_temperature,
                name=self.name + "_kernel",
            )
        ]

    def call(self, inputs, **kwargs):
        temperature = self.layers[0](inputs)
        return activations.softmax(inputs / temperature, axis=2)


class LearnableTensorLayer(layers.Layer):
    """Layer to learn a tensor.

    Parameters
    ----------
    shape : tuple
        Shape of the tensor.
    learn : bool
        Should we learn the tensor?
    initializer : tf.keras.initializers.Initializer
        Initializer for the tensor.
    initial_value : float
        Initial value for the tensor if initializer is not passed.
    regularizer : osl-dynamics regularizer
        Regularizer for the tensor. Must be from osl_dynamics.inference.regularizers.
    """

    def __init__(
        self,
        shape,
        learn=True,
        initializer=None,
        initial_value=None,
        regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Bool for if we're learning the tensor
        self.learn = learn

        # Shape of the tensor
        if shape is None:
            raise ValueError("A shape must be passed to LearnableTensorLayer.")
        self.shape = shape

        # Initial value of the tensor
        self.initial_value = initial_value
        if self.initial_value is not None:
            self.initial_value = np.array(initial_value).astype(np.float32)

        # Setup the tensor initializer
        if initializer is None:
            # Initializer not passed, use the initial value
            if self.initial_value is None:
                raise ValueError("initializer or initial_value must be passed.")
            elif self.initial_value.shape != shape:
                raise ValueError(
                    "Shape of initial_value must match that of the tensor. "
                    + f"Expected {shape}, got {self.initial_value.shape}."
                )
            self.tensor_initializer = osld_initializers.WeightInitializer(
                self.initial_value
            )
        else:
            # Use the initializer passed, initial_value is ignored
            self.tensor_initializer = initializer

        # Regularizer for the tensor
        # This should be a function of the tensor that returns a float
        self.regularizer = regularizer

    def add_regularization(self, tensor, inputs):
        # Calculate the regularisation from the tensor
        reg = self.regularizer(tensor)

        # Calculate the scaling factor for the regularization
        # Note, inputs.shape[0] must be the batch size
        batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)
        n_batches = self.regularizer.n_batches
        scaling_factor = batch_size * n_batches
        reg /= scaling_factor

        # Add regularization to the loss and display while training
        self.add_loss(reg)
        self.add_metric(reg, name=self.name)

    def build(self, input_shape):
        # Create a weight for the tensor
        self.tensor = self.add_weight(
            "tensor",
            shape=self.shape,
            dtype=tf.float32,
            initializer=self.tensor_initializer,
            trainable=self.learn,
        )
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        if self.regularizer is not None and training:
            self.add_regularization(self.tensor, inputs)
        return self.tensor


class VectorsLayer(layers.Layer):
    """Layer to learn a set of vectors.

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
    regularizer : tf.keras.regularizers.Regularizer
        Regularizer for vectors.
    """

    def __init__(
        self,
        n,
        m,
        learn,
        initial_value,
        regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if initial_value is not None:
            # Check initial_value is the correct shape
            if initial_value.shape != (n, m):
                raise ValueError(f"initial_value shape must be ({n}, {m}).")
            initial_value = initial_value.astype("float32")

            # We don't need an initializer
            initializer = None
        else:
            # No initial value has been passed, set the initializer
            if learn:
                initializer = initializers.TruncatedNormal(mean=0, stddev=0.02)
            else:
                initializer = initializers.Zeros()

        # We use self.layers for compatibility with
        # initializers.reinitialize_model_weights
        self.layers = [
            LearnableTensorLayer(
                shape=(n, m),
                learn=learn,
                initializer=initializer,
                initial_value=initial_value,
                regularizer=regularizer,
                name=self.name + "_kernel",
            )
        ]

    def call(self, inputs, **kwargs):
        learnable_tensors_layer = self.layers[0]
        vectors = learnable_tensors_layer(inputs, **kwargs)
        return vectors


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
    epsilon : float
        Error added to the diagonal of covariances matrices for numerical stability.
    regularizer : tf.keras.regularizers.Regularizer
        Regularizer for matrices.
    """

    def __init__(
        self,
        n,
        m,
        learn,
        initial_value,
        epsilon,
        regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon

        # Bijector used to transform learnable vectors to covariance matrices
        self.bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])

        # Do we have an initial value?
        if initial_value is not None:
            # Check it's the correct shape
            if initial_value.shape != (n, m, m):
                raise ValueError(f"initial_value shape must be ({n}, {m}, {m}).")

            # Calculate the flattened cholesky factors
            initial_value = initial_value.astype("float32")
            initial_flattened_cholesky_factors = self.bijector.inverse(initial_value)

            # We don't need an initializer
            initializer = None
        else:
            # No initial value has been passed
            initial_flattened_cholesky_factors = None
            if learn:
                # Use a random initializer
                initializer = osld_initializers.NormalIdentityCholeskyInitializer(
                    std=0.1
                )
            else:
                # Use the identity matrix for each mode/state
                initializer = osld_initializers.IdentityCholeskyInitializer()

        # Create a layer to learn the covariance matrices
        #
        # We use self.layers for compatibility with
        # initializers.reinitialize_model_weights
        self.layers = [
            LearnableTensorLayer(
                shape=(n, m * (m + 1) // 2),
                learn=learn,
                initializer=initializer,
                initial_value=initial_flattened_cholesky_factors,
                regularizer=regularizer,
                name=self.name + "_kernel",
            )
        ]

    def call(self, inputs, **kwargs):
        learnable_tensor_layer = self.layers[0]
        flattened_cholesky_factors = learnable_tensor_layer(inputs, **kwargs)
        covariances = self.bijector(flattened_cholesky_factors)
        covariances = add_epsilon(covariances, self.epsilon, diag=True)
        return covariances


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
    epsilon : float
        Error added to the diagonal of correlation matrices for numerical stability.
    regularizer : tf.keras.regularizers.Regularizer
        Regularizer for matrices.
    """

    def __init__(
        self,
        n,
        m,
        learn,
        initial_value,
        epsilon,
        regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon

        # Bijector used to transform learnable vectors to correlation matrices
        self.bijector = tfb.Chain(
            [tfb.CholeskyOuterProduct(), tfb.CorrelationCholesky()]
        )

        # Do we have an initial value?
        if initial_value is not None:
            # Check it's the correct shape
            if initial_value.shape != (n, m, m):
                raise ValueError(f"initial_value shape must be ({n}, {m}, {m}).")

            # Calculate the flattened cholesky factors
            initial_value = initial_value.astype("float32")
            initial_flattened_cholesky_factors = self.bijector.inverse(initial_value)

            # We don't need an initializer
            initializer = None
        else:
            # No initial value has been passed
            initial_flattened_cholesky_factors = None
            if learn:
                # Use a random initializer
                initializer = osld_initializers.NormalCorrelationCholeskyInitializer(
                    std=0.1
                )
            else:
                # Use the identity matrix for each mode/state
                initializer = osld_initializers.IdentityCholeskyInitializer()

        # Create a layer to learn the correlation matrices
        #
        # We use self.layers for compatibility with
        # initializers.reinitialize_model_weights
        self.layers = [
            LearnableTensorLayer(
                shape=(n, m * (m - 1) // 2),
                learn=learn,
                initializer=initializer,
                initial_value=initial_flattened_cholesky_factors,
                regularizer=regularizer,
                name=self.name + "_kernel",
            )
        ]

    def call(self, inputs, **kwargs):
        learnable_tensor_layer = self.layers[0]
        flattened_cholesky_factors = learnable_tensor_layer(inputs, **kwargs)
        correlations = self.bijector(flattened_cholesky_factors)
        correlations = add_epsilon(correlations, self.epsilon, diag=True)
        return correlations


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
    epsilon : float
        Error added to the diagonal matrices for numerical stability.
    regularizer : tf.keras.regularizers.Regularizer
        Regularizer for the diagonal entries.
    """

    def __init__(
        self,
        n,
        m,
        learn,
        initial_value,
        epsilon,
        regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon

        # Softplus transformation to ensure diagonal is positive
        self.bijector = tfb.Softplus()

        # Do we have an initial value?
        if initial_value is not None:
            # Check it's the correct shape
            if initial_value.shape == (n, m, m):
                # Keep the diagonal only
                initial_value = np.diagonal(initial_value, axis1=1, axis2=2)
            elif initial_value.shape != (n, m):
                raise ValueError(
                    f"initial_value shape must be ({n}, {m}, {m}) or ({n}, {m})."
                )

            # Calculate the initial value of the learnable tensor
            initial_value = initial_value.astype("float32")
            initial_diagonals = self.bijector.inverse(initial_value)

            # We don't need an initializer
            initializer = None
        else:
            # No initial value has been passed
            initial_diagonals = None
            if learn:
                # Use a random initializer
                initializer = osld_initializers.NormalDiagonalInitializer(std=0.05)
            else:
                # Use the identity matrix for each mode/state
                initializer = osld_initializers.IdentityCholeskyInitializer()

        # Create a layer to learn the matrices
        #
        # We use self.layers for compatibility with
        # initializers.reinitialize_model_weights
        self.layers = [
            LearnableTensorLayer(
                shape=(n, m),
                learn=learn,
                initializer=initializer,
                initial_value=initial_diagonals,
                regularizer=regularizer,
                name=self.name + "_kernel",
            )
        ]

    def call(self, inputs, **kwargs):
        learnable_tensor_layer = self.layers[0]
        diagonals = learnable_tensor_layer(inputs, **kwargs)
        diagonals = self.bijector(diagonals)
        diagonals = add_epsilon(diagonals, self.epsilon)
        return tf.linalg.diag(diagonals)


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
    epsilon : float
        Error added to the matrices for numerical stability.
    regularizer : tf.keras.regularizers.Regularizer
        Regularizer for the diagonal entries.
    """

    def __init__(
        self, m, constraint, learn, initial_value, epsilon, regularizer=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.constraint = constraint

        if initial_value is not None:
            if initial_value.shape[-1] != m:
                raise ValueError(
                    "Number of rows/columns in initial_value does not match m."
                )
            initial_value = initial_value[np.newaxis, ...]

        if constraint == "covariance":
            self.matrix_layer = CovarianceMatricesLayer(
                1, m, learn, initial_value, epsilon, regularizer
            )
        elif constraint == "diagonal":
            self.matrix_layer = DiagonalMatricesLayer(
                1, m, learn, initial_value, epsilon, regularizer
            )
        else:
            raise ValueError("Please use constraint='diagonal' or 'covariance.'")

        self.layers = [self.matrix_layer]

    def call(self, inputs, **kwargs):
        return self.matrix_layer(inputs, **kwargs)[0]


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


class ConcatVectorsMatricesLayer(layers.Layer):
    """Layer to concatenate vectors and matrices."""

    def call(self, inputs, **kwargs):
        m, C = inputs
        m = tf.expand_dims(m, axis=-1)
        C_m = tf.concat([C, m], axis=3)
        return C_m


class LogLikelihoodLossLayer(layers.Layer):
    """Layer to calculate the negative log likelihood.

    The negative log-likelihood is calculated assuming a multivariate normal
    probability density and its value is added to the loss function.

    Parameters
    ----------
    epsilon : float
        Error added to the covariance matrices for numerical stability.
    """

    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        x, mu, sigma = inputs

        # Add a small error for numerical stability
        sigma = add_epsilon(sigma, self.epsilon, diag=True)

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
    epsilon : float
        Error added to the standard deviations for numerical stability.
    clip_start : int
        Index to clip the sequences inputted to this layer.
    """

    def __init__(self, epsilon, clip_start=0, **kwargs):
        super().__init__(**kwargs)
        self.clip_start = clip_start
        self.epsilon = epsilon

    def call(self, inputs, **kwargs):
        inference_mu, inference_sigma, model_mu, model_sigma = inputs

        # Add a small error for numerical stability
        inference_sigma = add_epsilon(inference_sigma, self.epsilon)
        model_sigma = add_epsilon(model_sigma, self.epsilon)

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
    reg : str
        Regularization for each layer.
    """

    def __init__(
        self,
        rnn_type,
        norm_type,
        act_type,
        n_layers,
        n_units,
        drop_rate,
        reg,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers = []
        for n in range(n_layers):
            self.layers.append(
                layers.Bidirectional(
                    layer=RNNLayer(
                        rnn_type,
                        n_units,
                        kernel_regularizer=reg,
                        recurrent_regularizer=reg,
                        bias_regularizer=reg,
                        activity_regularizer=reg,
                        return_sequences=True,
                        stateful=False,
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
    reg : str
        Regularization for each layer.
    """

    def __init__(
        self,
        rnn_type,
        norm_type,
        act_type,
        n_layers,
        n_units,
        drop_rate,
        reg,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers = []
        for n in range(n_layers):
            self.layers.append(
                RNNLayer(
                    rnn_type,
                    n_units,
                    kernel_regularizer=reg,
                    recurrent_regularizer=reg,
                    bias_regularizer=reg,
                    activity_regularizer=reg,
                    return_sequences=True,
                    stateful=False,
                )
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
        Number of states.
    epsilon : float
        Error added to the covariances for numerical stability.
    """

    def __init__(self, n_states, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.epsilon = epsilon

    def call(self, inputs, **kwargs):
        x, mu, sigma, probs = inputs

        # Add a small error for numerical stability
        sigma = add_epsilon(sigma, self.epsilon, diag=True)

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


class ConcatEmbeddingsLayer(layers.Layer):
    """Layer for getting the concatenated embeddings.

    The concatenated embeddings are obtained by concatenating subject embeddings
    and mode spatial map embeddings.
    """

    def call(self, inputs):
        subject_embeddings, mode_embeddings = inputs
        n_subjects, subject_embedding_dim = subject_embeddings.shape
        n_modes, mode_embedding_dim = mode_embeddings.shape

        # Match dimensions for concatenation
        subject_embeddings = tf.broadcast_to(
            tf.expand_dims(subject_embeddings, axis=1),
            [n_subjects, n_modes, subject_embedding_dim],
        )
        mode_embeddings = tf.broadcast_to(
            tf.expand_dims(mode_embeddings, axis=0),
            [n_subjects, n_modes, mode_embedding_dim],
        )

        # Concatenate the embeddings
        concat_embeddings = tf.concat([subject_embeddings, mode_embeddings], -1)

        return concat_embeddings


class SubjectMapLayer(layers.Layer):
    """Layer for getting the subject specific maps.

    This layer adds subject specific deviations to the group spatial maps.

    Parameters
    ----------
    which_map : str
        Which spatial map are we using? Must be one of 'means' and 'covariances'.
    epsilon : float
        Error added to the diagonal of covariances for numerical stability.
    """

    def __init__(self, which_map, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.which_map = which_map
        self.epsilon = epsilon
        if which_map == "covariances":
            self.bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])
        elif which_map == "means":
            self.bijector = tfb.Identity()
        else:
            raise ValueError("which_map must be one of 'means' and 'covariances'.")

    def call(self, inputs):
        group_map, dev = inputs
        group_map = self.bijector.inverse(group_map)

        # Match dimensions for addition
        group_map = tf.expand_dims(group_map, axis=0)
        subject_map = tf.add(group_map, dev)
        subject_map = self.bijector(subject_map)

        if self.which_map == "covariances":
            subject_map = add_epsilon(subject_map, self.epsilon, diag=True)

        return subject_map


class MixSubjectSpecificParametersLayer(layers.Layer):
    """Class for mixing means and covariances for the
    subject embedding model.

    The mixture is calculated as
    - m_t = Sum_j alpha_jt mu_j^(s_t)
    - C_t = Sum_j alpha_jt D_j^(s_t)
    where s_t is the subject at time t.
    """

    def call(self, inputs):
        # Unpack inputs:
        # - alpha.shape   = (None, sequence_length, n_modes)
        # - mu.shape      = (n_subjects, n_modes, n_channels)
        # - D.shape       = (n_subjects, n_modes, n_channels, n_channels)
        # - subj_id.shape = (None, sequence_length)
        alpha, mu, D, subj_id = inputs
        subj_id = tf.cast(subj_id, tf.int32)

        # The parameters for each time point
        dynamic_mu = tf.gather(mu, subj_id)
        dynamic_D = tf.gather(D, subj_id)

        # Next mix with the time course
        alpha = tf.expand_dims(alpha, axis=-1)
        m = tf.reduce_sum(tf.multiply(alpha, dynamic_mu), axis=2)

        alpha = tf.expand_dims(alpha, axis=-1)
        C = tf.reduce_sum(tf.multiply(alpha, dynamic_D), axis=2)

        return m, C


class StaticKLDivergenceLayer(layers.Layer):
    """Layer to calculate KL divergence between Gamma posterior
    and exponential prior for static parameters.

    Parameters
    ----------
    epsilon : float
        Error added to the standard deviations for numerical stability.
    n_batches : int
        Number of batches in the data. This is for calculating
        the scaling factor.
    """

    def __init__(self, epsilon, n_batches=1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_batches = n_batches

    def call(self, inputs, **kwargs):
        data, inference_alpha, inference_beta, model_beta = inputs

        # Add a small error for numerical stability
        inference_alpha = add_epsilon(inference_alpha, self.epsilon)
        inference_beta = add_epsilon(inference_beta, self.epsilon)
        model_beta = add_epsilon(model_beta, self.epsilon)

        # Calculate the KL divergence
        prior = tfp.distributions.Exponential(rate=model_beta)
        posterior = tfp.distributions.Gamma(
            concentration=inference_alpha, rate=inference_beta
        )
        kl_loss = tfp.distributions.kl_divergence(
            posterior, prior, allow_nan_stats=False
        )

        # Calculate the scaling for KL loss
        batch_size = tf.cast(tf.shape(data)[0], tf.float32)
        scaling_factor = batch_size * self.n_batches
        kl_loss /= scaling_factor

        kl_loss = tf.reduce_sum(kl_loss)

        return kl_loss


class MultiLayerPerceptronLayer(layers.Layer):
    """Multi-Layer Perceptron layer.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    n_units : int
        Number of units/neurons.
    norm_type : str
        Normalization layer type. Can be None, "layer" or "batch".
    act_type : str
        Activation type.
    drop_rate : float
        Dropout rate.
    """

    def __init__(
        self,
        n_layers,
        n_units,
        norm_type,
        act_type,
        drop_rate,
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
