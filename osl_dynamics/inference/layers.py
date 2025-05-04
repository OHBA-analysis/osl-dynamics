"""Custom Tensorflow layers."""

import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import (
    activations,
    layers,
    initializers,
    regularizers,
    constraints,
)

import osl_dynamics.inference.initializers as osld_initializers

tfb = tfp.bijectors


@tf.function
def add_epsilon(A, epsilon, diag=False):
    """Adds epsilon the the diagonal of batches of square matrices
    or all elements of matrices.

    Parameters
    ----------
    A : tf.Tensor
        Batches of square matrices or vectors. Shape is (..., N, N) or (..., N).
    epsilon : float
        Small error added to the diagonal of the matrices or every element of
        the vectors.
    diag : bool, optional
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


class TFConcatLayer(layers.Layer):
    """Wrapper for `tf.concat \
    <https://www.tensorflow.org/api_docs/python/tf/concat>`_.

    Parameters
    ----------
    axis : int
        Axis to concatenate along.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.concat(inputs, axis=self.axis)


class TFMatMulLayer(layers.Layer):
    """Wrapper for `tf.matmul \
    <https://www.tensorflow.org/api_docs/python/tf/linalg/matmul>`_.
    """

    def call(self, inputs, **kwargs):
        # If [A, B, C] is passed, we return matmul(A, matmul(B, C))
        out = inputs[-1]
        for tensor in inputs[len(inputs) - 2 :: -1]:
            out = tf.matmul(tensor, out)
        return out


class TFRangeLayer(layers.Layer):
    """Wrapper for `tf.range \
    <https://www.tensorflow.org/api_docs/python/tf/range>`_.

    Parameters
    ----------
    limit : int
        Upper limit for range.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, limit, **kwargs):
        super().__init__(**kwargs)
        self.limit = limit

    def call(self, inputs):
        return tf.range(self.limit)


class TFZerosLayer(layers.Layer):
    """Wrapper for `tf.zeros \
    <https://www.tensorflow.org/api_docs/python/tf/zeros>`_.

    Parameters
    ----------
    shape : tuple
        Shape of the zeros tensor.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def call(self, inputs):
        """
        Note
        ----
        The :code:`inputs` passed to this method are not used.
        """
        return tf.zeros(self.shape)


class TFBroadcastToLayer(layers.Layer):
    """Wrapper for `tf.broadcast_to \
        <https://www.tensorflow.org/api_docs/python/tf/broadcast_to>`_.
    """

    def __init__(self, n_modes, n_channels, **kwargs):
        super().__init__(**kwargs)
        self.n_modes = n_modes
        self.n_channels = n_channels

    def call(self, inputs):
        data, batch_size = inputs
        return tf.broadcast_to(data, (batch_size, self.n_modes, self.n_channels))


class TFGatherLayer(layers.Layer):
    """Wrapper for `tf.gather \
        <https://www.tensorflow.org/api_docs/python/tf/gather>`_.
    """

    def __init__(self, axis, batch_dims=0, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.batch_dims = batch_dims

    def call(self, inputs, **kwargs):
        return tf.gather(
            inputs[0], inputs[1], axis=self.axis, batch_dims=self.batch_dims
        )


class TFAddLayer(layers.Layer):
    """Wrapper for `tf.add \
        <https://www.tensorflow.org/api_docs/python/tf/math/add>`_.
    """

    def call(self, inputs, **kwargs):
        out = inputs[0]
        for tensor in inputs[1:]:
            out = tf.add(out, tensor)
        return out


class TFConstantLayer(layers.Layer):
    """Wrapper for `tf.constant \
        <https://www.tensorflow.org/api_docs/python/tf/constant>`_.
    """

    def __init__(self, values, **kwargs):
        super().__init__(**kwargs)
        self.values = values

    def call(self, inputs, **kwargs):
        return tf.constant(self.values)


def NormalizationLayer(norm_type, *args, **kwargs):
    """Returns a normalization layer.

    Parameters
    ----------
    norm_type : str
        Type of normalization layer. Either :code:`'layer'`, :code:`'batch'` or
        :code:`None`.
    args : arguments
        Arguments to pass to the normalization layer.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the normalization layer.
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
    """Returns a Recurrent Neural Network (RNN) layer.

    Parameters
    ----------
    rnn_type : str
        Type of RNN. Either :code:`'lstm'` or :code:`'gru'`.
    args : arguments
        Arguments to pass to the normalization layer.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the normalization layer.
    """
    if rnn_type == "lstm":
        return layers.LSTM(*args, **kwargs)
    elif rnn_type == "gru":
        return layers.GRU(*args, **kwargs)
    else:
        raise NotImplementedError(rnn_type)


class DummyLayer(layers.Layer):
    """Returns the inputs without modification."""

    def call(self, inputs, **kwargs):
        return inputs


class InverseCholeskyLayer(layers.Layer):
    """Layer for getting Cholesky vectors from postive definite symmetric matrices.

    Parameters
    ----------
    epsilon : float
        Small error added to the diagonal of the matrices.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.bijector = tfb.Chain(
            [tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()],
        )

    def call(self, inputs):
        inputs = add_epsilon(inputs, self.epsilon, diag=True)
        return self.bijector.inverse(inputs)


class BatchSizeLayer(layers.Layer):
    """Layer for getting the batch size.

    Parameters
    ----------
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def call(self, inputs):
        """
        Note
        ----
        The :code:`inputs` passed to this method are not used.
        """
        return tf.shape(inputs)[0]


class SampleGammaDistributionLayer(layers.Layer):
    """Layer for sampling from a Gamma distribution.

    This layer is a wrapper for `tfp.distributions.Gamma
    <https://www.tensorflow.org/probability/api_docs/python/tfp\
    /distributions/Gamma>`_.

    Parameters
    ----------
    epsilon : float
        Error to add to the shape and rate for numerical stability.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, epsilon, do_annealing, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        if do_annealing:
            self.annealing_factor = tf.keras.Variable(
                0.0, trainable=False, name="gamma_anneal_factor"
            )
        else:
            self.annealing_factor = tf.keras.Variable(
                1.0, trainable=False, name="gamma_anneal_factor"
            )

    def call(self, inputs, training=None, **kwargs):
        """This method accepts the shape and rate and outputs the samples."""
        alpha, beta, session_id = inputs
        # alpha.shape = (n_sessions, n_states, 1)
        # beta.shape = (n_sessions, n_states, 1)
        # session_id.shape = (None, sequence_length)

        # output.shape = (None, n_states, 1)

        alpha = add_epsilon(alpha, self.epsilon)
        beta = add_epsilon(beta, self.epsilon)
        session_id = session_id[:, 0]

        alpha = tf.gather(alpha, session_id, axis=0)  # shape = (None, n_states, 1)
        beta = tf.gather(beta, session_id, axis=0)  # shape = (None, n_states, 1)
        output = alpha / beta

        return output


class SampleNormalDistributionLayer(layers.Layer):
    """Layer for sampling from a Normal distribution.

    This layer is a wrapper for `tfp.distributions.Normal 
    <https://www.tensorflow.org/probability/api_docs/python/tfp\
    /distributions/Normal>`_.

    Parameters
    ----------
    epsilon : float
        Error to add to the standard deviations for numerical stability.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs, training=None, **kwargs):
        """
        This method accepts the mean and the standard deviation and outputs
        the samples.
        """
        mu, sigma = inputs
        sigma = add_epsilon(sigma, self.epsilon)
        if training:
            N = tfp.distributions.Normal(loc=mu, scale=sigma)
            return N.sample()
        else:
            return mu


class SampleGumbelSoftmaxDistributionLayer(layers.Layer):
    """Layer for sampling from a Gumbel-Softmax distribution.

    This layer is a wrapper for `tfp.distributions.RelaxedOneHotCategorical 
    <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions\
    /RelaxedOneHotCategorical>`_.

    Parameters
    ----------
    temperature : float
        Temperature for the Gumbel-Softmax distribution.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs)
        self.temperature = tf.keras.Variable(
            temperature, trainable=False, dtype=tf.float32, name="gs_temperature"
        )

    def call(self, inputs, training=None, **kwargs):
        """This method accepts logits and outputs samples."""
        if training:
            gs = tfp.distributions.RelaxedOneHotCategorical(
                temperature=self.temperature, logits=inputs
            )
            return gs.sample()
        else:
            softmaxed_logits = tf.nn.softmax(inputs / self.temperature, axis=-1)
            return softmaxed_logits


class SampleOneHotCategoricalDistributionLayer(layers.Layer):
    """Layer for sampling from a Categorical distribution.

    This layer is a wrapper for `tfp.distributions.OneHotCategorical 
    <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions\
    /OneHotCategorical>`_.
    """

    def call(self, inputs, **kwargs):
        """The method accepts logits and outputs samples."""
        cat = tfp.distributions.OneHotCategorical(
            logits=inputs,
            dtype=tf.float32,
        )
        return cat.sample()


class SoftmaxLayer(layers.Layer):
    """Layer for applying a softmax activation function.

    Parameters
    ----------
    initial_temperature : float
        Temperature parameter.
    learn_temperature : bool
        Should we learn the alpha temperature?
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, initial_temperature, learn_temperature, **kwargs):
        super().__init__(**kwargs)
        self.layers = [
            LearnableTensorLayer(
                shape=(),
                learn=learn_temperature,
                initial_value=initial_temperature,
                name=self.name + "_kernel",
            )
        ]

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        temperature = self.layers[0](inputs)
        return activations.softmax(inputs / temperature, axis=-1)


class LearnableTensorLayer(layers.Layer):
    """Layer to learn a tensor.

    Parameters
    ----------
    shape : tuple
        Shape of the tensor.
    learn : bool, optional
        Should we learn the tensor?
    initializer : tf.keras.initializers.Initializer, optional
        Initializer for the tensor.
    initial_value : float, optional
        Initial value for the tensor if initializer is not passed.
    regularizer : osl-dynamics regularizer, optional
        Regularizer for the tensor. Must be from `inference.regularizers 
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /inference/regularizers/index.html>`_.
    constraint : tf.keras.constraints.Constraint, optional
        Constraint for the tensor. Limits the values the weights can take.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(
        self,
        shape,
        learn=True,
        initializer=None,
        initial_value=None,
        regularizer=None,
        constraint=None,
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

        # Constraint for the tensor
        self.constraint = constraint

    def add_regularization(self, tensor):
        reg = self.regularizer(tensor)
        self.add_loss(reg)

    def build(self, input_shape):
        # Create a weight for the tensor
        self.tensor = self.add_weight(
            name="tensor",
            shape=self.shape,
            dtype=tf.float32,
            initializer=self.tensor_initializer,
            trainable=self.learn,
            constraint=self.constraint,
        )
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        if self.regularizer is not None and training:
            self.add_regularization(self.tensor)
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
        Initial value for the vectors. Shape must be (n, m).
    regularizer : osl-dynamics regularizer, optional
        Regularizer for the tensor. Must be from `inference.regularizers \
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /inference/regularizers/index.html>`_.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
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

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Note
        ----
        The :code:`inputs` passed to this method are not used.
        """
        learnable_tensors_layer = self.layers[0]
        vectors = learnable_tensors_layer(inputs, **kwargs)
        return vectors


class CovarianceMatricesLayer(layers.Layer):
    """Layer to learn a set of covariance matrices.

    A cholesky factor is learnt and used to calculate a covariance matrix as
    :math:`C = LL^T`, where :math:`L` is the cholesky factor. The cholesky
    factor is learnt as a vector of free parameters.

    Parameters
    ----------
    n : int
        Number of matrices.
    m : int
        Number of rows/columns.
    learn : bool
        Should the matrices be learnable?
    initial_value : np.ndarray
        Initial values for the matrices. Shape must be (n, m, m).
    epsilon : float
        Error added to the diagonal of covariances matrices for numerical
        stability.
    regularizer : osl-dynamics regularizer, optional
        Regularizer for the tensor. Must be from `inference.regularizers 
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /inference/regularizers/index.html>`_.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
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
        self.bijector = tfb.Chain(
            [tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()],
        )

        # Do we have an initial value?
        if initial_value is not None:
            # Check it's the correct shape
            if initial_value.shape != (n, m, m):
                raise ValueError(f"initial_value shape must be ({n}, {m}, {m}).")

            # Calculate the flattened cholesky factors
            initial_value = initial_value.astype("float32")
            initial_flattened_cholesky_factors = self.bijector.inverse(
                initial_value,
            )

            # We don't need an initializer
            initializer = None
        else:
            # No initial value has been passed
            initial_flattened_cholesky_factors = None
            if learn:
                # Use a random initializer
                initializer = osld_initializers.NormalIdentityCholeskyInitializer(
                    std=0.1,
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

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Note
        ----
        The :code:`inputs` passed to this method are not used.
        """
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
        Initial values for the matrices. Shape must be (n, m, m).
    epsilon : float
        Error added to the diagonal of correlation matrices for numerical
        stability.
    regularizer : osl-dynamics regularizer, optional
        Regularizer for the tensor. Must be from `inference.regularizers 
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /inference/regularizers/index.html>`_.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
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
            initial_flattened_cholesky_factors = self.bijector.inverse(
                initial_value,
            )

            # We don't need an initializer
            initializer = None
        else:
            # No initial value has been passed
            initial_flattened_cholesky_factors = None
            if learn:
                # Use a random initializer
                initializer = osld_initializers.NormalCorrelationCholeskyInitializer(
                    std=0.1,
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

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Note
        ----
        The :code:`inputs` passed to this method are not used.
        """
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
        Initial values for the matrices. Shape must be (n, m, m) or (n, m).
    epsilon : float
        Error added to the diagonal matrices for numerical stability.
    regularizer : osl-dynamics regularizer, optional
        Regularizer for the tensor. Must be from `inference.regularizers 
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /inference/regularizers/index.html>`_.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
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
                initializer = osld_initializers.NormalDiagonalInitializer(
                    std=0.05,
                )
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

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Note
        ----
        The :code:`inputs` passed to this method are not used.
        """
        learnable_tensor_layer = self.layers[0]
        diagonals = learnable_tensor_layer(inputs, **kwargs)
        diagonals = self.bijector(diagonals)
        diagonals = add_epsilon(diagonals, self.epsilon)
        return tf.linalg.diag(diagonals)


class DampedOscillatorLayer(layers.Layer):
    """Layer to learn a set of damped oscillators.

    Parameters
    ----------
    n : int
        Number of oscillators.
    m : int
        Number of elements.
    sampling_frequency : float
        Sampling frequency in Hz.
    damping_limit : float
        Upper limit for the damping parameter.
        Values are clipped to [0, damping_limit].
    frequency_limit : tuple
        Limits for the frequency parameter.
        Upper limit should not be higher than the Nyquist frequency.
    learn_amplitude : bool
        Should the amplitudes be learnable?
        If not, they will be fixed to 1.0.
        Overriden if the general `learn` argument is False.
    learn : bool
        Should the oscillators be learnable?
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(
        self,
        n,
        m,
        sampling_frequency,
        damping_limit,
        frequency_limit,
        learn_amplitude,
        learn,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_frequency = sampling_frequency
        self.damping_limit = damping_limit

        self.tau = (
            tf.expand_dims(tf.range(0, m, dtype=tf.float32), axis=0)
            / sampling_frequency
        )

        self.damping = LearnableTensorLayer(
            shape=(n, 1),
            learn=learn,
            initializer=initializers.Constant(0.5),
            name=self.name + "_damping",
        )

        self.frequency = LearnableTensorLayer(
            shape=(n, 1),
            learn=learn,
            initializer=initializers.RandomUniform(
                minval=frequency_limit[0],
                maxval=frequency_limit[1],
            ),
            name=self.name + "_frequency",
        )

        self.amplitude = LearnableTensorLayer(
            shape=(n, 1),
            learn=learn and learn_amplitude,
            initializer=initializers.Constant(1.0),
            name=self.name + "_amplitude",
        )

        self.layers = [self.damping, self.frequency, self.amplitude]

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        """Calculate damped oscillator.

        Note
        ----
        The :code:`inputs` passed to this method are not used.
        """
        damping = self.damping(inputs, **kwargs)
        damping = tf.clip_by_value(damping, 0, self.damping_limit)
        frequency = self.frequency(inputs, **kwargs)
        frequency = tf.clip_by_value(frequency, 1, self.sampling_frequency / 2)
        omega = 2 * np.pi * frequency
        amplitude = self.amplitude(inputs, **kwargs)
        return amplitude * tf.exp(-damping * self.tau) * tf.cos(omega * self.tau)


class DampedOscillatorCovarianceMatricesLayer(layers.Layer):
    """Layer to learn a set of damped oscillator covariances.

    Parameters
    ----------
    n : int
        Number of matrices.
    m : int
        Number of rows/columns.
    sampling_frequency : float
        Sampling frequency in Hz.
    damping_limit : float
        Upper limit for the damping parameter.
        Values are clipped to [0, damping_limit].
    frequency_limit : tuple[float, float]
        Limits for the frequency parameter.
        Upper limit should not be higher than the Nyquist frequency.
    learn_amplitude : bool
        Should the amplitudes be learnable?
        If not, they will be fixed to 1.0.
        Overriden if the general `learn` argument is False.
    learn : bool
        Should the matrices be learnable?
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(
        self,
        n,
        m,
        sampling_frequency,
        damping_limit,
        frequency_limit,
        learn_amplitude,
        learn,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.oscillator_layer = DampedOscillatorLayer(
            n=n,
            m=m,
            sampling_frequency=sampling_frequency,
            damping_limit=damping_limit,
            frequency_limit=frequency_limit,
            learn_amplitude=learn_amplitude,
            learn=learn,
        )

        self.layers = [self.oscillator_layer]

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        """Retrieve the covariance matrices.

        Note
        ----
        The :code:`inputs` passed to this method are not used.
        """
        oscillator = self.oscillator_layer(inputs, **kwargs)
        return tf.linalg.LinearOperatorToeplitz(
            row=oscillator,
            col=oscillator,
        ).to_dense()


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
    regularizer : osl-dynamics regularizer, optional
        Regularizer for the tensor. Must be from `inference.regularizers 
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /inference/regularizers/index.html>`_.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(
        self,
        m,
        constraint,
        learn,
        initial_value,
        epsilon,
        regularizer=None,
        **kwargs,
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

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Note
        ----
        The :code:`inputs` passed to this method are not used.
        """
        return self.matrix_layer(inputs, **kwargs)[0]


class MixVectorsLayer(layers.Layer):
    r"""Mix a set of vectors.

    The mixture is calculated as :math:`m_t = \displaystyle\sum_j
    \alpha_{jt} \mu_j`, where :math:`\mu_j` are the vectors and
    :math:`\alpha_{jt}` are mixing coefficients.
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
    r"""Layer to mix matrices.

    The mixture is calculated as :math:`C_t = \displaystyle\sum_j
    \alpha_{jt} D_j`, where :math:`D_j` are the matrices and
    :math:`\alpha_{jt}` are mixing coefficients.
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
    """Layer to calculate the negative log-likelihood.

    We assume a multivariate normal probability density. This layer will add
    the negative log-likelihood to the loss.

    Parameters
    ----------
    calculation : str
        Operation for reducing the time dimension. Either 'mean' or 'sum'.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, calculation, **kwargs):
        super().__init__(**kwargs)
        self.calculation = calculation

    def call(self, inputs):
        """
        The method takes the data, mean vector and covariance matrix.
        """
        x, mu, sigma = inputs

        # Multivariate normal distribution
        mvn = tfp.distributions.MultivariateNormalTriL(
            loc=mu,
            scale_tril=tf.linalg.cholesky(sigma),
            allow_nan_stats=False,
        )

        # Calculate the log-likelihood
        ll_loss = mvn.log_prob(x)

        if self.calculation == "sum":
            # Sum over time dimension and average over the batch dimension
            ll_loss = tf.reduce_sum(ll_loss, axis=1)
            ll_loss = tf.reduce_mean(ll_loss, axis=0)
        else:
            # Average over time and batches
            ll_loss = tf.reduce_mean(ll_loss, axis=(0, 1))

        # Add the negative log-likelihood to the loss
        nll_loss = -ll_loss
        self.add_loss(nll_loss)

        return tf.expand_dims(nll_loss, axis=-1)


class KLDivergenceLayer(layers.Layer):
    """Layer to calculate a KL divergence between two Normal distributions.

    Parameters
    ----------
    epsilon : float
        Error added to the standard deviations for numerical stability.
    calculation : str
        Operation for reducing the time dimension. Either 'mean' or 'sum'.
    clip_start : int, optional
        Index to clip the sequences inputted to this layer.
        Default is no clipping.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, epsilon, calculation, clip_start=0, **kwargs):
        super().__init__(**kwargs)
        self.clip_start = clip_start
        self.epsilon = epsilon
        self.calculation = calculation

    def call(self, inputs, **kwargs):
        inference_mu, inference_sigma, model_mu, model_sigma = inputs

        # Add a small error for numerical stability
        inference_sigma = add_epsilon(inference_sigma, self.epsilon)
        model_sigma = add_epsilon(model_sigma, self.epsilon)

        # The model network predicts one time step into the future compared to
        # the inference network. We clip the sequences to ensure we are
        # comparing the correct time points.
        model_mu = model_mu[:, self.clip_start : -1]
        model_sigma = model_sigma[:, self.clip_start : -1]

        inference_mu = inference_mu[:, self.clip_start + 1 :]
        inference_sigma = inference_sigma[:, self.clip_start + 1 :]

        # Calculate the KL divergence between the posterior and prior
        prior = tfp.distributions.Normal(loc=model_mu, scale=model_sigma)
        posterior = tfp.distributions.Normal(
            loc=inference_mu,
            scale=inference_sigma,
        )
        kl_loss = tfp.distributions.kl_divergence(
            posterior, prior, allow_nan_stats=False
        )

        if self.calculation == "sum":
            # Sum the KL loss for each mode and time point and average over batches
            kl_loss = tf.reduce_sum(kl_loss, axis=2)
            kl_loss = tf.reduce_sum(kl_loss, axis=1)
            kl_loss = tf.reduce_mean(kl_loss, axis=0)
        else:
            # Sum the KL loss for each mode, average time points and batches
            kl_loss = tf.reduce_sum(kl_loss, axis=2)
            kl_loss = tf.reduce_mean(kl_loss, axis=(0, 1))

        return kl_loss


class KLLossLayer(layers.Layer):
    """Layer to calculate the KL loss.

    This layer sums KL divergences if multiple values as passed, applies an
    annealing factor and adds the value to the loss function.

    Parameters
    ----------
    do_annealing : bool
        Should we perform KL annealing?
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, do_annealing, **kwargs):
        super().__init__(**kwargs)
        if do_annealing:
            self.annealing_factor = tf.keras.Variable(
                0.0, trainable=False, name="kl_factor"
            )
        else:
            self.annealing_factor = tf.keras.Variable(
                1.0, trainable=False, name="kl_factor"
            )

    def call(self, inputs, training=False, **kwargs):
        kl_loss = inputs

        if isinstance(inputs, list):
            # Sum KL divergences
            kl_loss = tf.add_n(kl_loss)

        if training:
            # KL annealing
            kl_loss = tf.multiply(kl_loss, self.annealing_factor)

        # Add to loss
        self.add_loss(kl_loss)

        return tf.expand_dims(kl_loss, axis=-1)


class InferenceRNNLayer(layers.Layer):
    """RNN inference network.

    Parameters
    ----------
    rnn_type : str
        Either :code:`'lstm'` or :code:`'gru'`.
    norm_type : str
        Either :code:`'layer'`, :code:`'batch'` or :code:`None`.
    act_type : 'str'
        Activation type, e.g. :code:`'relu'`, :code:`'elu'`, etc.
    n_layers : int
        Number of layers.
    n_units : int
        Number of units/neurons per layer.
    drop_rate : float
        Dropout rate for the output of each layer.
    reg : str
        Regularization for each layer.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
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

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs


class ModelRNNLayer(layers.Layer):
    """RNN generative model.

    Parameters
    ----------
    rnn_type : str
        Either :code:`'lstm'` or :code:`'gru'`.
    norm_type : str
        Either :code:`'layer'`, :code:`'batch'` or :code:`None`.
    act_type : 'str'
        Activation type, e.g. :code:`'relu'`, :code:`'elu'`, etc.
    n_layers : int
        Number of layers.
    n_units : int
        Number of units/neurons per layer.
    drop_rate : float
        Dropout rate for the output of each layer.
    reg : str
        Regularization for each layer.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
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

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs


class CategoricalKLDivergenceLayer(layers.Layer):
    """Layer to calculate a KL divergence between two categorical distributions.

    Parameters
    ----------
    calculation : str
        Operation for reducing the time dimension. Either 'mean' or 'sum'.
    clip_start : int, optional
        Index to clip the sequences inputted to this layer.
        Default is no clipping.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, calculation, clip_start=0, **kwargs):
        super().__init__(**kwargs)
        self.calculation = calculation
        self.clip_start = clip_start

    def call(self, inputs, **kwargs):
        inference_logits, model_logits = inputs

        # The model network predicts one time step into the future compared to
        # the inference network. We clip the sequences to ensure we are
        # comparing the correct time points.
        model_logits = model_logits[:, self.clip_start : -1]
        inference_logits = inference_logits[:, self.clip_start + 1 :]

        # Calculate the KL divergence between the posterior and prior
        prior = tfp.distributions.Categorical(logits=model_logits)
        posterior = tfp.distributions.Categorical(logits=inference_logits)
        kl_loss = tfp.distributions.kl_divergence(
            posterior, prior, allow_nan_stats=False
        )

        if self.calculation == "sum":
            # Sum the KL loss for each time point and average over batches
            kl_loss = tf.reduce_sum(kl_loss, axis=1)
            kl_loss = tf.reduce_mean(kl_loss, axis=0)
        else:
            # Average over time and batches
            kl_loss = tf.reduce_mean(kl_loss, axis=(0, 1))

        return kl_loss


class CategoricalLogLikelihoodLossLayer(layers.Layer):
    """Layer to calculate the log-likelihood loss assuming a categorical model.

    Parameters
    ----------
    n_states : int
        Number of states.
    calculation : str
        Operation for reducing the time dimension. Either 'mean' or 'sum'.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, n_states, calculation, **kwargs):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.calculation = calculation

    def call(self, inputs, **kwargs):
        x, mu, sigma, probs = inputs

        # Log-likelihood for each state
        ll_loss = tf.zeros(shape=tf.shape(x)[:-1])
        for i in range(self.n_states):
            mvn = tfp.distributions.MultivariateNormalTriL(
                loc=tf.gather(mu, i, axis=-2),
                scale_tril=tf.linalg.cholesky(tf.gather(sigma, i, axis=-3)),
                allow_nan_stats=False,
            )
            a = mvn.log_prob(x)
            ll_loss += probs[:, :, i] * a

        if self.calculation == "sum":
            # Sum over time dimension and average over the batch dimension
            ll_loss = tf.reduce_sum(ll_loss, axis=1)
            ll_loss = tf.reduce_mean(ll_loss, axis=0)
        else:
            # Average over time and batches
            ll_loss = tf.reduce_mean(ll_loss, axis=(0, 1))

        # Add the negative log-likelihood to the loss
        nll_loss = -ll_loss
        self.add_loss(nll_loss)

        return tf.expand_dims(nll_loss, axis=-1)


class ConcatEmbeddingsLayer(layers.Layer):
    """Layer for getting the concatenated embeddings.

    The concatenated embeddings are obtained by concatenating embeddings
    and spatial embeddings.
    """

    def call(self, inputs):
        embeddings, spatial_embeddings = inputs
        batch_size, embeddings_dim = embeddings.shape
        n_states, spatial_embeddings_dim = spatial_embeddings.shape

        place_holder = tf.zeros_like(embeddings[:, 0])  # shape = (None,)

        # Broadcast the embeddings and spatial embeddings
        embeddings = tf.expand_dims(embeddings, axis=1) + tf.zeros(
            (1, n_states, embeddings_dim)
        )
        spatial_embeddings = tf.expand_dims(
            spatial_embeddings, axis=0
        ) + 0 * tf.expand_dims(tf.expand_dims(place_holder, axis=-1), axis=-1)

        concat_embeddings = tf.concat([embeddings, spatial_embeddings], axis=-1)

        return concat_embeddings


class SessionParamLayer(layers.Layer):
    """Layer for getting the array specific parameters.

    This layer adds deviations to the group spatial parameters.

    Parameters
    ----------
    param : str
        Which parameter are we using? Must be :code:`'means'` or
        :code:`'covariances'`.
    epsilon : float
        Error added to the diagonal of covariances for numerical stability.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, param, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.param = param
        self.epsilon = epsilon
        if param == "covariances":
            self.bijector = tfb.Chain(
                [tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()],
            )
        elif param == "means":
            self.bijector = tfb.Identity()
        else:
            raise ValueError("param must be one of 'means' and 'covariances'.")

    def call(self, inputs):
        group_param, dev = inputs
        group_param = self.bijector.inverse(group_param)

        # Match dimensions for addition
        group_param = tf.expand_dims(group_param, axis=0)
        session_param = tf.add(group_param, dev)
        session_param = self.bijector(session_param)

        if self.param == "covariances":
            session_param = add_epsilon(session_param, self.epsilon, diag=True)

        return session_param


class MixSessionSpecificParametersLayer(layers.Layer):
    r"""Class for mixing means and covariances.

    The mixture is calculated as

    - :math:`m_t = \displaystyle\sum_j \alpha_{jt} \mu_j^{s_t}`
    - :math:`C_t = \displaystyle\sum_j \alpha_{jt} D_j^{s_t}`

    where :math:`s_t` is the array at time :math:`t`.
    """

    def call(self, inputs):
        # Unpack inputs:
        # - alpha.shape   = (None, sequence_length, n_modes)
        # - mu.shape      = (None, n_modes, n_channels)
        # - D.shape       = (None, n_modes, n_channels, n_channels)
        alpha, mu, D = inputs

        # Add the sequence dimension
        mu = tf.expand_dims(mu, axis=1)
        D = tf.expand_dims(D, axis=1)

        # Next mix with the time course
        alpha = tf.expand_dims(alpha, axis=-1)
        m = tf.reduce_sum(tf.multiply(alpha, mu), axis=2)

        alpha = tf.expand_dims(alpha, axis=-1)
        C = tf.reduce_sum(tf.multiply(alpha, D), axis=2)

        return m, C


class GammaExponentialKLDivergenceLayer(layers.Layer):
    """Layer to calculate KL divergence between Gamma posterior and exponential
    prior for deviation magnitude.

    Parameters
    ----------
    epsilon : float
        Error added to the standard deviations for numerical stability.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.scale_factor = -1

    def call(self, inputs, training=None, **kwargs):
        # inference_alpha.shape = (n_sessions, n_states, 1)
        # inference_beta.shape  = (n_sessions, n_states, 1)
        # model_beta.shape      = (n_sessions, n_states, 1)

        if training and self.scale_factor == -1:
            raise AttributeError(
                "Static loss scaling factor not set. "
                "Please run model.set_regularizers()."
            )

        inference_alpha, inference_beta, model_beta = inputs

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
        kl_loss = tf.reduce_sum(kl_loss)

        return self.scale_factor * kl_loss


class MultiLayerPerceptronLayer(layers.Layer):
    """Multi-Layer Perceptron layer.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    n_units : int
        Number of units/neurons.
    norm_type : str
        Normalization layer type. Can be :code:`'layer'`, :code:`'batch'` or
        :code:`None`.
    act_type : str
        Activation type.
    drop_rate : float
        Dropout rate.
    regularizer : str, optional
        Regularizer type.
    regularizer_strength : float, optional
        Constant to multiply the regularization by.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the base class.
    """

    def __init__(
        self,
        n_layers,
        n_units,
        norm_type,
        act_type,
        drop_rate,
        regularizer=None,
        regularizer_strength=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.regularizer = regularizers.get(regularizer)
        self.regularizer_strength = regularizer_strength
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(layers.Dense(n_units))
            self.layers.append(NormalizationLayer(norm_type))
            self.layers.append(layers.Activation(act_type))
            self.layers.append(layers.Dropout(drop_rate))

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        reg = 0.0
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
            if self.regularizer is not None and isinstance(layer, layers.Dense):
                reg += self.regularizer(layer.kernel)
                reg += self.regularizer(layer.bias)
        if self.regularizer is not None and training:
            reg *= self.regularizer_strength
            self.add_loss(reg)
        return inputs


class HiddenMarkovStateInferenceLayer(layers.Layer):
    """Hidden Markov state inference layer.

    This layer uses the Baum-Welch algorithm to calculate the posterior
    for the hidden state in a Hidden Markov Model (HMM).

    Parameters
    ----------
    n_states : int
        Number of states.
    sequence_length : int
        Length of the sequence.
    initial_trans_prob : np.ndarray
        Initial transition probability matrix.
        Shape must be (n_states, n_states).
    initial_state_probs : np.ndarray
        Initial transition probability matrix.
        Shape must be (n_states,)
    learn_trans_prob : bool
        Should we learn the transition probability matrix?
    learn_initial_state_probs : bool
        Should we learn the initial state probabilities?
    implementation : str, optional
        'rescale' or 'log' implementation of the Baum-Welch algorithm.
    use_stationary_distribution : bool, optional
        Should we use the stationary distribution (estimated from the
        transition probability matrix) for the initial state probabilities?
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the normalization layer.
    """

    def __init__(
        self,
        n_states,
        sequence_length,
        initial_trans_prob,
        initial_state_probs,
        learn_trans_prob,
        learn_initial_state_probs,
        implementation="rescale",
        use_stationary_distribution=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.sequence_length = sequence_length
        self.use_stationary_distribution = use_stationary_distribution

        # Implementation for Baum-Welch algorithm
        if implementation not in ["rescale", "log"]:
            raise ValueError("implementation should be 'rescale' or 'log'.")
        self.implementation = implementation

        # Initial state probabilities
        if initial_state_probs is None:
            initial_state_probs = np.ones(self.n_states) / self.n_states

        if initial_state_probs.shape != (n_states,):
            raise ValueError(f"initial_trans_prob shape must be ({n_states},).")
        initial_state_probs = initial_state_probs.astype("float32")

        initial_state_probs_layer = LearnableTensorLayer(
            shape=(n_states,),
            learn=learn_initial_state_probs,
            initializer=osld_initializers.WeightInitializer(initial_state_probs),
            initial_value=initial_state_probs,
            regularizer=None,
            name=self.name + "_initial_state_probs_kernel",
        )

        # Transition probability matrix
        if initial_trans_prob is None:
            initial_trans_prob = np.ones((n_states, n_states)) * 0.1 / (n_states - 1)
            np.fill_diagonal(initial_trans_prob, 0.9)

        if initial_trans_prob.shape != (n_states, n_states):
            raise ValueError(
                f"initial_trans_prob shape must be ({n_states}, {n_states})."
            )
        initial_trans_prob = initial_trans_prob.astype("float32")

        trans_prob_layer = LearnableTensorLayer(
            shape=(n_states, n_states),
            learn=learn_trans_prob,
            initializer=osld_initializers.WeightInitializer(initial_trans_prob),
            initial_value=initial_trans_prob,
            regularizer=None,
            name=self.name + "_trans_prob_kernel",
        )

        # We use self.layers for compatibility with
        # initializers.reinitialize_model_weights
        self.layers = [initial_state_probs_layer, trans_prob_layer]

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
        self.built = True

    def get_stationary_distribution(self):
        trans_prob = self.get_trans_prob()
        eigval, eigvec = tf.linalg.eig(trans_prob)
        eigvec = tf.boolean_mask(
            eigvec, tf.experimental.numpy.isclose(eigval, 1), axis=1
        )
        stationary_distribution = tf.math.real(tf.squeeze(eigvec))
        stationary_distribution /= tf.reduce_sum(stationary_distribution)
        return stationary_distribution

    def get_initial_state_probs(self):
        if self.use_stationary_distribution:
            return self.get_stationary_distribution()
        else:
            learnable_tensors_layer = self.layers[0]
            return learnable_tensors_layer(tf.constant(1))

    def get_trans_prob(self):
        learnable_tensors_layer = self.layers[1]
        return learnable_tensors_layer(tf.constant(1))

    @tf.function
    def _baum_welch(self, log_B):
        def _get_indices(time, batch_size):
            return tf.concat(
                [
                    tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1),
                    tf.fill((batch_size, 1), time),
                ],
                axis=1,
            )

        # Small error for improving the numerical stability of the log-likelihood
        eps = tf.experimental.numpy.finfo(self.compute_dtype).eps

        # Batch size
        batch_size = tf.shape(log_B)[0]

        # Transition probability matrix
        P = self.get_trans_prob()
        P = tf.cast(P, self.compute_dtype)

        # Initial state probailities
        Pi_0 = self.get_initial_state_probs()
        Pi_0 = tf.cast(Pi_0, self.compute_dtype)

        if self.implementation == "rescale":

            def _rescale(probs, scale, indices, time, update_scale=True):
                # Rescale probabilities to help with numerical stability
                # (over/underflow)
                if update_scale:
                    scale = tf.tensor_scatter_nd_update(
                        scale, indices, tf.reduce_sum(probs[:, time], axis=-1)
                    )
                probs = tf.tensor_scatter_nd_update(
                    probs,
                    indices,
                    probs[:, time] / tf.expand_dims(scale[:, time] + eps, axis=-1),
                )
                return probs, scale

            # Temporary variables used in the calculation
            alpha = tf.zeros(
                [batch_size, self.sequence_length, self.n_states],
                dtype=self.compute_dtype,
            )
            beta = tf.zeros(
                [batch_size, self.sequence_length, self.n_states],
                dtype=self.compute_dtype,
            )
            scale = tf.zeros(
                [batch_size, self.sequence_length], dtype=self.compute_dtype
            )

            # Renormalise the log-likelihood for numerical stability
            max_values = tf.reduce_max(log_B, axis=-1, keepdims=True)
            max_values = tf.minimum(max_values, 0.0)
            log_B -= max_values

            # Calculate likelihood
            B = tf.exp(log_B)

            # Forward pass
            def cond(i, alpha, scale):
                return i < self.sequence_length

            def body(i, alpha, scale):
                indices = _get_indices(i, batch_size)
                values = tf.cond(
                    tf.equal(i, 0),
                    lambda: Pi_0 * B[:, 0],
                    lambda: (
                        tf.reduce_sum(
                            tf.expand_dims(alpha[:, i - 1], axis=1) * tf.transpose(P),
                            axis=-1,
                        )
                        * B[:, i]
                    ),
                )
                alpha = tf.tensor_scatter_nd_update(alpha, indices, values)
                alpha, scale = _rescale(alpha, scale, indices, i)
                return i + 1, alpha, scale

            i = tf.constant(0)
            _, alpha, scale = tf.while_loop(
                cond=cond, body=body, loop_vars=[i, alpha, scale]
            )

            # Backward pass
            def cond(i, beta, scale):
                return i > 0

            def body(i, beta, scale):
                indices = _get_indices(i - 1, batch_size)
                values = tf.cond(
                    tf.equal(i, self.sequence_length),
                    lambda: tf.ones_like(beta[:, -1]),
                    lambda: tf.reduce_sum(
                        tf.expand_dims(beta[:, i] * B[:, i], axis=1) * P,
                        axis=-1,
                    ),
                )
                beta = tf.tensor_scatter_nd_update(beta, indices, values)
                beta, _ = _rescale(beta, scale, indices, i - 1, update_scale=False)
                return i - 1, beta, scale

            i = tf.constant(self.sequence_length)
            _, beta, scale = tf.while_loop(
                cond=cond, body=body, loop_vars=[i, beta, scale]
            )

            # Marginal probabilities
            gamma = alpha * beta
            gamma /= tf.reduce_sum(gamma, axis=-1, keepdims=True)

            # Joint probabilities
            b = beta[:, 1:] * B[:, 1:]
            xi = P * tf.expand_dims(alpha[:, :-1], axis=3) * tf.expand_dims(b, axis=2)
            xi /= tf.reduce_sum(xi, axis=(2, 3), keepdims=True)

        if self.implementation == "log":
            # Temporary variables used in the calculation
            log_alpha = tf.zeros(
                [batch_size, self.sequence_length, self.n_states],
                dtype=self.compute_dtype,
            )
            log_beta = tf.zeros(
                [batch_size, self.sequence_length, self.n_states],
                dtype=self.compute_dtype,
            )

            # Calculate log probabilities
            log_P = tf.math.log(P + eps)
            log_Pi_0 = tf.math.log(Pi_0 + eps)

            # Forward pass
            def cond(i, log_alpha):
                return i < self.sequence_length

            def body(i, log_alpha):
                indices = _get_indices(i, batch_size)
                values = tf.cond(
                    tf.equal(i, 0),
                    lambda: log_Pi_0 + log_B[:, 0],
                    lambda: (
                        tf.reduce_logsumexp(
                            tf.expand_dims(log_alpha[:, i - 1], axis=1)
                            + tf.transpose(log_P),
                            axis=-1,
                        )
                        + log_B[:, i]
                    ),
                )
                log_alpha = tf.tensor_scatter_nd_update(log_alpha, indices, values)
                return i + 1, log_alpha

            i = tf.constant(0)
            _, log_alpha = tf.while_loop(cond=cond, body=body, loop_vars=[i, log_alpha])

            # Backward pass
            def cond(i, log_beta):
                return i > 0

            def body(i, log_beta):
                indices = _get_indices(i - 1, batch_size)
                values = tf.cond(
                    tf.equal(i, self.sequence_length),
                    lambda: tf.zeros_like(log_beta[:, -1]),
                    lambda: tf.reduce_logsumexp(
                        tf.expand_dims(log_beta[:, i] + log_B[:, i], axis=1) + log_P,
                        axis=-1,
                    ),
                )
                log_beta = tf.tensor_scatter_nd_update(log_beta, indices, values)
                return i - 1, log_beta

            i = tf.constant(self.sequence_length)
            _, log_beta = tf.while_loop(cond=cond, body=body, loop_vars=[i, log_beta])

            # Marginal probabilities
            log_gamma = log_alpha + log_beta
            log_gamma -= tf.reduce_logsumexp(log_gamma, axis=-1, keepdims=True)

            # Joint probabilities
            log_b = log_beta[:, 1:] + log_B[:, 1:]
            log_xi = (
                log_P
                + tf.expand_dims(log_alpha[:, :-1], axis=3)
                + tf.expand_dims(log_b, axis=2)
            )
            log_xi -= tf.reduce_logsumexp(log_xi, axis=(2, 3), keepdims=True)

            # Convert from log probabilities to probabilities
            gamma = tf.exp(log_gamma)
            xi = tf.exp(log_xi)

        return gamma, xi

    def _trans_prob_update(self, gamma, xi):
        sum_xi = tf.reduce_sum(xi, axis=(0, 1))
        sum_gamma = tf.reduce_sum(gamma[:, :-1], axis=(0, 1))
        sum_gamma = tf.expand_dims(sum_gamma, axis=-1)
        return sum_xi / sum_gamma

    def call(self, log_B, **kwargs):
        @tf.custom_gradient
        def posterior(log_B):
            # Calculate marginal (gamma) and joint (xi) posterior
            gamma, xi = self._baum_welch(log_B)

            # Calculate gradient
            def grad(*args, **kwargs):
                # Note, this function actually returns the estimated
                # value for what the transition probability matrix
                # should be based on the joint and marginal posterior
                # rather than the gradient (i.e. change in the parameter).
                #
                # This is accounted for when updating the variable
                # in inference.optimizers.ExponentialMovingAverageOptimizer

                if len(kwargs) == 0:
                    # No trainable variables to calculate the gradient for
                    return

                grads = []
                variables = kwargs["variables"]
                for v in variables:
                    if "trans_prob" in v.name:
                        update = self._trans_prob_update(gamma, xi)
                    if "initial_state_probs" in v.name:
                        update = tf.reduce_mean(gamma[:, 0], axis=0)
                    update = tf.cast(update, tf.float32)
                    grads.append(update)
                return None, grads

            return (gamma, xi), grad

        return posterior(log_B)

    def compute_output_shape(self, input_shape):
        output_shape_0 = input_shape
        output_shape_1 = [input_shape[0], input_shape[1], self.n_states, self.n_states]
        return (output_shape_0, tuple(output_shape_1))


class SeparateLogLikelihoodLayer(layers.Layer):
    """Layer to calculate the log-likelihood for different HMM states.

    Parameters
    ----------
    n_states : int
        Number of states.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the keras.layers.Layer.
    """

    def __init__(self, n_states, **kwargs):
        super().__init__(**kwargs)
        self.n_states = n_states

    def call(self, inputs, **kwargs):
        x, mu, sigma = inputs
        # x.shape = (None, sequence_length, n_channels)
        # mu.shape = (None, n_states, n_channels)
        # sigma.shape = (None, n_states, n_channels, n_channels)

        # Add the sequence length dimension
        mu = tf.expand_dims(mu, axis=-3)
        sigma = tf.expand_dims(sigma, axis=-4)

        # Calculate log-likelihood for each state
        log_likelihood = tf.TensorArray(tf.float32, size=self.n_states)
        for state in range(self.n_states):
            mvn = tfp.distributions.MultivariateNormalTriL(
                loc=tf.gather(mu, state, axis=-2),
                scale_tril=tf.linalg.cholesky(tf.gather(sigma, state, axis=-3)),
                allow_nan_stats=False,
            )
            log_likelihood = log_likelihood.write(state, mvn.log_prob(x))
        log_likelihood = tf.transpose(log_likelihood.stack(), perm=[1, 2, 0])

        return log_likelihood  # shape = (None, sequence_length, n_states)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0]
        return output_shape


class SeparatePoissonLogLikelihoodLayer(layers.Layer):
    """Layer to calculate the log-likelihood for different HMM-Poisson states.

    Parameters
    ----------
    n_states : int
        Number of states.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the keras.layers.Layer.
    """

    def __init__(self, n_states, **kwargs):
        super().__init__(**kwargs)
        self.n_states = n_states

    def call(self, inputs, **kwargs):
        x, log_rate = inputs

        # Add the sequence length dimension
        log_rate = tf.expand_dims(tf.expand_dims(log_rate, axis=0), axis=0)

        # Add states dimension
        x = tf.expand_dims(x, axis=2)

        # Calculate log-likelihood for each state
        poi = tfp.distributions.Poisson(log_rate=log_rate, allow_nan_stats=False)
        log_likelihood = tf.reduce_sum(poi.log_prob(x), axis=-1)

        return log_likelihood  # shape = (None, sequence_length, n_states)


class SumLogLikelihoodLossLayer(layers.Layer):
    """Layer for summing log-likelihoods.

    Parameters
    ----------
    calculation : str
        Operation for reducing the time dimension. Either 'mean' or 'sum'.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to the keras.layers.Layer.
    """

    def __init__(self, calculation, **kwargs):
        super().__init__(**kwargs)
        self.calculation = calculation

    def call(self, inputs, **kwargs):
        ll, gamma = inputs
        ll_loss = tf.reduce_sum(gamma * ll, axis=-1)

        if self.calculation == "sum":
            # Sum over time dimension and average over the batch dimension
            ll_loss = tf.reduce_sum(ll_loss, axis=1)
            ll_loss = tf.reduce_mean(ll_loss, axis=0)
        else:
            # Average over time and batches
            ll_loss = tf.reduce_mean(ll_loss, axis=(0, 1))

        # Add the negative log-likelihood to the loss
        nll_loss = -ll_loss
        self.add_loss(nll_loss)

        return tf.expand_dims(nll_loss, axis=-1)


class EmbeddingLayer(layers.Layer):
    """Layer for embeddings.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    unit_norm : bool, optional
        Should the embeddings be unit norm?
    """

    def __init__(self, input_dim, output_dim, unit_norm, **kwargs):
        super().__init__(**kwargs)
        if unit_norm:
            output_dim = output_dim - 1

        self.embedding_layer = layers.Embedding(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        self.layers = [self.embedding_layer]
        self.unit_norm = unit_norm

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        output = self.embedding_layer(inputs)

        # Add the last element to ensure the embeddings are unit norm
        if self.unit_norm:
            norm_sq = tf.reduce_sum(tf.square(output), axis=-1, keepdims=True)
            output = tf.concat([2 * output, norm_sq - 1], axis=-1) / (norm_sq + 1)
        return output

    @property
    def embeddings(self):
        output = self.embedding_layer.embeddings
        if self.unit_norm:
            norm_sq = tf.reduce_sum(tf.square(output), axis=-1, keepdims=True)
            output = tf.concat([2 * output, norm_sq - 1], axis=-1) / (norm_sq + 1)
        return output


class ShiftForForecastingLayer(layers.Layer):
    """Clip two tensors to ensure they align for causal forecasting.

    Parameters
    ----------
    clip : int
        Number of elements to clip.
    """

    def __init__(self, clip, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip

    def call(self, inputs, **kwargs):
        A, B = inputs
        A = A[:, : -self.clip]
        B = B[:, self.clip :]
        return A, B


class SequentialLayer(layers.Layer):
    """Sequential layer.

    Parameters
    ----------
    layers : list
        List of layers.
    """

    def __init__(self, layers, **kwargs):
        super().__init__(**kwargs)
        self.layers = []
        for layer in layers:
            self.layers.append(layer)

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs
