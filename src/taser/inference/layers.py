import logging
import warnings
from typing import List, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from taser.inference.inference_functions import (
    normalise_covariance,
    pseudo_sigma_to_sigma,
)
from taser.inference.initializers import (
    Identity3D,
    MeansInitializer,
    PseudoSigmaInitializer,
    UnchangedInitializer,
)
from tensorflow.python.keras.utils import tf_utils


class ReparameterizationLayer(Layer):
    """Resample a normal distribution.

    The reparameterization trick is used to provide a differentiable random sample.
    By optimizing the values which define the distribution (i.e. mu and sigma), a model
    can be trained while still maintaining its stochasticity.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs: Union[tf.Tensor, tf.Tensor], **kwargs):
        """Given the mean and log standard deviation, create a normal distribution.

        Parameters
        ----------
        inputs : Union[tf.Tensor, tf.Tensor]
            Mean and log of standard deviation.
        kwargs

        Returns
        -------
        new_distribution : tf.Tensor
            The newly sampled normal distribution created from its mean and the log
            of its standard deviation.
        """
        z_mean, z_log_var = inputs
        new_distribution = tf.random.normal(
            z_mean.shape, mean=z_mean, stddev=tf.exp(0.5 * z_log_var)
        )
        return new_distribution


class MVNLayer(Layer):
    """`Layer` for storing a means and standard deviations of a multivariate normal.

    This layer doesn't act on input tensors. It only returns its internal state,
    allowing means and standard deviations to be used to parameterize a
    multivariate normal distribution.

    Parameters
    ----------
    num_gaussians : int
        The number of independent gaussian distributions.
    dim : int
        The dimensions of the space containing the gaussians.
    learn_means : bool
        If True, means are trainable.
    learn_covariances : bool
        If True, covariances are trainable.
    initial_means : tf.Tensor
        Starting values (priors) for the means. Default is all zero.
    initial_pseudo_sigmas : tf.Tensor
        Starting values (priors) for the covariances.
        Default is num_gaussians x Identity(dim x dim).
    """

    def __init__(
        self,
        num_gaussians: int,
        dim: int,
        learn_means: bool = True,
        learn_covariances: bool = True,
        initial_means: tf.Tensor = None,
        initial_pseudo_sigmas: tf.Tensor = None,
        initial_sigmas: tf.Tensor = None,
        **kwargs,
    ):

        if not ((initial_pseudo_sigmas is None) != (initial_sigmas is None)):
            raise ValueError(
                "Exactly one of initial_pseudo_sigmas "
                "and initial_sigmas may be specified."
            )

        super().__init__(**kwargs)
        self.num_gaussians = num_gaussians
        self.dim = dim
        self.learn_means = learn_means
        self.learn_covariances = learn_covariances
        self.initial_means = initial_means
        self.initial_pseudo_sigmas = initial_pseudo_sigmas
        self.initial_true_sigmas = initial_sigmas

        if self.initial_means is None:
            self.means_initializer = tf.keras.initializers.Zeros
        else:
            self.means_initializer = MeansInitializer(self.initial_means)

        if self.initial_pseudo_sigmas is not None:
            self.sigmas_initializer = PseudoSigmaInitializer(self.initial_pseudo_sigmas)
        if self.initial_true_sigmas is not None:
            self.sigmas_initializer = UnchangedInitializer(self.initial_true_sigmas)
        else:
            self.sigmas_initializer = Identity3D

        self.means = None
        self.pseudo_sigmas = None
        self.sigmas = None

    def build(self, input_shape):
        """Add weights to layer using initializers

        Parameters
        ----------
        input_shape
        """
        self.means = self.add_weight(
            "means",
            shape=(self.num_gaussians, self.dim),
            initializer=self.means_initializer,
            trainable=self.learn_means,
        )
        self.pseudo_sigmas = self.add_weight(
            "pseudo_sigmas",
            shape=(self.num_gaussians, self.dim, self.dim),
            initializer=self.sigmas_initializer,
            trainable=self.learn_covariances,
        )
        super().build(input_shape)

    def compute_output_shape(self, input_shape) -> List[tf.TensorShape]:
        """Compute output shape

        Parameters
        ----------
        input_shape
            Not used internally.

        Returns
        -------
        output_shape : List[tf.TensorShape]
        """
        return [
            tf.TensorShape([self.num_gaussians, self.dim]),
            tf.TensorShape([self.num_gaussians, self.dim, self.dim]),
        ]

    def call(self, inputs, burn_in=False, **kwargs):
        """

        Parameters
        ----------
        inputs
            Not used, but required to allow eager execution
        kwargs

        Returns
        -------
        means : tf.Tensor
            Means of a multivariate normal distribution
        sigmas : tf.Tensor
            Standard deviations of a multivariate normal distribution

        """

        def no_grad():
            result = tf.stop_gradient(normalise_covariance(self.pseudo_sigmas))
            return result

        self.sigmas = tf_utils.smart_cond(
            burn_in,
            no_grad,
            lambda: normalise_covariance(pseudo_sigma_to_sigma(self.pseudo_sigmas)),
        )

        # self.sigmas = normalise_covariance(pseudo_sigma_to_sigma(self.pseudo_sigmas))
        # self.sigmas = tf.stop_gradient(normalise_covariance(self.pseudo_sigmas))
        return self.means, self.sigmas


class TrainableVariablesLayer(Layer):
    """More abstract `Layer` allowing for calls which don't act on inputs.

    This layer holds its own values and provides then to a model when called. It
    doesn't process any inputs provided to it - merely returns its internal state.

    Parameters
    ----------
    shape : List[int]
    initial_values : tf.Tensor
    trainable : bool
    kwargs

    """

    def __init__(self, shape, initial_values=None, trainable=True, **kwargs):

        super().__init__(**kwargs)
        self.shape = shape
        self.initial_values = initial_values
        self.trainable = trainable

        self.values = None

        if self.initial_values is None:
            self.values_initializer = tf.keras.initializers.Zeros
        else:
            self.values_initializer = UnchangedInitializer(self.initial_values)

    def build(self, input_shape):
        """Add weights to layer.

        Parameters
        ----------
        input_shape
        """
        self.values = self.add_weight(
            "values",
            shape=self.shape,
            initializer=self.values_initializer,
            trainable=self.trainable,
        )
        super().build(self.shape)

    def compute_output_shape(self, input_shape):
        """Provide the output shape of the layer.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Not used internally.

        Returns
        -------
        output_shape : tf.TensorShape
            The shape specified during layer creation.

        """
        return tf.zeros(self.shape).shape

    def call(self, inputs, **kwargs):
        """Return internal values.

        Parameters
        ----------
        inputs : tf.Tensor
            Not used.
        kwargs

        Returns
        -------
        internal_values : tf.Tensor
        """
        return self.values
