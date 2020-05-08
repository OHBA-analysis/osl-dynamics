"""TensorFlow layers for calculating losses.

"""
import functools
from typing import List

import tensorflow as tf
from tensorflow.keras.activations import linear, softmax, softplus
from tensorflow.keras.layers import Layer

from taser.losses.functions import normal_kl_divergence


class LogLikelihoodLayer(Layer):
    """Layer for calculating the log likelihood loss.

    Parameters
    ----------
    n_states : int
    n_channels : int
    alpha_xform : str
        The function to apply to the resampled normal distribution `theta_ast`.
        Options are ["softplus", "softmax", "none", "linear" (default)]. "none" and
        "linear" leave their inputs unchanged.
    kwargs
    """

    alpha_xforms = {
        "softplus": softplus,
        "softmax": functools.partial(softmax, axis=2),
        "none": linear,
        "linear": linear,
    }

    def __init__(
        self, n_states: int, n_channels: int, alpha_xform: str = "linear", **kwargs
    ):

        super().__init__(**kwargs)

        self.n_states = n_states
        self.n_channels = n_channels
        self.run_eagerly = True

        try:
            self.alpha_xform = self.alpha_xforms[alpha_xform]
        except KeyError:
            print(
                f"available options for alpha_xform are: {', '.join(self.alpha_xforms)}"
            )
            raise

    def compute_output_shape(self, input_shape):
        """Compute output shape

        Parameters
        ----------
        input_shape : tf.TensorShape
            Not used internally.

        Returns
        -------
        output_shape : tf.TensorShape
            Always returns a shape of 1.
        """
        return tf.TensorShape([1])

    def call(self, inputs: List[tf.Tensor], **kwargs):
        """Calculate the log likelihood loss of a set of inputs.

        Parameters
        ----------
        inputs : List[tf.Tensor]
            List of Tensors formed of:
                - data batch
                - resampled distribution
                - matrix of means
                - matrix of covariances
        kwargs

        Returns
        -------
        log_likelihood_loss : tf.Tensor
            Scalar tensor containing the log_likelihood_loss of the model and dataset.
        """
        y_portioned, theta_ast, mean_matrix, covariance_matrix = inputs

        alpha_ext = self.alpha_xform(theta_ast)[..., tf.newaxis]
        mean_ext = mean_matrix[tf.newaxis, tf.newaxis, ...]

        mn_arg = tf.reduce_sum(alpha_ext * mean_ext, axis=2)

        alpha_ext = self.alpha_xform(theta_ast)[..., tf.newaxis, tf.newaxis]
        covariance_ext = covariance_matrix[tf.newaxis, tf.newaxis, ...]

        cov_arg = tf.reduce_sum(alpha_ext * covariance_ext, axis=2)

        safety_add = 1e-8 * tf.eye(self.n_channels)
        cov_arg += safety_add

        inv_cov_arg = tf.linalg.inv(cov_arg)
        log_det = -0.5 * tf.linalg.logdet(cov_arg)

        y_exp = y_portioned[:, :, tf.newaxis, ...]
        mn_exp = mn_arg[:, :, tf.newaxis, ...]

        tmp = y_exp - mn_exp

        attempt = -0.5 * (tmp @ inv_cov_arg) @ tf.transpose(tmp, perm=[0, 1, 3, 2])

        log_likelihood = -tf.reduce_sum(log_det + tf.squeeze(attempt))

        return log_likelihood


class KLDivergenceLayer(Layer):
    """Layer for calculating the KL divergence loss.

    Parameters
    ----------
    n_states : int
    n_channels : int
    kwargs

    """

    def __init__(self, n_states: int, n_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.n_channels = n_channels
        self.run_eagerly = True

    def call(self, inputs: List[tf.Tensor], **kwargs):
        """Calculate the KL divergence loss.

        Given a the means and standard deviations of two univariate normal
        distributions, calculate their KL divergence.

        Parameters
        ----------
        inputs : List[tf.Tensor]
            List of Tensors formed of:
                - Inference mean
                - Inference standard deviation
                - Model mean
                - Model standard deviation
        kwargs

        Returns
        -------
        kl_divergence_loss : tf.Tensor
            Scalar Tensor containing the KL divergence loss of the two input
            distributions.
        """
        inference_mu, inference_sigma, model_mu, log_sigma_theta_j = inputs

        shifted_model_mu = tf.roll(model_mu, shift=1, axis=1)

        log_sigma_theta_j_ext = log_sigma_theta_j[tf.newaxis, tf.newaxis, ...]

        kl_divergence = tf.reduce_sum(
            normal_kl_divergence(
                inference_mu, inference_sigma, shifted_model_mu, log_sigma_theta_j_ext
            )
        )

        return kl_divergence

    def compute_output_shape(self, input_shape: tf.TensorShape):
        """Compute output shape

        Parameters
        ----------
        input_shape : tf.TensorShape
            Not used internally.

        Returns
        -------
        output_shape : tf.TensorShape
            Always returns a scalar tensor (tf.TensorShape([1]))
        """
        return tf.TensorShape([1])
