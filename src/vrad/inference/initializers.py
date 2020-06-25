"""Initializers for TensorFlow layers

A series of classes which inherit from `tensorflow.keras.initializers.Initializer`.
They are used to initialize weights in custom layers.

"""
import logging

import tensorflow as tf
from tensorflow.keras.initializers import Initializer

_logger = logging.getLogger("VRAD")


class Identity3D(Initializer):
    """Initializer to create stacked identity matrices.

    """

    def __call__(self, shape, **kwargs):
        """Create stacked identity matrices with dimensions [M x N x N]

        Parameters
        ----------
        shape : Iterable
            Shape of the Tensor [M x N x N].

        Returns
        -------
        stacked_identity_matrix : tf.Tensor
        """
        if len(shape) != 3 or shape[1] != shape[2]:
            raise ValueError("Weight shape must be [M x N x N]")

        return tf.eye(shape[1], shape[2], [shape[0]])


class PseudoSigmaInitializer(Initializer):
    """Initialize weights with provided variable, assuming dimensions are accepted.

    """

    def __init__(self, initial_pseudo_sigmas):
        self.initial_pseudo_sigmas = initial_pseudo_sigmas

    def __call__(self, shape, dtype=None):
        if not (len(shape) == 3 and shape == self.initial_pseudo_sigmas.shape):
            raise ValueError(
                f"shape must be 3D and be equal to initial_pseudo_sigmas.shape. "
                f"shape == {[*shape]}, "
                f"initial_means.shape = {[*self.initial_pseudo_sigmas.shape]} "
            )
        return self.initial_pseudo_sigmas


class MeansInitializer(Initializer):
    """Initialize weights with provided variable, assuming dimensions are accepted.

    """

    def __init__(self, initial_means):
        self.initial_means = initial_means
        logger.info(
            f"Creating MeansInitializer with "
            f"initial_means.shape = {initial_means.shape}"
        )

    def __call__(self, shape, dtype=None):
        if not (len(shape) == 2 and shape == self.initial_means.shape):
            raise ValueError(
                f"shape must be 2D and be equal to initial_means.shape. "
                f"shape == {[*shape]}, "
                f"initial_means.shape = {[*self.initial_means.shape]} "
            )
        return self.initial_means


class UnchangedInitializer(Initializer):
    """Initializer which returns unchanged values when called.

    """

    def __init__(self, initial_values):
        self.initial_values = initial_values

    def __call__(self, shape, dtype=None):
        return self.initial_values
