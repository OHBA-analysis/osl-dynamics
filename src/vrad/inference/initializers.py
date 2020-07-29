"""Initializers for TensorFlow layers

A series of classes which inherit from `tensorflow.keras.initializers.Initializer`.
They are used to initialize weights in custom layers.

"""
import logging

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.initializers import Initializer
from vrad import inference

_logger = logging.getLogger("VRAD")


class Identity3D(Initializer):
    """Initializer to create stacked identity matrices.

    """

    def __call__(self, shape, dtype=None):
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


class CholeskyCovariancesInitializer(Initializer):
    """Initialize weights with provided variable, assuming dimensions are accepted.

    """

    def __init__(self, initial_cholesky_covariances):
        self.initial_cholesky_covariances = initial_cholesky_covariances

    def __call__(self, shape, dtype=None):
        if not (len(shape) == 3 and shape == self.initial_cholesky_covariances.shape):
            raise ValueError(
                f"shape must be 3D and be equal to initial_cholesky_covariances.shape. "
                f"shape == {[*shape]}, "
                f"initial_means.shape = {[*self.initial_cholesky_covariances.shape]} "
            )
        return self.initial_cholesky_covariances


class MeansInitializer(Initializer):
    """Initialize weights with provided variable, assuming dimensions are accepted.

    """

    def __init__(self, initial_means):
        self.initial_means = initial_means
        _logger.info(
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


def reinitialize_layer_weights(layer):
    """Re-initialises the weights in a particular layer.

    This function relies on each layer having an initializer attribute.
    Therefore, you must specific a self.*_initializer attribute in custom
    layers, otherwise this function will break.
    """
    # Get the initialisation container
    if hasattr(layer, "cell"):
        init_container = layer.cell
    else:
        init_container = layer

    # Re-initialise
    for key, initializer in init_container.__dict__.items():
        if "initializer" not in key:
            continue
        if key == "recurrent_initializer":
            var = getattr(init_container, "recurrent_kernel")
        else:
            var = getattr(init_container, key.replace("_initializer", ""))
        var.assign(initializer(var.shape, var.dtype))


def reinitialize_model_weights(model):
    """Re-initialises the weights in the model."""
    for layer in model.layers:
        # If the layer consists and multiple layers pass the layer back
        # to this function
        if (
            isinstance(layer, Model)
            or isinstance(layer, inference.layers.InferenceRNNLayers)
            or isinstance(layer, inference.layers.ModelRNNLayers)
        ):
            for l in layer.layers:
                # If the layer is bidirectional we need to re-initialise the
                # forward and backward layers
                if isinstance(l, layers.Bidirectional):
                    reinitialize_layer_weights(l.forward_layer)
                    reinitialize_layer_weights(l.backward_layer)
                # Otherwise, just re-initialise as a normal layer
                else:
                    reinitialize_layer_weights(l)

        # Otherwise, this is just a single layer
        else:
            reinitialize_layer_weights(layer)
