"""Initializers for TensorFlow layers.

"""
import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, layers
from tensorflow.keras.initializers import Initializer
from vrad import models

_logger = logging.getLogger("VRAD")


class FlattenedCholeskyCovariancesInitializer(Initializer):
    """Initialize weights for flattened cholesky factor of covariances.

    Provided variables are used for initialization assuming dimensions are accepted.

    Parameters
    ----------
    initial_cholesky_covariances : np.ndarray
        Cholesky factors of the state covariance matrices used for initialization.
        Shape must be (n_states, n_channels, n_channels).
    """

    def __init__(self, initial_cholesky_covariances: np.ndarray):
        self.initial_cholesky_covariances = initial_cholesky_covariances

    def __call__(self, shape, dtype=None):
        return tfp.math.fill_triangular_inverse(self.initial_cholesky_covariances)


class MeansInitializer(Initializer):
    """Initialize weights for means.

    Provided variables are used for initialization assuming dimensions are accepted.

    Parameters
    ----------
    initial_means : np.ndarray
        State mean vectors used for initialization.
        Shape must be [n_states, n_channels].
    """

    def __init__(self, initial_means: np.ndarray):
        self.initial_means = initial_means

    def __call__(self, shape, dtype=None):
        return self.initial_means


def reinitialize_layer_weights(layer: tf.keras.layers.Layer):
    """Re-initializes the weights in a particular layer.

    This function relies on each layer having an initializer attribute.
    Therefore, you must specific a self.*_initializer attribute in custom
    layers, otherwise this function will break.

    Parameters
    ----------
    layer: tensorflow.keras.layers.Layer
        Layer to initialize weights for.
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


def reinitialize_model_weights(model: tf.keras.Model):
    """Re-initialize the weights in the model.

    Parameters
    ----------
    model: tensorflow.keras.Model
        Model to re-initialize weights for.
    """

    for layer in model.layers:
        # If the layer consists and multiple layers pass the layer back
        # to this function
        if (
            isinstance(layer, Model)
            or isinstance(layer, models.InferenceRNNLayers)
            or isinstance(layer, models.ModelRNNLayers)
        ):
            for rnn_or_model_layer in layer.layers:
                # If the layer is bidirectional we need to re-initialise the
                # forward and backward layers
                if isinstance(rnn_or_model_layer, layers.Bidirectional):
                    reinitialize_layer_weights(rnn_or_model_layer.forward_layer)
                    reinitialize_layer_weights(rnn_or_model_layer.backward_layer)
                # Otherwise, just re-initialise as a normal layer
                else:
                    reinitialize_layer_weights(rnn_or_model_layer)

        # Otherwise, this is just a single layer
        else:
            reinitialize_layer_weights(layer)
