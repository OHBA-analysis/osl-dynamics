"""Initializers for TensorFlow layers.

"""

from tensorflow.keras import Model, layers
from tensorflow.keras.initializers import Initializer
from osl_dynamics import inference


class WeightInitializer(Initializer):
    """Initialize weights to given value.

    Parameters
    ----------
    initial_value : numpy.ndarray
        Value to initialise weights to.
        Note, the shape is not checked.
    """

    def __init__(self, initial_value):
        self.initial_value = initial_value

    def __call__(self, shape, dtype=None):
        return self.initial_value


class CopyTensorInitializer(Initializer):
    """Initialize weights to another Tensor's value.

    Parameters
    ----------
    tensor : tensorflow.Tensor
        Tensor to copy.
    """

    def __init__(self, tensor):
        self.tensor = tensor

    def __call__(self, shape, dtype=None):
        return self.tensor.read_value()


def reinitialize_layer_weights(layer):
    """Re-initializes the weights in a particular layer.

    This function relies on each layer having an initializer attribute.
    Therefore, you must specify a self.*_initializer attribute in custom
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


def reinitialize_model_weights(model, keep=None):
    """Re-initialize the weights in the model.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Model to re-initialize weights for.
    keep : list
        List of str containing names for layers to not reinitialize.
    """
    if keep is None:
        keep = []

    for layer in model.layers:
        # Skip layers that we want to keep
        if layer.name in keep:
            continue

        # If the layer consists and multiple layers pass the layer back
        # to this function
        if (
            isinstance(layer, Model)
            or isinstance(layer, inference.layers.InferenceRNNLayer)
            or isinstance(layer, inference.layers.ModelRNNLayer)
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
