"""Class for a convolutional neural network observation model.

"""

import numpy as np
from tqdm import trange
from tensorflow.keras import Model, layers
from vrad.models.layers import ConvNetObservationsLayer, MeanSquaredErrorLayer
from vrad.models.obs_mod_base import ObservationModelBase


class CNNO(ObservationModelBase):
    """Convoluational Neural Network Observations (CNNO) model.

    Parameters
    ----------
    config : vrad.models.Config
    """

    def __init__(self, config):
        if config.observation_model != "conv_net":
            raise ValueError("Observation model must be conv_net.")

        ObservationModelBase.__init__(self, config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)


def _model_structure(config):

    # Layers for inputs
    inp_data = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )
    alpha = layers.Input(shape=(config.sequence_length, config.n_states), name="alpha")

    # Definition of layers
    cnn_obs_layer = ConvNetObservationsLayer(config.n_channels, name="conv_net")
    mse_layer = MeanSquaredErrorLayer(clip=1, name="mse")

    # Data flow
    gen_data = cnn_obs_layer(inp_data)
    mse = mse_layer([inp_data, gen_data])

    return Model(inputs=[inp_data, alpha], outputs=[mse], name="CNNO")
