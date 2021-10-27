"""Class for a convolutional neural network observation model.

"""

import numpy as np
from tqdm import trange
from tensorflow.keras import Model, layers
from dynemo.models.layers import WaveNetLayer, MeanSquaredErrorLayer
from dynemo.models.obs_mod_base import ObservationModelBase


class CNNO(ObservationModelBase):
    """Convoluational Neural Network Observations (CNNO) model.

    Parameters
    ----------
    config : dynemo.models.Config
    """

    def __init__(self, config):
        if config.observation_model != "wavenet":
            raise ValueError("Observation model must be wavenet.")

        ObservationModelBase.__init__(self, config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def loss(self, dataset):
        """Mean squared error loss.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to evaluate loss for.

        Returns
        -------
        float
            Mean squared error loss.
        """
        losses = self.model.predict(dataset)
        return np.mean(losses)

    def sample(self, n_samples, std_dev, alpha):
        """Sample from the observation model.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        std_dev : float
            Standard deviation to use for sampling.
        alpha : int
            Index for mode to sample.

        Returns
        -------
        np.ndarray
            Sample from the observation model.
        """
        cnn_layer = self.model.get_layer("wavenet")

        x = np.zeros(
            [1, self.config.sequence_length, self.config.n_channels], dtype=np.float32
        )
        x[0, -1] = np.random.normal(size=self.config.n_channels)

        a = np.zeros(
            [1, self.config.sequence_length, self.config.n_modes], dtype=np.float32
        )
        a[:, alpha] = 1

        d = np.empty([n_samples, self.config.n_channels])
        for i in trange(n_samples, desc=f"Sampling mode {alpha}", ncols=98):
            y = cnn_layer([x, a])[0, -1]
            x = np.roll(x, shift=-1, axis=1)
            x[0, -1] = y + np.random.normal(scale=std_dev)
            d[i] = y.numpy()

        return d


def _model_structure(config):

    # Layers for inputs
    inp_data = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )
    alpha = layers.Input(shape=(config.sequence_length, config.n_modes), name="alpha")

    # Definition of layers
    cnn_obs_layer = WaveNetLayer(
        config.n_channels,
        config.wavenet_n_filters,
        config.wavenet_n_layers,
        name="wavenet",
    )
    mse_layer = MeanSquaredErrorLayer(clip=1, name="mse")

    # Data flow
    gen_data = cnn_obs_layer([inp_data, alpha])
    mse = mse_layer([inp_data, gen_data])

    return Model(inputs=[inp_data, alpha], outputs=[mse], name="CNNO")
