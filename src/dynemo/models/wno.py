"""Class for a WaveNet observation model.

"""

import numpy as np
from tqdm import trange
from tensorflow.keras import Model, layers
from dynemo.models.layers import WaveNetLayer, StdDevLayer, LogLikelihoodLayer
from dynemo.models.obs_mod_base import ObservationModelBase


class WNO(ObservationModelBase):
    """WaveNet Observations (WNO) model.

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
        """Negative log-likelihood loss.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to evaluate loss for.

        Returns
        -------
        float
            Negative log-likelihood loss.
        """
        losses = self.model.predict(dataset)
        return np.mean(losses)

    def get_std_dev(self):
        """Learnt standard deviation.

        Returns
        np.ndarray
            Standard deviation.
        """
        std_dev_layer = self.model.get_layer("std_dev")
        return std_dev_layer(1).numpy()

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

        # Get layer for the WaveNet model
        cnn_layer = self.model.get_layer("means")

        # Historic data to input to WaveNet
        x = np.zeros(
            [1, self.config.sequence_length, self.config.n_channels], dtype=np.float32
        )
        x[0, -1] = np.random.normal(size=self.config.n_channels)

        # Mode mixing factor vector for local conditioning,
        # we select one mode to be active
        a = np.zeros(
            [1, self.config.sequence_length, self.config.n_modes], dtype=np.float32
        )
        a[0, :, alpha] = 1

        # Generate a sample
        s = np.empty([n_samples, self.config.n_channels])
        for i in trange(n_samples, desc=f"Sampling mode {alpha}", ncols=98):
            y = cnn_layer([x, a])[0, -1]
            x = np.roll(x, shift=-1, axis=1)
            x[0, -1] = y + np.random.normal(scale=std_dev)
            s[i] = y.numpy()

        return s


def _model_structure(config):

    # Layers for inputs
    inp_data = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )
    alpha = layers.Input(shape=(config.sequence_length, config.n_modes), name="alpha")

    # Definition of layers
    means_layer = WaveNetLayer(
        config.n_channels,
        config.wavenet_n_filters,
        config.wavenet_n_layers,
        name="means",
    )
    std_devs_layer = StdDevLayer(config.n_channels, learn_std_dev=True, name="std_dev")
    ll_loss_layer = LogLikelihoodLayer(diag_only=True, clip=1, name="ll")

    # Data flow
    means = means_layer([inp_data, alpha])
    std_devs = std_devs_layer(inp_data)  # inp_data not used
    ll_loss = ll_loss_layer([inp_data, means, std_devs])

    return Model(inputs=[inp_data, alpha], outputs=[ll_loss], name="WNO")
