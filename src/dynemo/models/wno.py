"""Class for a WaveNet observation model.

"""

import numpy as np
from tqdm import trange
from tensorflow.keras import Model, layers
from dynemo.models.layers import (
    WaveNetLayer,
    CovsLayer,
    MixCovsLayer,
    LogLikelihoodLayer,
)
from dynemo.models.obs_mod_base import ObservationModelBase


class WNO(ObservationModelBase):
    """WaveNet Observations (WNO) model.

    Parameters
    ----------
    config : dynemo.models.Config
    """

    def __init__(self, config):
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

    def get_covariances(self):
        """Learnt covariances.

        Returns
        -------
        np.ndarray
            Covariance.
        """
        covs_layer = self.model.get_layer("covs")
        return covs_layer(1).numpy()

    def sample(self, n_samples, covs, mode):
        """Sample data from the observation model.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        covs : float
            Covariances to use for sampling.
        mode : int
            Index for mode to sample.

        Returns
        -------
        np.ndarray
            Sample from the observation model.
        """

        # Get layer for the WaveNet model
        cnn_layer = self.model.get_layer("mean")

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
        a[0, :, mode] = 1
        cov = covs[mode]

        # Generate a sample
        s = np.empty([n_samples, self.config.n_channels])
        for i in trange(n_samples, desc=f"Sampling mode {mode}", ncols=98):
            mean = cnn_layer([x, a])[0, -1]
            x = np.roll(x, shift=-1, axis=1)
            x[0, -1] = np.random.multivariate_normal(mean, cov)
            s[i] = mean.numpy()

        return s


def _model_structure(config):

    # Layers for inputs
    inp_data = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )
    alpha = layers.Input(shape=(config.sequence_length, config.n_modes), name="alpha")

    # Definition of layers
    mean_layer = WaveNetLayer(
        config.n_channels,
        config.wavenet_n_filters,
        config.wavenet_n_layers,
        name="mean",
    )
    covs_layer = CovsLayer(
        config.n_modes,
        config.n_channels,
        config.diag_covs,
        config.learn_covariances,
        config.initial_covariances,
        name="covs",
    )
    mix_covs_layer = MixCovsLayer(name="cov")
    ll_loss_layer = LogLikelihoodLayer(clip=1, name="ll")

    # Data flow
    mean = mean_layer([inp_data, alpha])
    covs = covs_layer(inp_data)  # inp_data not used
    cov = mix_covs_layer([alpha, covs])
    ll_loss = ll_loss_layer([inp_data, mean, cov])

    return Model(inputs=[inp_data, alpha], outputs=[ll_loss], name="WNO")
