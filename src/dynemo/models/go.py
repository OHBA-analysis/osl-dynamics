"""Class for a Gaussian observation model.

"""

import numpy as np
from tensorflow.keras import Model, layers
from tensorflow.nn import softplus
from dynemo.models.layers import LogLikelihoodLayer, MeansCovsLayer, MixMeansCovsLayer
from dynemo.models.obs_mod_base import ObservationModelBase


class GO(ObservationModelBase):
    """Gaussian Observations (GO) model.

    Parameters
    ----------
    config : dynemo.models.Config
    """

    def __init__(self, config):
        if config.observation_model != "multivariate_normal":
            raise ValueError("Observation model must be multivariate_normal.")

        ObservationModelBase.__init__(self, config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_covariances(self, alpha_scale=True):
        """Get the covariances of each mode.

        Parameters
        ----------
        alpah_scale : bool
            Should we apply alpha scaling? Default is True.

        Returns
        -------
        np.ndarary
            Mode covariances.
        """
        # Get the means and covariances from the MeansCovsLayer
        means_covs_layer = self.model.get_layer("means_covs")
        _, covariances = means_covs_layer(1)
        covariances = covariances.numpy()

        # Apply alpha scaling
        if alpha_scale:
            alpha_scaling = self.get_alpha_scaling()
            covariances *= alpha_scaling[:, np.newaxis, np.newaxis]

        return covariances

    def get_means_covariances(self, alpha_scale=True):
        """Get the means and covariances of each mode.

        Parameters
        ----------
        alpah_scale : bool
            Should we apply alpha scaling? Default is True.

        Returns
        -------
        means : np.ndarary
            Mode means.
        covariances : np.ndarray
            Mode covariances.
        """
        # Get the means and covariances from the MeansCovsLayer
        means_covs_layer = self.model.get_layer("means_covs")
        means, covariances = means_covs_layer(1)

        # Apply alpha scaling
        if alpha_scale:
            alpha_scaling = self.get_alpha_scaling()
            covariances *= alpha_scaling[:, np.newaxis, np.newaxis]

        return means.numpy(), covariances.numpy()

    def set_means(self, means, update_initializer=True):
        """Set the means of each mode.

        Parameters
        ----------
        means : np.ndarray
            Mode covariances.
        update_initializer : bool
            Do we want to use the passed means when we re-initialize
            the model? Optional, default is True.
        """
        means = means.astype(np.float32)
        means_covs_layer = self.model.get_layer("means_covs")
        layer_weights = means_covs_layer.means
        layer_weights.assign(means)

        if update_initializer:
            means_covs_layer.initial_means = means
            means_covs_layer.means_initializer.initial_value = means

    def set_covariances(self, covariances, update_initializer=True):
        """Set the covariances of each mode.

        Parameters
        ----------
        covariances : np.ndarray
            Mode covariances.
        update_initializer : bool
            Do we want to use the passed covariances when we re-initialize
            the model? Optional, default is True.
        """
        covariances = covariances.astype(np.float32)
        means_covs_layer = self.model.get_layer("means_covs")
        layer_weights = means_covs_layer.flattened_cholesky_covariances
        flattened_cholesky_covariances = means_covs_layer.bijector.inverse(covariances)
        layer_weights.assign(flattened_cholesky_covariances)

        if update_initializer:
            means_covs_layer.initial_covariances = covariances
            means_covs_layer.initial_flattened_cholesky_covariances = (
                flattened_cholesky_covariances
            )
            means_covs_layer.flattened_cholesky_covariances_initializer.initial_value = (
                flattened_cholesky_covariances
            )

    def get_alpha_scaling(self):
        """Get the alpha scaling of each mode.

        Returns
        ----------
        bool
            Alpha scaling for each mode.
        """
        mix_means_covs_layer = self.model.get_layer("mix_means_covs")
        alpha_scaling = mix_means_covs_layer.alpha_scaling.numpy()
        alpha_scaling = softplus(alpha_scaling).numpy()
        return alpha_scaling


def _model_structure(config):

    # Layers for inputs
    data = layers.Input(shape=(config.sequence_length, config.n_channels), name="data")
    alpha = layers.Input(shape=(config.sequence_length, config.n_modes), name="alpha")

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each mode as the observation model.
    # - We calculate the likelihood of generating the training data with alpha
    #   and the observation model.

    # Definition of layers
    means_covs_layer = MeansCovsLayer(
        config.n_modes,
        config.n_channels,
        learn_means=config.learn_means,
        learn_covariances=config.learn_covariances,
        normalize_covariances=config.normalize_covariances,
        initial_means=config.initial_means,
        initial_covariances=config.initial_covariances,
        name="means_covs",
    )
    mix_means_covs_layer = MixMeansCovsLayer(
        config.n_modes,
        config.n_channels,
        config.learn_alpha_scaling,
        name="mix_means_covs",
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    mu, D = means_covs_layer(data)  # data not used
    m, C = mix_means_covs_layer([alpha, mu, D])
    ll_loss = ll_loss_layer([data, m, C])

    return Model(inputs=[data, alpha], outputs=[ll_loss], name="GO")
