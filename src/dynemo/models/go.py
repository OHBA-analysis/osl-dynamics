"""Class for a Gaussian observation model.

"""

import numpy as np
from tensorflow.keras import Model, layers
from tensorflow.nn import softplus
from dynemo.models.layers import (
    LogLikelihoodLossLayer,
    VectorsLayer,
    MatricesLayer,
    MixVectorsLayer,
    MixMatricesLayer,
)
from dynemo.models.obs_mod_base import ObservationModelBase


class GO(ObservationModelBase):
    """Gaussian Observations (GO) model.

    Parameters
    ----------
    config : dynemo.models.Config
    """

    def __init__(self, config):
        ObservationModelBase.__init__(self, config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_covariances(self):
        """Get the covariances of each mode.

        Returns
        -------
        np.ndarary
            Mode covariances.
        """
        covs_layer = self.model.get_layer("covs")
        covs = covs_layer(1)
        return covs.numpy()

    def get_means_covariances(self):
        """Get the means and covariances of each mode.

        Returns
        -------
        means : np.ndarary
            Mode means.
        covariances : np.ndarray
            Mode covariances.
        """
        means_layer = self.model.get_layer("means")
        covs_layer = self.model.get_layer("covs")
        means = means_layer(1)
        covs = covs_layer(1)
        return means.numpy(), covs.numpy()

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
        means_layer = self.model.get_layer("means")
        layer_weights = means_layer.means
        layer_weights.assign(means)

        if update_initializer:
            means_layer.initial_value = means
            means_covs_layer.vectors_initializer.initial_value = means

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
        covs_layer = self.model.get_layer("covs")
        layer_weights = covs_layer.flattened_cholesky_matrices
        flattened_cholesky_matrices = covs_layer.bijector.inverse(covariances)
        layer_weights.assign(flattened_cholesky_matrices)

        if update_initializer:
            covs_layer.initial_value = covariances
            covs_layer.initial_flattened_cholesky_matrices = (
                flattened_cholesky_matricecs
            )
            covs_layer.flattened_cholesky_matrices_initializer.initial_value = (
                flattened_cholesky_matrices
            )


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
    means_layer = VectorsLayer(
        config.n_modes,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        name="means",
    )
    covs_layer = MatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_covariances,
        config.initial_covariances,
        diag_only=False,
        name="covs",
    )
    mix_means_layer = MixVectorsLayer(name="mix_means")
    mix_covs_layer = MixMatricesLayer(name="mix_covs")
    ll_loss_layer = LogLikelihoodLossLayer(name="ll_loss")

    # Data flow
    mu = means_layer(data)  # data not used
    D = covs_layer(data)  # data not used
    m = mix_means_layer([alpha, mu])
    C = mix_covs_layer([alpha, D])
    ll_loss = ll_loss_layer([data, m, C])

    return Model(inputs=[data, alpha], outputs=[ll_loss], name="GO")
