"""Class for a Gaussian observation model.

"""

import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras import Model, layers
from tensorflow.nn import softplus
from vrad.inference.functions import (
    cholesky_factor,
    cholesky_factor_to_full_matrix,
    trace_normalize,
)
from vrad.models.layers import LogLikelihoodLayer, MeansCovsLayer, MixMeansCovsLayer
from vrad.models.obs_mod_base import ObservationModelBase


class GO(ObservationModelBase):
    """Gaussian Observations (GO) model.

    Parameters
    ----------
    config : vrad.models.Config
    """

    def __init__(self, config):
        if config.observation_model != "multivariate_normal":
            raise ValueError("Observation model must be multivariate_normal.")

        ObservationModelBase.__init__(self, config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_covariances(self, alpha_scale=True):
        """Get the covariances of each state.

        Parameters
        ----------
        alpah_scale : bool
            Should we apply alpha scaling? Default is True.

        Returns
        -------
        np.ndarary
            State covariances.
        """
        # Get the means and covariances from the MeansCovsLayer
        means_covs_layer = self.model.get_layer("means_covs")
        cholesky_covariances = tfp.math.fill_triangular(
            means_covs_layer.flattened_cholesky_covariances
        )
        covariances = cholesky_factor_to_full_matrix(cholesky_covariances).numpy()

        # Normalise covariances
        if self.config.normalize_covariances:
            covariances = trace_normalize(covariances).numpy()

        # Apply alpha scaling
        if alpha_scale:
            alpha_scaling = self.get_alpha_scaling()
            covariances *= alpha_scaling[:, np.newaxis, np.newaxis]

        return covariances

    def set_covariances(self, covariances):
        """Set the covariances of each state.

        Parameters
        ----------
        covariances : np.ndarray
            State covariances.
        """
        means_covs_layer = self.model.get_layer("means_covs")
        layer_weights = means_covs_layer.get_weights()

        flattened_covariances_shape = (
            covariances.shape[0],
            covariances.shape[1] * (covariances.shape[1] + 1) // 2,
        )

        # Replace covariances in the layer weights
        for i in range(len(layer_weights)):
            if layer_weights[i].shape == flattened_covariances_shape:
                cholesky_covariances = cholesky_factor(covariances)
                flattened_cholesky_covariances = tfp.math.fill_triangular_inverse(
                    cholesky_covariances
                )
                layer_weights[i] = flattened_cholesky_covariances

        # Set the weights of the layer
        means_covs_layer.set_weights(layer_weights)

    def get_alpha_scaling(self):
        """Get the alpha scaling of each state.

        Returns
        ----------
        bool
            Alpha scaling for each state.
        """
        mix_means_covs_layer = self.model.get_layer("mix_means_covs")
        alpha_scaling = mix_means_covs_layer.alpha_scaling.numpy()
        alpha_scaling = softplus(alpha_scaling).numpy()
        return alpha_scaling


def _model_structure(config):

    # Layers for inputs
    data = layers.Input(shape=(config.sequence_length, config.n_channels), name="data")
    alpha = layers.Input(shape=(config.sequence_length, config.n_states), name="alpha")

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each state as the observation model.
    # - We calculate the likelihood of generating the training data with alpha
    #   and the observation model.

    # Definition of layers
    means_covs_layer = MeansCovsLayer(
        config.n_states,
        config.n_channels,
        learn_means=config.learn_means,
        learn_covariances=config.learn_covariances,
        normalize_covariances=config.normalize_covariances,
        initial_means=config.initial_means,
        initial_covariances=config.initial_covariances,
        name="means_covs",
    )
    mix_means_covs_layer = MixMeansCovsLayer(
        config.n_states,
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
