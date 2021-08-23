"""Class for a multivariate autoregressive observation model.

"""

import numpy as np
from tensorflow.keras import Model, layers, activations
from vrad.models.layers import (
    CoeffsCovsLayer,
    MixCoeffsCovsLayer,
    MARMeansCovsLayer,
    LogLikelihoodLayer,
)
from vrad.models.obs_mod_base import ObservationModelBase


class MARO(ObservationModelBase):
    """Multivariate Autoregressive Observations (MARO) model.

    Parameters
    ----------
    config : vrad.models.Config
    """

    def __init__(self, config):
        if config.observation_model != "multivariate_autoregressive":
            raise ValueError("Observation model must be multivariate_autoregressive.")

        ObservationModelBase.__init__(self, config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_params(self):
        """Get the parameters of the MAR model.

        Returns
        -------
        coeffs : np.ndarray
            MAR coefficients. Shape is (n_states, n_lags, n_channels, n_channels).
        cov : np.ndarray
            Mar covariance. Shape is (n_states, n_channels, n_channels).
        """
        coeffs_covs_layer = self.model.get_layer("coeffs_covs")
        coeffs = coeffs_covs_layer.coeffs.numpy()
        covs = np.array(
            [np.diag(activations.softplus(c)) for c in coeffs_covs_layer.covs.numpy()]
        )
        return coeffs, covs


def _model_structure(config):

    # Layers for inputs
    data = layers.Input(shape=(config.sequence_length, config.n_channels), name="data")
    alpha = layers.Input(shape=(config.sequence_length, config.n_states), name="alpha")

    # Observation model:
    # - We use x_t ~ N(mu_t, sigma_t), where
    #      - mu_t = Sum_j Sum_l alpha_jt coeffs_jlt x_{t-l}.
    #      - sigma_t = Sum_j alpha^2_jt cov_j, where cov_j is a learnable
    #        diagonal covariance matrix.
    # - We calculate the likelihood of generating the training data with alpha_jt
    #   and the observation model.

    # Definition of layers
    coeffs_covs_layer = CoeffsCovsLayer(
        config.n_states,
        config.n_channels,
        config.n_lags,
        config.initial_coeffs,
        config.initial_covs,
        config.learn_coeffs,
        config.learn_covs,
        name="coeffs_covs",
    )
    mix_coeffs_covs_layer = MixCoeffsCovsLayer(
        config.n_states,
        config.n_channels,
        config.sequence_length,
        config.n_lags,
        name="mix_coeffs_covs",
    )
    mar_means_covs_layer = MARMeansCovsLayer(
        config.n_channels,
        config.sequence_length,
        config.n_lags,
        name="means_covs",
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    coeffs_jl, covs_j = coeffs_covs_layer(data)  # data not used
    coeffs_lt, covs_t = mix_coeffs_covs_layer([alpha, coeffs_jl, covs_j])
    x_t, mu_t, sigma_t = mar_means_covs_layer([data, coeffs_lt, covs_t])
    ll_loss = ll_loss_layer([x_t, mu_t, sigma_t])

    return Model(inputs=[data, alpha], outputs=[ll_loss], name="MARO")
