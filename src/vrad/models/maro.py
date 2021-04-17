"""Class for a multivariate autoregressive observation model.

"""

import numpy as np
from tensorflow.keras import Model, layers
from vrad.models.obs_mod_base import ObservationModelBase
from vrad.models.layers import LogLikelihoodLayer, MARMeanCovLayer, MARParametersLayer


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
        mar_params_layer = self.model.get_layer("mar_params")
        coeffs = mar_params_layer.coeffs.numpy()
        cov = np.array([np.diag(c) for c in mar_params_layer.cov.numpy()])
        return coeffs, cov

    def set_params(self, coeffs, cov):
        """Set the parameters of the MAR model.

        Parameters
        ----------
        coeffs : np.ndarray
            MAR coefficients. Shape is (n_states, n_lags, n_channels, n_channels).
        cov : np.ndarray
            Mar covariance. Shape is (n_states, n_channels, n_channels).
        """
        mar_params_layer = self.model.get_layer("mar_params")
        layer_weights = mar_params_layer.get_weights()
        cov = np.array([np.diag(c) for c in cov])
        for i in range(len(layer_weights)):
            if layer_weights[i].shape == coeffs.shape:
                layer_weights[i] = coeffs
            if layer_weights[i].shape == cov.shape:
                layer_weights[i] = cov
        mar_params_layer.set_weights(layer_weights)


def _model_structure(config):

    # Layers for inputs
    data = layers.Input(shape=(config.sequence_length, config.n_channels), name="data")
    alpha = layers.Input(shape=(config.sequence_length, config.n_states), name="alpha")

    # Observation model:
    # - We use x_t ~ N(mu_t, sigma_t), where
    #      - mu_t = Sum_j Sum_l alpha_jt coeffs_jt data_{t-l}.
    #      - sigma_t = Sum_j alpha^2_jt cov_j, where cov_j is a learnable
    #        diagonal covariance matrix.
    # - We calculate the likelihood of generating the training data with alpha_jt
    #   and the observation model.

    # Definition of layers
    mar_params_layer = MARParametersLayer(
        config.n_states,
        config.n_channels,
        config.n_lags,
        config.initial_coeffs,
        config.initial_cov,
        config.learn_coeffs,
        config.learn_cov,
        name="mar_params",
    )
    mean_cov_layer = MARMeanCovLayer(
        config.n_states,
        config.n_channels,
        config.sequence_length,
        config.n_lags,
        name="mean_cov",
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    coeffs, cov = mar_params_layer(data)  # data not used
    clipped_data, mu, sigma = mean_cov_layer([data, alpha, coeffs, cov])
    ll_loss = ll_loss_layer([clipped_data, mu, sigma])

    return Model(inputs=[data, alpha], outputs=[ll_loss], name="MARO")
