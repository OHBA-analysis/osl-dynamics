"""Model class for a multi-time-scale generative model with Gaussian observations.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from dynemo.models.inf_mod_base import InferenceModelBase
from dynemo.models.obs_mod_base import ObservationModelBase
from dynemo.models.layers import (
    InferenceRNNLayers,
    LogLikelihoodLossLayer,
    MeanVectorsLayer,
    DiagonalMatricesLayer,
    CovarianceMatricesLayer,
    MixMeansStdsFcsLayer,
    ModelRNNLayers,
    NormalizationLayer,
    KLDivergenceLayer,
    KLLossLayer,
    SampleNormalDistributionLayer,
    ThetaActivationLayer,
    FillConstantLayer,
    DummyLayer,
)


class MRIGO(InferenceModelBase, ObservationModelBase):
    """Multi-time-scale RNN Inference/model network and Gaussian
    Observations (MRIGO).

    Parameters
    ----------
    config : dynemo.models.Config
    """

    def __init__(self, config):
        InferenceModelBase.__init__(self)
        ObservationModelBase.__init__(self, config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_means_stds_fcs(self):
        """Get the mean, standard devation and functional connectivity of each mode.

        Returns
        -------
        means : np.ndarray
            Mode means.
        stds : np.ndarray
            Mode standard deviations.
        fcs : np.ndarray
            Mode functional connectivities.
        """
        means_layer = self.model.get_layer("means")
        stds_layer = self.model.get_layer("stds")
        fcs_layer = self.model.get_layer("fcs")
        means = means_layer(1)
        stds = stds_layer(1)
        fcs = fcs_layer(1)
        return means.numpy(), stds.numpy(), fcs.numpy()

    def set_means_stds_fcs(self, means, stds, fcs, update_initializer=True):
        """Set the means, standard deviations, functional connectivities of each mode.

        Parameters
        ----------
        means: np.ndarray
            Mode means with shape (n_modes, n_channels).
        stds: np.ndarray
            Mode standard deviations with shape (n_modes, n_channels) or
            (n_modes, n_channels, n_channels).
        fcs: np.ndarray
            Mode functional connectivities with shape (n_modes, n_channels, n_channels).
        update_initializer: bool
            Do we want to use the passed parameters when we re_initialize
            the model? Optional.
        """
        if stds.ndim == 3:
            # Only keep the diagonal as a vector
            stds = np.diagonal(stds, axis1=1, axis2=2)

        means = means.astype(np.float32)
        stds = stds.astype(np.float32)
        fcs = fcs.astype(np.float32)

        # Transform the matrices to layer weights
        stds = stds_layer.bijector.inverse(stds)
        flattened_cholesky_factors = fcs_layer.bijector.inverse(fcs)

        # Get layers
        means_layer = self.model.get_layer("means")
        stds_layer = self.model.get_layer("stds")
        fcs_layer = self.model.get_layer("fcs")

        # Set values
        means_layer.vectors.assign(means)
        stds_layer.diagonals.assign(stds)
        fcs_layer.flattened_cholesky_factors.assign(flattened_cholesky_factors)

        # Update initialisers
        if update_initializer:
            means_layer.initial_value = means
            stds_layer.initial_value = stds
            fcs_layer.initial_value = fcs
            fcs_layer.initial_flattened_cholesky_factors = flattened_cholesky_factors

            means_layer.vectors_initializer.initial_value = means
            stds_layer.diagonals_initializer.initial_value = stds
            fcs_layer.flattened_cholesky_factors_initializer.initial_value = (
                flattened_cholesky_factors
            )


def _model_structure(config):

    # Layer for input
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )

    #
    # Inference RNN
    #

    # Layers
    inference_input_dropout_layer = layers.Dropout(
        config.inference_dropout_rate, name="data_drop_mean"
    )
    inference_output_layers = InferenceRNNLayers(
        config.inference_rnn,
        config.inference_normalization,
        config.inference_activation,
        config.inference_n_layers,
        config.inference_n_units,
        config.inference_dropout_rate,
        name="inf_rnn_mean",
    )

    # Data flow
    inference_input_dropout = inference_input_dropout_layer(inputs)
    inference_output = inference_output_layers(inference_input_dropout)

    #
    # Mode time course for the mean
    #

    # Layers
    mean_inf_mu_layer = layers.Dense(config.n_modes, name="mean_inf_mu")
    mean_inf_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="mean_inf_sigma"
    )
    mean_theta_layer = SampleNormalDistributionLayer(name="mean_theta")
    mean_theta_norm_layer = NormalizationLayer(
        config.theta_normalization, name="mean_theta_norm"
    )
    alpha_layer = ThetaActivationLayer(
        config.alpha_xform,
        config.initial_alpha_temperature,
        config.learn_alpha_temperature,
        name="alpha",
    )

    # Data flow
    mean_inf_mu = mean_inf_mu_layer(inference_output)
    mean_inf_sigma = mean_inf_sigma_layer(inference_output)
    mean_theta = mean_theta_layer([mean_inf_mu, mean_inf_sigma])
    mean_theta_norm = mean_theta_norm_layer(mean_theta)
    alpha = alpha_layer(mean_theta_norm)

    #
    # Mode time course for the standard deviation
    #

    # Layers
    if config.fix_std:
        beta_layer = FillConstantLayer(1 / config.n_modes, name="beta")
    elif config.tie_mean_std:
        beta_layer = DummyLayer(name="beta")
    else:
        std_inf_mu_layer = layers.Dense(config.n_modes, name="std_inf_mu")
        std_inf_sigma_layer = layers.Dense(
            config.n_modes, activation="softplus", name="std_inf_sigma"
        )
        std_theta_layer = SampleNormalDistributionLayer(name="std_theta")
        std_theta_norm_layer = NormalizationLayer(
            config.theta_normalization, name="std_theta_norm"
        )
        beta_layer = ThetaActivationLayer(
            config.alpha_xform,
            config.initial_alpha_temperature,
            config.learn_alpha_temperature,
            name="beta",
        )

    # Data flow
    if config.fix_std or config.tie_mean_std:
        beta = beta_layer(alpha)
    else:
        std_inf_mu = std_inf_mu_layer(inference_output)
        std_inf_sigma = std_inf_sigma_layer(inference_output)
        std_theta = std_theta_layer([std_inf_mu, std_inf_sigma])
        std_theta_norm = std_theta_norm_layer(std_theta)
        beta = beta_layer(std_theta_norm)

    #
    # Mode time course for the FCs
    #

    # Layers
    fc_inf_mu_layer = layers.Dense(config.n_modes, name="fc_inf_mu")
    fc_inf_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="fc_inf_sigma"
    )
    fc_theta_layer = SampleNormalDistributionLayer(name="fc_theta")
    fc_theta_norm_layer = NormalizationLayer(
        config.theta_normalization, name="fc_theta_norm"
    )
    gamma_layer = ThetaActivationLayer(
        config.alpha_xform,
        config.initial_alpha_temperature,
        config.learn_alpha_temperature,
        name="gamma",
    )

    # Data flow
    fc_inf_mu = fc_inf_mu_layer(inference_output)
    fc_inf_sigma = fc_inf_sigma_layer(inference_output)
    fc_theta = fc_theta_layer([fc_inf_mu, fc_inf_sigma])
    fc_theta_norm = fc_theta_norm_layer(fc_theta)
    gamma = gamma_layer(fc_theta_norm)

    #
    # Observation model
    #

    # Layers
    means_layer = MeanVectorsLayer(
        config.n_modes,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        name="means",
    )
    stds_layer = DiagonalMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_stds,
        config.initial_stds,
        name="stds",
    )
    fcs_layer = CovarianceMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_fcs,
        config.initial_fcs,
        regularize=config.regularize_fcs,
        name="fcs",
    )
    mix_means_stds_fcs_layer = MixMeansStdsFcsLayer(name="mix_means_stds_fcs")
    ll_loss_layer = LogLikelihoodLossLayer(name="ll_loss")

    # Data flow
    mu = means_layer(inputs)  # inputs not used
    E = stds_layer(inputs)  # inputs not used
    D = fcs_layer(inputs)  # inputs not used
    m, C = mix_means_stds_fcs_layer([alpha, beta, gamma, mu, E, D])
    ll_loss = ll_loss_layer([inputs, m, C])

    #
    # Model RNN
    #

    # Layers
    model_input_dropout_layer = layers.Dropout(
        config.model_dropout_rate, name="theta_norm_drop"
    )
    model_output_layer = ModelRNNLayers(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout_rate,
        name="mod_rnn_mean",
    )

    # Data flow
    if config.fix_std or config.tie_mean_std:
        theta_norm = tf.concat([mean_theta_norm, fc_theta_norm], axis=2)
    else:
        theta_norm = tf.concat([mean_theta_norm, std_theta_norm, fc_theta_norm], axis=2)
    model_input_dropout = model_input_dropout_layer(theta_norm)
    model_output = model_output_layer(model_input_dropout)

    #
    # Mode time course for the mean
    #

    # Layers
    mean_mod_mu_layer = layers.Dense(config.n_modes, name="mean_mod_mu")
    mean_mod_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="mean_mod_sigma"
    )
    kl_div_layer_mean = KLDivergenceLayer(name="mean_kl_div")

    # Data flow
    mean_mod_mu = mean_mod_mu_layer(model_output)
    mean_mod_sigma = mean_mod_sigma_layer(model_output)
    mean_kl_div = kl_div_layer_mean(
        [mean_inf_mu, mean_inf_sigma, mean_mod_mu, mean_mod_sigma]
    )

    #
    # Mode time course for the standard deviation
    #

    if not (config.fix_std or config.tie_mean_std):
        # Layers
        std_mod_mu_layer = layers.Dense(config.n_modes, name="std_mod_mu")
        std_mod_sigma_layer = layers.Dense(
            config.n_modes, activation="softplus", name="std_mod_sigma"
        )
        std_kl_div_layer = KLDivergenceLayer(name="std_kl_div")

        # Data flow
        std_mod_mu = std_mod_mu_layer(model_output)
        std_mod_sigma = std_mod_sigma_layer(model_output)
        std_kl_div = std_kl_div_layer(
            [std_inf_mu, std_inf_sigma, std_mod_mu, std_mod_sigma]
        )

    #
    # Mode time course for the functional connectivity
    #

    # Layers
    fc_mod_mu_layer = layers.Dense(config.n_modes, name="fc_mod_mu")
    fc_mod_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="fc_mod_sigma"
    )
    fc_kl_div_layer = KLDivergenceLayer(name="fc_kl_div")

    # Data flow
    fc_mod_mu = fc_mod_mu_layer(model_output)
    fc_mod_sigma = fc_mod_sigma_layer(model_output)
    fc_kl_div = fc_kl_div_layer([fc_inf_mu, fc_inf_sigma, fc_mod_mu, fc_mod_sigma])

    #
    # Total KL loss
    #
    kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")
    if config.fix_std or config.tie_mean_std:
        kl_loss = kl_loss_layer([mean_kl_div, fc_kl_div])
    else:
        kl_loss = kl_loss_layer([mean_kl_div, std_kl_div, fc_kl_div])

    return Model(
        inputs=inputs, outputs=[ll_loss, kl_loss, alpha, beta, gamma], name="MRIGO"
    )
