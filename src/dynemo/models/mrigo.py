"""Model class for a generative model with Gaussian observations.

"""

import logging
from operator import lt

import numpy as np
from tensorflow.keras import Model, layers
from tqdm import trange
from dynemo.models.go import GO
from dynemo.models.inf_mod_base import InferenceModelBase
from dynemo.models.layers import (
    InferenceRNNLayers,
    LogLikelihoodLayer,
    MeansVarsFcsLayer,
    MixMeansVarsFcsLayer,
    ModelRNNLayers,
    NormalizationLayer,
    NormalKLDivergenceLayer,
    SampleNormalDistributionLayer,
    ThetaActivationLayer,
    KLsum,
)

from dynemo.utils.misc import check_arguments

_logger = logging.getLogger("DyNeMo")


class MRIGO(InferenceModelBase, GO):
    """RNN Inference/model network and Gaussian Observations (RIGO).

    Parameters
    ----------
    config : dynemo.models.Config
    """

    def __init__(self, config):
        InferenceModelBase.__init__(self, config)
        GO.__init__(self, config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)


def _model_structure(config):

    # Layer for input
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )

    ## Definition of the inference layers

    # Mode time course for the mean

    inference_input_dropout_layer_mean = layers.Dropout(
        config.inference_dropout_rate, name="data_drop_mean"
    )
    inference_output_layers_mean = InferenceRNNLayers(
        config.inference_rnn,
        config.inference_normalization,
        config.inference_activation,
        config.inference_n_layers,
        config.inference_n_units,
        config.inference_dropout_rate,
        name="inf_rnn_mean",
    )
    inf_mu_layer_mean = layers.Dense(config.n_modes, name="inf_mu_mean")
    inf_sigma_layer_mean = layers.Dense(
        config.n_modes, activation="softplus", name="inf_sigma_mean"
    )

    # Mode time course for the variances
    inference_input_dropout_layer_var = layers.Dropout(
        config.inference_dropout_rate, name="data_drop_var"
    )
    inference_output_layers_var = InferenceRNNLayers(
        config.inference_rnn,
        config.inference_normalization,
        config.inference_activation,
        config.inference_n_layers,
        config.inference_n_units,
        config.inference_dropout_rate,
        name="inf_rnn_var",
    )
    inf_mu_layer_var = layers.Dense(config.n_modes, name="inf_mu_var")
    inf_sigma_layer_var = layers.Dense(
        config.n_modes, activation="softplus", name="inf_sigma_var"
    )

    # Mode time course for the fcs
    inference_input_dropout_layer_fc = layers.Dropout(
        config.inference_dropout_rate, name="data_drop_fc"
    )
    inference_output_layers_fc = InferenceRNNLayers(
        config.inference_rnn,
        config.inference_normalization,
        config.inference_activation,
        config.inference_n_layers,
        config.inference_n_units,
        config.inference_dropout_rate,
        name="inf_rnn_fc",
    )
    inf_mu_layer_fc = layers.Dense(config.n_modes, name="inf_mu_fc")
    inf_sigma_layer_fc = layers.Dense(
        config.n_modes, activation="softplus", name="inf_sigma_fc"
    )

    # Layers to sample theta from q(theta)
    # and to convert to state mixing factors alpha, beta, gamma
    theta_layer_mean = SampleNormalDistributionLayer(name="theta_mean")
    theta_norm_layer_mean = NormalizationLayer(
        config.theta_normalization, name="theta_norm_mean"
    )
    alpha_layer = ThetaActivationLayer(
        config.alpha_xform,
        config.initial_alpha_temperature,
        config.learn_alpha_temperature,
        name="alpha",
    )
    theta_layer_var = SampleNormalDistributionLayer(name="theta_var")
    theta_norm_layer_var = NormalizationLayer(
        config.theta_normalization, name="theta_norm_var"
    )
    beta_layer = ThetaActivationLayer(
        config.alpha_xform,
        config.initial_alpha_temperature,
        config.learn_alpha_temperature,
        name="beta",
    )
    theta_layer_fc = SampleNormalDistributionLayer(name="theta_fc")
    theta_norm_layer_fc = NormalizationLayer(
        config.theta_normalization, name="theta_norm_fc"
    )
    gamma_layer = ThetaActivationLayer(
        config.alpha_xform,
        config.initial_alpha_temperature,
        config.learn_alpha_temperature,
        name="gamma",
    )

    # Data flow
    inference_input_dropout_mean = inference_input_dropout_layer_mean(inputs)
    inference_output_mean = inference_output_layers_mean(inference_input_dropout_mean)
    inf_mu_mean = inf_mu_layer_mean(inference_output_mean)
    inf_sigma_mean = inf_sigma_layer_mean(inference_output_mean)
    theta_mean = theta_layer_mean([inf_mu_mean, inf_sigma_mean])
    theta_norm_mean = theta_norm_layer_mean(theta_mean)
    alpha = alpha_layer(theta_norm_mean)

    inference_input_dropout_var = inference_input_dropout_layer_var(inputs)
    inference_output_var = inference_output_layers_var(inference_input_dropout_var)
    inf_mu_var = inf_mu_layer_var(inference_output_var)
    inf_sigma_var = inf_sigma_layer_var(inference_output_var)
    theta_var = theta_layer_var([inf_mu_var, inf_sigma_var])
    theta_norm_var = theta_norm_layer_var(theta_var)
    beta = beta_layer(theta_norm_var)

    inference_input_dropout_fc = inference_input_dropout_layer_fc(inputs)
    inference_output_fc = inference_output_layers_fc(inference_input_dropout_fc)
    inf_mu_fc = inf_mu_layer_fc(inference_output_fc)
    inf_sigma_fc = inf_sigma_layer_fc(inference_output_fc)
    theta_fc = theta_layer_fc([inf_mu_fc, inf_sigma_fc])
    theta_norm_fc = theta_norm_layer_fc(theta_fc)
    gamma = gamma_layer(theta_norm_fc)

    # Next the observation model
    means_vars_fcs_layer = MeansVarsFcsLayer(
        config.n_modes,
        config.n_channels,
        learn_means=config.learn_means,
        learn_vars=config.learn_vars,
        learn_fcs=config.learn_fcs,
        initial_means=config.initial_means,
        initial_vars=config.initial_vars,
        initial_fcs=config.initial_fcs,
        name="means_vars_fcs",
    )

    mix_means_vars_fcs_layer = MixMeansVarsFcsLayer(
        config.n_modes,
        config.n_channels,
        config.learn_alpha_scaling,
        config.learn_beta_scaling,
        config.learn_gamma_scaling,
        name="mix_means_vars_fcs",
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    mu, E, D = means_vars_fcs_layer(inputs)
    m, C = mix_means_vars_fcs_layer([alpha, beta, gamma, mu, E, D])
    ll_loss = ll_loss_layer([inputs, m, C])

    # Definition of layers of the model (prior)
    model_input_dropout_layer_mean = layers.Dropout(
        config.model_dropout_rate, name="theta_norm_drop_mean"
    )
    model_input_dropout_layer_var = layers.Dropout(
        config.model_dropout_rate, name="theta_norm_drop_var"
    )
    model_input_dropout_layer_fc = layers.Dropout(
        config.model_dropout_rate, name="theta_norm_drop_fc"
    )

    model_output_layer_mean = ModelRNNLayers(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout_rate,
        name="mod_rnn_mean",
    )
    mod_mu_layer_mean = layers.Dense(config.n_modes, name="mod_mu_mean")
    mod_sigma_layer_mean = layers.Dense(
        config.n_modes, activation="softplus", name="mod_sigma_mean"
    )
    kl_loss_layer_mean = NormalKLDivergenceLayer(name="kl_mean")

    model_output_layer_var = ModelRNNLayers(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout_rate,
        name="mod_rnn_var",
    )
    mod_mu_layer_var = layers.Dense(config.n_modes, name="mod_mu_var")
    mod_sigma_layer_var = layers.Dense(
        config.n_modes, activation="softplus", name="mod_sigma_var"
    )
    kl_loss_layer_var = NormalKLDivergenceLayer(name="kl_var")

    model_output_layer_fc = ModelRNNLayers(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout_rate,
        name="mod_rnn_fc",
    )
    mod_mu_layer_fc = layers.Dense(config.n_modes, name="mod_mu_fc")
    mod_sigma_layer_fc = layers.Dense(
        config.n_modes, activation="softplus", name="mod_sigma_fc"
    )
    kl_loss_layer_fc = NormalKLDivergenceLayer(name="kl_fc")

    kl_sum_layer = KLsum(name="kl")

    # Data flow
    model_input_dropout_mean = model_input_dropout_layer_mean(theta_norm_mean)
    model_output_mean = model_output_layer_mean(model_input_dropout_mean)
    mod_mu_mean = mod_mu_layer_mean(model_output_mean)
    mod_sigma_mean = mod_sigma_layer_mean(model_output_mean)
    kl_loss_mean = kl_loss_layer_mean(
        [inf_mu_mean, inf_sigma_mean, mod_mu_mean, mod_sigma_mean]
    )

    model_input_dropout_var = model_input_dropout_layer_var(theta_norm_var)
    model_output_var = model_output_layer_var(model_input_dropout_var)
    mod_mu_var = mod_mu_layer_var(model_output_var)
    mod_sigma_var = mod_sigma_layer_var(model_output_var)
    kl_loss_var = kl_loss_layer_var(
        [inf_mu_var, inf_sigma_var, mod_mu_var, mod_sigma_var]
    )

    model_input_dropout_fc = model_input_dropout_layer_fc(theta_norm_fc)
    model_output_fc = model_output_layer_fc(model_input_dropout_fc)
    mod_mu_fc = mod_mu_layer_fc(model_output_fc)
    mod_sigma_fc = mod_sigma_layer_fc(model_output_fc)
    kl_loss_fc = kl_loss_layer_fc([inf_mu_fc, inf_sigma_fc, mod_mu_fc, mod_sigma_fc])

    # Total KL loss
    kl_loss = kl_sum_layer([kl_loss_mean, kl_loss_var, kl_loss_fc])

    return Model(
        inputs=inputs, outputs=[ll_loss, kl_loss, alpha, beta, gamma], name="MRIGO"
    )
