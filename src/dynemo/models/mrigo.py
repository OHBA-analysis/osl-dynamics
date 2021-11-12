"""Model class for a multi-time-scale generative model with Gaussian observations.

"""

from tensorflow.keras import Model, layers
from dynemo.models.go import GO
from dynemo.models.inf_mod_base import InferenceModelBase
from dynemo.models.layers import (
    InferenceRNNLayers,
    LogLikelihoodLayer,
    MeansStdsFcsLayer,
    MixMeansStdsFcsLayer,
    ModelRNNLayers,
    NormalizationLayer,
    NormalKLDivergenceLayer,
    SampleNormalDistributionLayer,
    ThetaActivationLayer,
    Sum,
    FillConstant,
)


class MRIGO(InferenceModelBase, GO):
    """Multi-time-scale RNN Inference/model network and Gaussian
    Observations (MRIGO).

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

    #
    # Inference RNN
    #
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

    if not config.fix_std:
        # Mode time course for the standard deviations
        inference_input_dropout_layer_std = layers.Dropout(
            config.inference_dropout_rate, name="data_drop_std"
        )
        inference_output_layers_std = InferenceRNNLayers(
            config.inference_rnn,
            config.inference_normalization,
            config.inference_activation,
            config.inference_n_layers,
            config.inference_n_units,
            config.inference_dropout_rate,
            name="inf_rnn_std",
        )
        inf_mu_layer_std = layers.Dense(config.n_modes, name="inf_mu_std")
        inf_sigma_layer_std = layers.Dense(
            config.n_modes, activation="softplus", name="inf_sigma_std"
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

    if not config.fix_std:
        theta_layer_std = SampleNormalDistributionLayer(name="theta_std")
        theta_norm_layer_std = NormalizationLayer(
            config.theta_normalization, name="theta_norm_std"
        )
        beta_layer = ThetaActivationLayer(
            config.alpha_xform,
            config.initial_alpha_temperature,
            config.learn_alpha_temperature,
            name="beta",
        )
    else:
        beta_layer = FillConstant(1 / config.n_modes, name="beta")

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

    if not config.fix_std:
        inference_input_dropout_std = inference_input_dropout_layer_std(inputs)
        inference_output_std = inference_output_layers_std(inference_input_dropout_std)
        inf_mu_std = inf_mu_layer_std(inference_output_std)
        inf_sigma_std = inf_sigma_layer_std(inference_output_std)
        theta_std = theta_layer_std([inf_mu_std, inf_sigma_std])
        theta_norm_std = theta_norm_layer_std(theta_std)
        beta = beta_layer(theta_norm_std)
    else:
        beta = beta_layer(alpha)

    inference_input_dropout_fc = inference_input_dropout_layer_fc(inputs)
    inference_output_fc = inference_output_layers_fc(inference_input_dropout_fc)
    inf_mu_fc = inf_mu_layer_fc(inference_output_fc)
    inf_sigma_fc = inf_sigma_layer_fc(inference_output_fc)
    theta_fc = theta_layer_fc([inf_mu_fc, inf_sigma_fc])
    theta_norm_fc = theta_norm_layer_fc(theta_fc)
    gamma = gamma_layer(theta_norm_fc)

    #
    # Observation model
    #
    means_stds_fcs_layer = MeansStdsFcsLayer(
        config.n_modes,
        config.n_channels,
        learn_means=config.learn_means,
        learn_stds=config.learn_stds,
        learn_fcs=config.learn_fcs,
        initial_means=config.initial_means,
        initial_stds=config.initial_stds,
        initial_fcs=config.initial_fcs,
        name="means_stds_fcs",
    )

    mix_means_stds_fcs_layer = MixMeansStdsFcsLayer(
        config.n_modes,
        config.n_channels,
        config.learn_alpha_scaling,
        config.learn_beta_scaling,
        config.learn_gamma_scaling,
        config.fix_std,
        name="mix_means_stds_fcs",
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    mu, E, D = means_stds_fcs_layer(inputs)
    m, C = mix_means_stds_fcs_layer([alpha, beta, gamma, mu, E, D])
    ll_loss = ll_loss_layer([inputs, m, C])

    #
    # Model RNN
    #
    model_input_dropout_layer_mean = layers.Dropout(
        config.model_dropout_rate, name="theta_norm_drop_mean"
    )
    model_input_dropout_layer_std = layers.Dropout(
        config.model_dropout_rate, name="theta_norm_drop_std"
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

    model_output_layer_std = ModelRNNLayers(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout_rate,
        name="mod_rnn_std",
    )
    mod_mu_layer_std = layers.Dense(config.n_modes, name="mod_mu_std")
    mod_sigma_layer_std = layers.Dense(
        config.n_modes, activation="softplus", name="mod_sigma_std"
    )
    kl_loss_layer_std = NormalKLDivergenceLayer(name="kl_std")

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

    kl_sum_layer = Sum(name="kl")

    # Data flow
    model_input_dropout_mean = model_input_dropout_layer_mean(theta_norm_mean)
    model_output_mean = model_output_layer_mean(model_input_dropout_mean)
    mod_mu_mean = mod_mu_layer_mean(model_output_mean)
    mod_sigma_mean = mod_sigma_layer_mean(model_output_mean)
    kl_loss_mean = kl_loss_layer_mean(
        [inf_mu_mean, inf_sigma_mean, mod_mu_mean, mod_sigma_mean]
    )

    if not config.fix_std:
        model_input_dropout_std = model_input_dropout_layer_std(theta_norm_std)
        model_output_std = model_output_layer_std(model_input_dropout_std)
        mod_mu_std = mod_mu_layer_std(model_output_std)
        mod_sigma_std = mod_sigma_layer_std(model_output_std)
        kl_loss_std = kl_loss_layer_std(
            [inf_mu_std, inf_sigma_std, mod_mu_std, mod_sigma_std]
        )

    model_input_dropout_fc = model_input_dropout_layer_fc(theta_norm_fc)
    model_output_fc = model_output_layer_fc(model_input_dropout_fc)
    mod_mu_fc = mod_mu_layer_fc(model_output_fc)
    mod_sigma_fc = mod_sigma_layer_fc(model_output_fc)
    kl_loss_fc = kl_loss_layer_fc([inf_mu_fc, inf_sigma_fc, mod_mu_fc, mod_sigma_fc])

    # Total KL loss
    if not config.fix_std:
        kl_loss = kl_sum_layer([kl_loss_mean, kl_loss_std, kl_loss_fc])
    else:
        kl_loss = kl_sum_layer([kl_loss_mean, kl_loss_fc])

    return Model(
        inputs=inputs, outputs=[ll_loss, kl_loss, alpha, beta, gamma], name="MRIGO"
    )
