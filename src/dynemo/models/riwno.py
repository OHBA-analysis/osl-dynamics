"""Model class for a generative model with a WaveNet observation model.

"""

from tensorflow.keras import Model, layers
from dynemo.models.inf_mod_base import InferenceModelBase
from dynemo.models.layers import (
    InferenceRNNLayers,
    LogLikelihoodLayer,
    ModelRNNLayers,
    NormalizationLayer,
    KLDivergenceLayer,
    SampleNormalDistributionLayer,
    CovsLayer,
    MixCovsLayer,
    ThetaActivationLayer,
    WaveNetLayer,
    VectorQuantizerLayer,
)
from dynemo.models.wno import WNO


class RIWNO(InferenceModelBase, WNO):
    """RNN Inference/model network and WaveNet Observations (RIWNO).

    Parameters
    ----------
    config : dynemo.models.Config
    """

    def __init__(self, config):
        InferenceModelBase.__init__(self, config)
        WNO.__init__(self, config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)


def _model_structure(config):

    # Layer for input
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )

    # Inference RNN:
    # - Learns q(theta) ~ N(theta | inf_mu, inf_sigma), where
    #     - inf_mu        ~ affine(RNN(inputs_<=t))
    #     - log_inf_sigma ~ affine(RNN(inputs_<=t))

    # Definition of layers
    inference_input_dropout_layer = layers.Dropout(
        config.inference_dropout_rate, name="data_drop"
    )
    inference_output_layers = InferenceRNNLayers(
        config.inference_rnn,
        config.inference_normalization,
        config.inference_activation,
        config.inference_n_layers,
        config.inference_n_units,
        config.inference_dropout_rate,
        name="inf_rnn",
    )
    inf_mu_layer = layers.Dense(config.n_modes, name="inf_mu")
    inf_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="inf_sigma"
    )

    # Layers to sample theta from q(theta) and to convert to mode mixing
    # factors alpha
    theta_layer = SampleNormalDistributionLayer(name="theta")
    theta_norm_layer = NormalizationLayer(config.theta_normalization, name="theta_norm")
    if config.n_quantized_vectors:
        quant_theta_norm_layer = VectorQuantizerLayer(
            config.n_quantized_vectors,
            config.n_modes,
            config.quantized_vector_beta,
            config.initial_quantized_vectors,
            config.learn_quantized_vectors,
            name="quant_theta_norm",
        )
    alpha_layer = ThetaActivationLayer(
        config.alpha_xform,
        config.initial_alpha_temperature,
        config.learn_alpha_temperature,
        name="alpha",
    )

    # Data flow
    inference_input_dropout = inference_input_dropout_layer(inputs)
    inference_output = inference_output_layers(inference_input_dropout)
    inf_mu = inf_mu_layer(inference_output)
    inf_sigma = inf_sigma_layer(inference_output)
    theta = theta_layer([inf_mu, inf_sigma])
    theta_norm = theta_norm_layer(theta)
    if config.n_quantized_vectors:
        theta_norm = quant_theta_norm_layer(theta_norm)
    alpha = alpha_layer(theta_norm)

    # Observation model:
    # - We use x_t ~ N(mu_t, sigma), where
    #      - mu_t = WaveNet(x_<t, alpha_t).
    #      - sigma = a learnable diagonal covariance matrix.
    # - We calculate the likelihood of generating the training data with
    #   the observation model.

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
    mean = mean_layer([inputs, alpha])
    covs = covs_layer(inputs)  # inputs not used
    cov = mix_covs_layer([alpha, covs])
    ll_loss = ll_loss_layer([inputs, mean, cov])

    # Model RNN:
    # - Learns p(theta|theta_<t) ~ N(theta | mod_mu, mod_sigma), where
    #     - mod_mu        ~ affine(RNN(theta_<t))
    #     - log_mod_sigma ~ affine(RNN(theta_<t))

    # Definition of layers
    model_input_dropout_layer = layers.Dropout(
        config.model_dropout_rate, name="theta_norm_drop"
    )
    model_output_layers = ModelRNNLayers(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout_rate,
        name="mod_rnn",
    )
    mod_mu_layer = layers.Dense(config.n_modes, name="mod_mu")
    mod_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="mod_sigma"
    )
    kl_loss_layer = KLDivergenceLayer(name="kl")

    # Data flow
    model_input_dropout = model_input_dropout_layer(theta_norm)
    model_output = model_output_layers(model_input_dropout)
    mod_mu = mod_mu_layer(model_output)
    mod_sigma = mod_sigma_layer(model_output)
    kl_loss = kl_loss_layer([inf_mu, inf_sigma, mod_mu, mod_sigma])

    return Model(inputs=inputs, outputs=[ll_loss, kl_loss, alpha], name="RIWNO")
