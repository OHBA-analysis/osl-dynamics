"""Model class for a generative model with Gaussian observations using a vector
quantized latent space.

"""

import numpy as np
from tensorflow.keras import Model, layers
from vrad.models.go import GO
from vrad.models.inf_mod_base import InferenceModelBase
from vrad.models.layers import (
    InferenceRNNLayers,
    LogLikelihoodLayer,
    MeansCovsLayer,
    MixMeansCovsLayer,
    ModelRNNLayers,
    NormalizationLayer,
    NormalKLDivergenceLayer,
    SampleNormalDistributionLayer,
    ThetaActivationLayer,
    VectorQuantizerLayer,
)


class RIVQGO(InferenceModelBase, GO):
    """RNN Inference/model network, Vector Quantized latent space
    and Gaussian Observations (RIVQGO).

    Parameters
    ----------
    config : vrad.models.Config
    """

    def __init__(self, config):
        InferenceModelBase.__init__(self, config)
        GO.__init__(self, config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_quantized_alpha(self) -> np.ndarray:
        """Inferred quantized alpha vectors.

        Returns
        -------
        np.ndarray
            Quantized alpha vectors. Shape is (n_vectors, vector_dim).
        """
        vec_quant_layer = self.model.get_layer("quant_theta_norm")
        alpha_layer = self.model.get_layer("alpha")
        quantized_vectors = vec_quant_layer.quantized_vectors
        quantized_alpha = alpha_layer(quantized_vectors).numpy()
        return quantized_alpha


def _model_structure(config):

    # Layer for input
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )

    # Inference RNN:
    # - Learns q(theta) ~ N(theta | inf_mu, inf_sigma), where
    #     - inf_mu    ~ affine(RNN(inputs_<=t))
    #     - inf_sigma ~ softplus(RNN(inputs_<=t))

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
    inf_mu_layer = layers.Dense(config.n_states, name="inf_mu")
    inf_sigma_layer = layers.Dense(
        config.n_states, activation="softplus", name="inf_sigma"
    )

    # Layers to sample theta from q(theta) and to convert to state mixing
    # factors alpha
    theta_layer = SampleNormalDistributionLayer(name="theta")
    theta_norm_layer = NormalizationLayer(config.theta_normalization, name="theta_norm")
    quant_theta_norm_layer = VectorQuantizerLayer(
        config.n_quantized_vectors,
        config.n_states,
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
    quant_theta_norm = quant_theta_norm_layer(theta_norm)
    alpha = alpha_layer(quant_theta_norm)

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
    mu, D = means_covs_layer(inputs)  # inputs not used
    m, C = mix_means_covs_layer([alpha, mu, D])
    ll_loss = ll_loss_layer([inputs, m, C])

    # Model RNN:
    # - Learns p(theta_t |theta_<t) ~ N(theta_t | mod_mu, mod_sigma), where
    #     - mod_mu    ~ affine(RNN(theta_<t))
    #     - mod_sigma ~ softplus(RNN(theta_<t))

    # Definition of layers
    model_input_dropout_layer = layers.Dropout(
        config.model_dropout_rate, name="quant_theta_norm_drop"
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
    mod_mu_layer = layers.Dense(config.n_states, name="mod_mu")
    mod_sigma_layer = layers.Dense(
        config.n_states, activation="softplus", name="mod_sigma"
    )
    kl_loss_layer = NormalKLDivergenceLayer(name="kl")

    # Data flow
    model_input_dropout = model_input_dropout_layer(quant_theta_norm)
    model_output = model_output_layers(model_input_dropout)
    mod_mu = mod_mu_layer(model_output)
    mod_sigma = mod_sigma_layer(model_output)
    kl_loss = kl_loss_layer([inf_mu, inf_sigma, mod_mu, mod_sigma])

    return Model(inputs=inputs, outputs=[ll_loss, kl_loss, alpha], name="RIVQGO")
