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
    MeansCovsLayer,
    MixMeansCovsLayer,
    ModelRNNLayers,
    NormalizationLayer,
    NormalKLDivergenceLayer,
    SampleNormalDistributionLayer,
    ThetaActivationLayer,
    VectorQuantizerLayer,
)
from dynemo.utils.misc import check_arguments

_logger = logging.getLogger("DyNeMo")


class RIGO(InferenceModelBase, GO):
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

    def burn_in(
        self,
        *args,
        learn_means_covariances: bool = False,
        learn_alpha_temperature: bool = False,
        **kwargs,
    ):
        """Burn-in training phase.

        Fits the model with means and covariances or alpha_temperature
        non-trainable.

        Parameters
        ----------
        learn_means_covariances : bool
            Should we learn the means and covariances during the burn-in training?
            Optional, default is False.
        learn_alpha_temperature : bool
            Should we learn the alpha temperature during the burn-in taining?
            Optional, default is False.
        """
        if check_arguments(args, kwargs, 3, "epochs", 1, lt):
            _logger.warning(
                "Number of burn-in epochs is less than 1. Skipping burn-in."
            )
            return

        # Make means and covariances non-trainable and compile
        if not learn_means_covariances:
            means_covs_layer = self.model.get_layer("means_covs")
            means_covs_layer.trainable = False
            self.compile()

        # Make alpha temperature non-trainable and compile
        if not learn_alpha_temperature:
            alpha_layer = self.model.get_layer("alpha_layer")
            alpha_layer.trainable = False
            self.compile()

        # Train the model
        self.fit(*args, **kwargs)

        # Make means and covariances trainable again and compile
        if not learn_means_covariances:
            means_covs_layer.trainable = True
            self.compile()

        # Make alpha temperature trainable again and compile
        if not learn_alpha_temperature:
            alpha_layer.trainable = True
            self.compile()

    def sample_alpha(
        self, n_samples: int, theta_norm: np.ndarray = None, rescale_sigma: float = 1.0
    ) -> np.ndarray:
        """Uses the model RNN to sample mode mixing factors, alpha.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.
        theta_norm : np.ndarray
            Normalized logits to initialise the sampling with. Shape must be
            (sequence_length, n_modes). Optional.
        rescale_sigma : float
            Factor to rescale the standard deviation of theta by. Optional.

        Returns
        -------
        np.ndarray
            Sampled alpha.
        """
        # Get layers
        model_rnn_layer = self.model.get_layer("mod_rnn")
        mod_mu_layer = self.model.get_layer("mod_mu")
        mod_sigma_layer = self.model.get_layer("mod_sigma")
        theta_norm_layer = self.model.get_layer("theta_norm")
        alpha_layer = self.model.get_layer("alpha")

        # Normally distributed random numbers used to sample the logits theta
        epsilon = np.random.normal(0, 1, [n_samples + 1, self.config.n_modes]).astype(
            np.float32
        )

        if theta_norm is None:
            # Sequence of the underlying logits theta
            theta_norm = np.zeros(
                [self.config.sequence_length, self.config.n_modes],
                dtype=np.float32,
            )

            # Randomly sample the first time step
            theta_norm[-1] = np.random.normal(size=self.config.n_modes)

        # Sample the mode fixing factors
        alpha = np.empty([n_samples, self.config.n_modes], dtype=np.float32)
        for i in trange(n_samples, desc="Sampling mode time course", ncols=98):

            # If there are leading zeros we trim theta so that we don't pass the zeros
            trimmed_theta = theta_norm[~np.all(theta_norm == 0, axis=1)][
                np.newaxis, :, :
            ]

            # Predict the probability distribution function for theta one time step
            # in the future,
            # p(theta|theta_<t) ~ N(mod_mu, sigma_theta_jt)
            model_rnn = model_rnn_layer(trimmed_theta)
            mod_mu = mod_mu_layer(model_rnn)[0, -1]
            mod_sigma = mod_sigma_layer(model_rnn)[0, -1]

            # Shift theta one time step to the left
            theta_norm = np.roll(theta_norm, -1, axis=0)

            # Sample from the probability distribution function
            theta = mod_mu + mod_sigma * epsilon[i] * rescale_sigma
            theta_norm[-1] = theta_norm_layer(theta[np.newaxis, np.newaxis, :])[0]

            # Calculate the mode mixing factors
            alpha[i] = alpha_layer(theta_norm[-1][np.newaxis, np.newaxis, :])[0, 0]

        return alpha


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
    mu, D = means_covs_layer(inputs)  # inputs not used
    m, C = mix_means_covs_layer([alpha, mu, D])
    ll_loss = ll_loss_layer([inputs, m, C])

    # Model RNN:
    # - Learns p(theta_t |theta_<t) ~ N(theta_t | mod_mu, mod_sigma), where
    #     - mod_mu    ~ affine(RNN(theta_<t))
    #     - mod_sigma ~ softplus(RNN(theta_<t))

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
    kl_loss_layer = NormalKLDivergenceLayer(name="kl")

    # Data flow
    model_input_dropout = model_input_dropout_layer(theta_norm)
    model_output = model_output_layers(model_input_dropout)
    mod_mu = mod_mu_layer(model_output)
    mod_sigma = mod_sigma_layer(model_output)
    kl_loss = kl_loss_layer([inf_mu, inf_sigma, mod_mu, mod_sigma])

    return Model(inputs=inputs, outputs=[ll_loss, kl_loss, alpha], name="RIGO")
