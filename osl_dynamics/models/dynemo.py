"""Dynamic Network Modes (DyNeMo).

"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import trange

from osl_dynamics.models import dynemo_obs
from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.models.inf_mod_base import (
    VariationalInferenceModelConfig,
    VariationalInferenceModelBase,
)
from osl_dynamics.inference.layers import (
    InferenceRNNLayer,
    LogLikelihoodLossLayer,
    MeanVectorsLayer,
    CovarianceMatricesLayer,
    MixVectorsLayer,
    MixMatricesLayer,
    ModelRNNLayer,
    NormalizationLayer,
    KLDivergenceLayer,
    KLLossLayer,
    SampleNormalDistributionLayer,
    SoftmaxLayer,
)


@dataclass
class Config(BaseModelConfig, VariationalInferenceModelConfig):
    """Settings for DyNeMo.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.

    inference_rnn : str
        RNN to use, either 'gru' or 'lstm'.
    inference_n_layers : int
        Number of layers.
    inference_n_units : int
        Number of units.
    inference_normalization : str
        Type of normalization to use. Either None, 'batch' or 'layer'.
    inference_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. 'relu', 'elu', etc.
    inference_dropout : float
        Dropout rate.

    model_rnn : str
        RNN to use, either 'gru' or 'lstm'.
    model_n_layers : int
        Number of layers.
    model_n_units : int
        Number of units.
    model_normalization : str
        Type of normalization to use. Either None, 'batch' or 'layer'.
    model_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. 'relu', 'elu', etc.
    model_dropout : float
        Dropout rate.

    theta_normalization : str
        Type of normalization to apply to the posterior samples, theta.
        Either 'layer', 'batch' or None.
    learn_alpha_temperature : bool
        Should we learn the alpha temperature?
    initial_alpha_temperature : float
        Initial value for the alpha temperature.

    learn_means : bool
        Should we make the mean vectors for each mode trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each mode trainable?
    initial_means : np.ndarray
        Initialisation for mean vectors.
    initial_covariances : np.ndarray
        Initialisation for mode covariances.

    do_kl_annealing : bool
        Should we use KL annealing during training?
    kl_annealing_curve : str
        Type of KL annealing curve. Either 'linear' or 'tanh'.
    kl_annealing_sharpness : float
        Parameter to control the shape of the annealing curve if
        kl_annealing_curve='tanh'.
    n_kl_annealing_epochs : int
        Number of epochs to perform KL annealing.

    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
    gradient_clip : float
        Value to clip gradients by. This is the clipnorm argument passed to
        the Keras optimizer. Cannot be used if multi_gpu=True.
    n_epochs : int
        Number of training epochs.
    optimizer : str or tensorflow.keras.optimizers.Optimizer
        Optimizer to use. 'adam' is recommended.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    """

    # Inference network parameters
    inference_rnn: Literal["gru", "lstm"] = "lstm"
    inference_n_layers: int = 1
    inference_n_units: int = None
    inference_normalization: Literal[None, "batch", "layer"] = None
    inference_activation: str = None
    inference_dropout: float = 0.0

    # Model network parameters
    model_rnn: Literal["gru", "lstm"] = "lstm"
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: Literal[None, "batch", "layer"] = None
    model_activation: str = None
    model_dropout: float = 0.0

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None

    def __post_init__(self):
        self.validate_rnn_parameters()
        self.validate_observation_model_parameters()
        self.validate_alpha_parameters()
        self.validate_kl_annealing_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_rnn_parameters(self):
        if self.inference_n_units is None:
            raise ValueError("Please pass inference_n_units.")

        if self.model_n_units is None:
            raise ValueError("Please pass model_n_units.")

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")


class Model(VariationalInferenceModelBase):
    """DyNeMo model class.

    Parameters
    ----------
    config : osl_dynamics.models.dynemo.Config
    """

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_covariances(self):
        """Get the covariances of each mode.

        Returns
        -------
        covariances : np.ndarary
            Mode covariances.
        """
        return dynemo_obs.get_covariances(self.model)

    def get_means_covariances(self):
        """Get the means and covariances of each mode.

        Returns
        -------
        means : np.ndarary
            Mode means.
        covariances : np.ndarray
            Mode covariances.
        """
        return dynemo_obs.get_means_covariances(self.model)

    def set_means(self, means, update_initializer=True):
        """Set the means of each mode.

        Parameters
        ----------
        means : np.ndarray
            Mode covariances.
        update_initializer : bool
            Do we want to use the passed means when we re-initialize
            the model?
        """
        dynemo_obs.set_means(self.model, means, update_initializer)

    def set_covariances(self, covariances, update_initializer=True):
        """Set the covariances of each mode.

        Parameters
        ----------
        covariances : np.ndarray
            Mode covariances.
        update_initializer : bool
            Do we want to use the passed covariances when we re-initialize
            the model?
        """
        dynemo_obs.set_covariances(self.model, covariances, update_initializer)

    def sample_alpha(self, n_samples, theta_norm=None):
        """Uses the model RNN to sample mode mixing factors, alpha.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.
        theta_norm : np.ndarray
            Normalized logits to initialise the sampling with. Shape must be
            (sequence_length, n_modes).

        Returns
        -------
        alpha : np.ndarray
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
            theta = mod_mu + mod_sigma * epsilon[i]
            theta_norm[-1] = theta_norm_layer(theta[np.newaxis, np.newaxis, :])[0]

            # Calculate the mode mixing factors
            alpha[i] = alpha_layer(mod_mu[np.newaxis, np.newaxis, :])[0, 0]

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
    data_drop_layer = layers.Dropout(config.inference_dropout, name="data_drop")
    inf_rnn_layer = InferenceRNNLayer(
        config.inference_rnn,
        config.inference_normalization,
        config.inference_activation,
        config.inference_n_layers,
        config.inference_n_units,
        config.inference_dropout,
        name="inf_rnn",
    )
    inf_mu_layer = layers.Dense(config.n_modes, name="inf_mu")
    inf_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="inf_sigma"
    )
    theta_layer = SampleNormalDistributionLayer(name="theta")
    theta_norm_layer = NormalizationLayer(config.theta_normalization, name="theta_norm")
    alpha_layer = SoftmaxLayer(
        config.initial_alpha_temperature,
        config.learn_alpha_temperature,
        name="alpha",
    )

    # Data flow
    data_drop = data_drop_layer(inputs)
    inf_rnn = inf_rnn_layer(data_drop)
    inf_mu = inf_mu_layer(inf_rnn)
    inf_sigma = inf_sigma_layer(inf_rnn)
    theta = theta_layer([inf_mu, inf_sigma])
    theta_norm = theta_norm_layer(theta)
    alpha = alpha_layer(theta_norm)

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each mode as the observation model.
    # - We calculate the likelihood of generating the training data with alpha
    #   and the observation model.

    # Definition of layers
    means_layer = MeanVectorsLayer(
        config.n_modes,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        name="means",
    )
    covs_layer = CovarianceMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_covariances,
        config.initial_covariances,
        name="covs",
    )
    mix_means_layer = MixVectorsLayer(name="mix_means")
    mix_covs_layer = MixMatricesLayer(name="mix_covs")
    ll_loss_layer = LogLikelihoodLossLayer(name="ll_loss")

    # Data flow
    mu = means_layer(inputs)  # inputs not used
    D = covs_layer(inputs)  # inputs not used
    m = mix_means_layer([alpha, mu])
    C = mix_covs_layer([alpha, D])
    ll_loss = ll_loss_layer([inputs, m, C])

    # Model RNN:
    # - Learns p(theta_t |theta_<t) ~ N(theta_t | mod_mu, mod_sigma), where
    #     - mod_mu    ~ affine(RNN(theta_<t))
    #     - mod_sigma ~ softplus(RNN(theta_<t))

    # Definition of layers
    theta_norm_drop_layer = layers.Dropout(config.model_dropout, name="theta_norm_drop")
    mod_rnn_layer = ModelRNNLayer(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout,
        name="mod_rnn",
    )
    mod_mu_layer = layers.Dense(config.n_modes, name="mod_mu")
    mod_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="mod_sigma"
    )
    kl_div_layer = KLDivergenceLayer(name="kl_div")
    kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")

    # Data flow
    theta_norm_drop = theta_norm_drop_layer(theta_norm)
    mod_rnn = mod_rnn_layer(theta_norm_drop)
    mod_mu = mod_mu_layer(mod_rnn)
    mod_sigma = mod_sigma_layer(mod_rnn)
    kl_div = kl_div_layer([inf_mu, inf_sigma, mod_mu, mod_sigma])
    kl_loss = kl_loss_layer(kl_div)

    return tf.keras.Model(
        inputs=inputs, outputs=[ll_loss, kl_loss, alpha], name="DyNeMo"
    )
