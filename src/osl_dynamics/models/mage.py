"""Multi-dynamic Adversarial Generator Encoder (MAGE) model.

"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from osl_dynamics.models import mdynemo_obs
from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.models.inf_mod_base import MultiAdversarialInferenceModelBase
from osl_dynamics.inference.layers import (
    InferenceRNNLayer,
    MeanVectorsLayer,
    DiagonalMatricesLayer,
    CorrelationMatricesLayer,
    MixVectorsLayer,
    MixMatricesLayer,
    MatMulLayer,
    ModelRNNLayer,
    MixVectorsMatricesLayer,
)


@dataclass
class Config(BaseModelConfig):
    """Settings for MAGE.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference, generative and descriminator network.

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

    descriminator_rnn : str
        RNN to use, either 'gru' or 'lstm'.
    descriminator_n_layers : int
        Number of layers.
    descriminator_n_units : int
        Number of units.
    descriminator_normalization : str
        Type of normalization to use. Either None, 'batch' or 'layer'.
    descriminator_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. 'relu', 'elu', etc.
    descriminator_dropout : float
        Dropout rate.

    learn_means : bool
        Should we make the mean vectors for each mode trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each mode trainable?
    initial_means : np.ndarray
        Initialisation for mean vectors.
    initial_covariances : np.ndarray
        Initialisation for mode covariances.

    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
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
    inference_activation: str = "elu"
    inference_dropout: float = 0.0

    # Model network parameters
    model_rnn: Literal["gru", "lstm"] = "lstm"
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: Literal[None, "batch", "layer"] = None
    model_activation: str = "elu"
    model_dropout: float = 0.0

    # Descriminator network parameters
    descriminator_rnn: Literal["gru", "lstm"] = "lstm"
    descriminator_n_layers: int = 1
    descriminator_n_units: int = None
    descriminator_normalization: Literal[None, "batch", "layer"] = None
    descriminator_activation: str = "elu"
    descriminator_dropout: float = 0.0

    # Observation model parameters
    learn_means: bool = None
    learn_stds: bool = None
    learn_fcs: bool = None
    initial_means: np.ndarray = None
    initial_stds: np.ndarray = None
    initial_fcs: np.ndarray = None
    multiple_dynamics: bool = True

    def __post_init__(self):
        self.validate_dimension_parameters()
        self.validate_training_parameters()
        self.validate_rnn_parameters()
        self.validate_observation_model_parameters()
    
    def validate_rnn_parameters(self):
        if self.inference_n_units is None:
            raise ValueError("Please pass inference_n_units.")

        if self.model_n_units is None:
            raise ValueError("Please pass model_n_units.")

        if self.descriminator_n_units is None:
            raise ValueError("Please pass descriminator_n_units.")

    def validate_observation_model_parameters(self):
        if (
            self.learn_means is None
            or self.learn_stds is None
            or self.learn_fcs is None
        ):
            raise ValueError("learn_means, learn_stds and learn_fcs must be passed.")


class Model(MultiAdversarialInferenceModelBase):
    """MAGE model class.

    Parameters
    ----------
    config : osl_dynamics.models.sage.Config
    """

    def build_model(self):
        """Builds a keras model for the inference, generator and discriminator model
        and the full MAGE model.
        """
        print("Build models")
        self.inference_model = _build_inference_model(self.config)
        self.inference_model.summary()
        print()
  
        self.generator_model_mean = _build_generator_model_mean(self.config)
        self.generator_model_mean.summary()
        print()
        self.generator_model_cov = _build_generator_model_cov(self.config)
        self.generator_model_cov.summary()

        self.discriminator_model_mean = _build_discriminator_model_mean(self.config)
        self.discriminator_model_mean.summary()
        print()
        self.discriminator_model_cov = _build_discriminator_model_cov(self.config)
        self.discriminator_model_cov.summary()

        data = layers.Input(
            shape=(self.config.sequence_length, self.config.n_channels), name="data"
        )

        # Connecting the inputs and outputs
        C_m, alpha_posterior, gamma_posterior = self.inference_model(data)
        alpha_prior = self.generator_model_mean(alpha_posterior)
        gamma_prior = self.generator_model_cov(gamma_posterior)

        discriminator_output_alpha = self.discriminator_model_mean(alpha_prior)
        discriminator_output_gamma = self.discriminator_model_cov(gamma_prior)
        self.model = models.Model(data, [C_m, discriminator_output_alpha, discriminator_output_gamma],
             name="MAGE")
        self.model.summary()
        print()

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
            return mdynemo_obs.get_means_stds_fcs(self.inference_model)

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
                the model?
            """
            mdynemo_obs.set_means_stds_fcs(self.inference_model, means, stds, fcs, update_initializer)


def _build_inference_model(config):
    # Inference RNN:
    #   \alpha_{t} = \zeta({\theta^{m}_{t}}) where
    #   \mu^{m,\theta}_t  = f(LSTM_{bi}(Y,\omega^m_e),\lambda_e^m)
    #   \mu^{c,\theta}_t = f(LSTM_{bi}(Y,\omega^c_e),\lambda_e^c)

    # Definition of layers
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )
    data_drop_layer = layers.TimeDistributed(
        layers.Dropout(config.inference_dropout, name="data_drop_inf")
    )
    inf_rnn_layer = InferenceRNNLayer(
        config.inference_rnn,
        config.inference_normalization,
        config.inference_activation,
        config.inference_n_layers,
        config.inference_n_units,
        config.inference_dropout,
        name="inf_rnn",
    )

    alpha_layer = layers.TimeDistributed(
        layers.Dense(config.n_modes, activation="softmax"), name="alpha_inf"
    )

    gamma_layer = layers.TimeDistributed(
        layers.Dense(config.n_modes, activation="softmax"), name="gamma_inf"
    )

    # Data flow
    data_drop = data_drop_layer(inputs)
    theta = inf_rnn_layer(data_drop)
    alpha = alpha_layer(theta)
    gamma = gamma_layer(theta)

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

    stds_layer = DiagonalMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_stds,
        config.initial_stds,
        name="stds",
    )

    fcs_layer = CorrelationMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_fcs,
        config.initial_fcs,
        name="fcs",
    )

    mix_means_layer = MixVectorsLayer(name="mix_means")
    mix_stds_layer = MixMatricesLayer(name="mix_stds")
    mix_fcs_layer = MixMatricesLayer(name="mix_fcs")
    matmul_layer = MatMulLayer(name="cov")
    mix_means_covs_layer = MixVectorsMatricesLayer(name="mix_means_covs")
    
    # Data flow
    mu = means_layer(inputs)  # inputs not used
    E = stds_layer(inputs)  # inputs not used
    D = fcs_layer(inputs)  # inputs not used

    m = mix_means_layer([alpha, mu])
    G = mix_stds_layer([alpha, E])
    F = mix_fcs_layer([gamma, D])
    C = matmul_layer([G, F, G])
    C_m = mix_means_covs_layer([m, C])

    return models.Model(inputs, [C_m, alpha, gamma], name="inference")


def _build_generator_model_mean(config):
    # Model RNN:
    #   \alpha_{t} = \zeta({\theta^{m}_{t}}) where
    #   \hat{\mu}^{m,\theta}_{t} = f (LSTM_{uni} (\theta^{m}_{<t},\omega^m_g), \lambda_g^m)
    #   \hat{\mu}^{c,\theta}_{t}   = f (LSTM_{uni} (\theta^{c}_{<t},\omega^c_g), \lambda_g^c)

    # Input
    generator_input_mean = layers.Input(
        shape=(config.sequence_length, config.n_modes),
        name="generator_input_mean",
    )

    # Definition of layers
    drop_layer = layers.TimeDistributed(
        layers.Dropout(config.model_dropout, name="data_drop_gen_mean")
    )
    mod_rnn_layer = ModelRNNLayer(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout,
        name="mod_rnn_mean",
    )
    alpha_layer = layers.TimeDistributed(
        layers.Dense(config.n_modes, activation="softmax"), name="alpha_gen_mean"
    )

    # Data flow
    theta_drop = drop_layer(generator_input_mean)
    theta_drop_prior = mod_rnn_layer(theta_drop)
    alpha_prior = alpha_layer(theta_drop_prior)

    return models.Model(generator_input_mean, alpha_prior, name="generator_mean")

def _build_generator_model_cov(config):

    # Model RNN:
    #   \alpha_{t} = \zeta({\theta^{m}_{t}}) where
    #   \hat{\mu}^{m,\theta}_{t} = f (LSTM_{uni} (\theta^{m}_{<t},\omega^m_g), \lambda_g^m)
    #   \hat{\mu}^{c,\theta}_{t}   = f (LSTM_{uni} (\theta^{c}_{<t},\omega^c_g), \lambda_g^c)

    # Input
    generator_input_cov = layers.Input(
        shape=(config.sequence_length, config.n_modes),
        name="generator_input_cov",
    )

    # Definition of layers
    drop_layer = layers.TimeDistributed(
        layers.Dropout(config.model_dropout, name="data_drop_gen_cov")
    )
    mod_rnn_layer = ModelRNNLayer(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout,
        name="mod_rnn_cov",
    )
    gamma_layer = layers.TimeDistributed(
        layers.Dense(config.n_modes, activation="softmax"), 
        name="alpha_gen_cov"
    )

    # Data flow
    theta_drop = drop_layer(generator_input_cov)
    theta_drop_prior = mod_rnn_layer(theta_drop)
    gamma_prior = gamma_layer(theta_drop_prior)

    return models.Model(generator_input_cov, gamma_prior, name="generator_cov")


def _build_discriminator_model_mean(config):
    # Descriminator RNN:
    #   D_{\theta^m_t} = \sigma (f (LSTM_{bi}([\zeta(\hat{\mu}^{m,\theta}_{t}), \zeta(\mu^{m,\theta}_t)],\omega^m_d), \lambda_d^m))
    #   D_{\theta^c_t} = \sigma (f (LSTM_{bi}([\zeta(\hat{\mu}^{c,\theta}_{t}), \zeta(\mu^{c,\theta}_t)],\omega^c_d), \lambda_d^c))

    # Definition of layers
    discriminator_input_mean = layers.Input(
        shape=(config.sequence_length, config.n_modes), name="data_mean"
    )

    drop_layer = layers.TimeDistributed(
        layers.Dropout(config.model_dropout, name="data_drop_des_mean")
    )
    des_rnn_layer = ModelRNNLayer(
        config.descriminator_rnn,
        config.descriminator_normalization,
        config.descriminator_activation,
        config.descriminator_n_layers,
        config.descriminator_n_units,
        config.descriminator_dropout,
        name="des_rnn_mean",
    )
    sigmoid_layer_mean = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"), 
        name="sigmoid_mean")

    # Data flow
    theta_norm_drop = drop_layer(discriminator_input_mean)
    discriminator_sequence = des_rnn_layer(theta_norm_drop)
    discriminator_output_mean = sigmoid_layer_mean(discriminator_sequence)

    return models.Model(discriminator_input_mean,
        discriminator_output_mean, name="discriminator_mean")

def _build_discriminator_model_cov(config):
    # Descriminator RNN:
    #   D_{\theta^m_t} = \sigma (f (LSTM_{bi}([\zeta(\hat{\mu}^{m,\theta}_{t}), \zeta(\mu^{m,\theta}_t)],\omega^m_d), \lambda_d^m))
    #   D_{\theta^c_t} = \sigma (f (LSTM_{bi}([\zeta(\hat{\mu}^{c,\theta}_{t}), \zeta(\mu^{c,\theta}_t)],\omega^c_d), \lambda_d^c))

    # Definition of layers
    discriminator_input_cov = layers.Input(
        shape=(config.sequence_length, config.n_modes), name="data_cov"
    )

    drop_layer = layers.TimeDistributed(
        layers.Dropout(config.model_dropout, name="data_drop_des_cov")
    )
    des_rnn_layer = ModelRNNLayer(
        config.descriminator_rnn,
        config.descriminator_normalization,
        config.descriminator_activation,
        config.descriminator_n_layers,
        config.descriminator_n_units,
        config.descriminator_dropout,
        name="des_rnn_cov",
    )
    sigmoid_layer_cov = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"), 
        name="sigmoid_cov")

    # Data flow
    theta_norm_drop = drop_layer(discriminator_input_cov)
    discriminator_sequence = des_rnn_layer(theta_norm_drop)
    discriminator_output_cov = sigmoid_layer_cov(discriminator_sequence)

    return models.Model(discriminator_input_cov,
        discriminator_output_cov, name="discriminator_cov")
