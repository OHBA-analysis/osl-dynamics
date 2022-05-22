"""Single-dynamic Adversarial Generator Encoder (SAGE) model.

"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from osl_dynamics.models import dynemo_obs
from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.models.inf_mod_base import AdversarialInferenceModelBase
from osl_dynamics.inference.layers import (
    InferenceRNNLayer,
    MeanVectorsLayer,
    CovarianceMatricesLayer,
    MixVectorsLayer,
    MixMatricesLayer,
    ModelRNNLayer,
    NormalizationLayer,
    MixVectorsMatricesLayer,
)


@dataclass
class Config(BaseModelConfig):
    """Settings for SAGE.

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
    des_rnn: Literal["gru", "lstm"] = "lstm"
    des_n_layers: int = 1
    des_n_units: int = None
    des_normalization: Literal[None, "batch", "layer"] = None
    des_activation: str = "elu"
    des_dropout: float = 0.0

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None

    def __post_init__(self):
        self.validate_dimension_parameters()
        self.validate_training_parameters()


class Model(AdversarialInferenceModelBase):
    """SAGE model class.

    Parameters
    ----------
    config : osl_dynamics.models.sage.Config
    """

    def build_model(self):
        """Builds a keras model for the inference, generator and discriminator model
        and the full SAGE model.
        """
        print("Build models")
        self.inference_model = _build_inference_model(self.config)
        self.inference_model.summary()
        print()
        self.generator_model = _build_generator_model(self.config)
        self.generator_model.summary()
        print()
        self.discriminator_model = _build_discriminator_model(self.config)
        self.discriminator_model.summary()
        print()

        data = layers.Input(
            shape=(self.config.sequence_length, self.config.n_channels), name="data"
        )
        C_m, alpha_posterior = self.inference_model(data)
        alpha_prior = self.generator_model(alpha_posterior)
        discriminator_output_prior = self.discriminator_model(alpha_prior)
        self.model = models.Model(data, [C_m, discriminator_output_prior], name="SAGE")
        self.model.summary()
        print()

    def get_covariances(self):
        """Get the covariances of each mode.

        Returns
        -------
        np.ndarary
            Mode covariances.
        """
        return dynemo_obs.get_covariances(self.inference_model)

    def get_means_covariances(self):
        """Get the means and covariances of each mode.

        Returns
        -------
        means : np.ndarary
            Mode means.
        covariances : np.ndarray
            Mode covariances.
        """
        return dynemo_obs.get_means_covariances(self.inference_model)


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

    # Data flow
    data_drop = data_drop_layer(inputs)
    theta = inf_rnn_layer(data_drop)
    alpha = alpha_layer(theta)

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
    mix_means_covs_layer = MixVectorsMatricesLayer(name="mix_means_covs")

    # Data flow
    mu = means_layer(inputs)  # inputs not used
    D = covs_layer(inputs)  # inputs not used
    m = mix_means_layer([alpha, mu])
    C = mix_covs_layer([alpha, D])
    C_m = mix_means_covs_layer([m, C])

    return models.Model(inputs, [C_m, alpha], name="inference")


def _build_generator_model(config):
    # Model RNN:
    #   \alpha_{t} = \zeta({\theta^{m}_{t}}) where
    #   \hat{\mu}^{m,\theta}_{t} = f (LSTM_{uni} (\theta^{m}_{<t},\omega^m_g), \lambda_g^m)
    #   \hat{\mu}^{c,\theta}_{t}   = f (LSTM_{uni} (\theta^{c}_{<t},\omega^c_g), \lambda_g^c)

    # Definition of layers
    generator_input = layers.Input(
        shape=(config.sequence_length, config.n_modes),
        name="generator_input",
    )
    drop_layer = layers.TimeDistributed(
        layers.Dropout(config.model_dropout, name="data_drop_gen")
    )
    mod_rnn_layer = ModelRNNLayer(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout,
        name="mod_rnn",
    )
    alpha_layer = layers.TimeDistributed(
        layers.Dense(config.n_modes, activation="softmax"), name="alpha_gen"
    )

    # Data flow
    theta_drop = drop_layer(generator_input)
    theta_drop_prior = mod_rnn_layer(theta_drop)
    alpha_prior = alpha_layer(theta_drop_prior)

    return models.Model(generator_input, alpha_prior, name="generator")


def _build_discriminator_model(config):
    # Descriminator RNN:
    #   D_{\theta^m_t} = \sigma (f (LSTM_{bi}([\zeta(\hat{\mu}^{m,\theta}_{t}), \zeta(\mu^{m,\theta}_t)],\omega^m_d), \lambda_d^m))
    #   D_{\theta^c_t} = \sigma (f (LSTM_{bi}([\zeta(\hat{\mu}^{c,\theta}_{t}), \zeta(\mu^{c,\theta}_t)],\omega^c_d), \lambda_d^c))

    # Definition of layers
    discriminator_input = layers.Input(
        shape=(config.sequence_length, config.n_modes), name="data"
    )
    drop_layer = layers.TimeDistributed(
        layers.Dropout(config.model_dropout, name="data_drop_des")
    )
    des_rnn_layer = ModelRNNLayer(
        config.des_rnn,
        config.des_normalization,
        config.des_activation,
        config.des_n_layers,
        config.des_n_units,
        config.model_dropout,
        name="des_rnn",
    )
    sigmoid_layer = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))

    # Data flow
    theta_norm_drop = drop_layer(discriminator_input)
    discriminator_sequence = des_rnn_layer(theta_norm_drop)
    discriminator_output = sigmoid_layer(discriminator_sequence)

    return models.Model(discriminator_input, discriminator_output, name="discriminator")
