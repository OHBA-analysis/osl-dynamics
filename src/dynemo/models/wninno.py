"""Model class for a WaveNet + Multi-Layer Perceptron generative model.

"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from dynemo.models.mod_base import BaseModelConfig
from dynemo.models.inf_mod_base import InferenceModelConfig, InferenceModelBase
from dynemo.inference.layers import (
    LogLikelihoodLossLayer,
    NormalizationLayer,
    KLDivergenceLayer,
    KLLossLayer,
    SampleNormalDistributionLayer,
    ThetaActivationLayer,
    WaveNetLayer,
    MultiLayerPerceptronLayer,
)


@dataclass
class Config(BaseModelConfig, InferenceModelConfig):
    """Settings for WNINNO.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.

    wn_n_layers : int
        Number of layers for both the inference WaveNet and model WaveNet.
    wn_n_filters : int
        Number of filters for both the inference Wavenet and model WaveNet.

    alpha_xform : str
        Functional form of alpha. Either 'gumbel-softmax', 'softmax' or 'softplus'.
    learn_alpha_temperature : bool
        Should we learn the alpha temperature when alpha_xform='softmax' or
        'gumbel-softmax'?
    initial_alpha_temperature : float
        Initial value for the alpha temperature.

    mlp_n_layers : int
        Number of layers in the multi-layer perceptron observation model.
    mlp_n_units : int
        Number of units in the multi-layer perceptron observation model.
    mlp_normalization : str
        Normalization layer type for the multi-layer perceptron observation model.
    mlp_activation : str
        Activation function for the multi-layer perceptron observation model.
    mlp_dropout_rate : float
        Dropout rate for the multi-layer perceptron observation model.

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

    wn_n_layers: int = None
    wn_n_filters: int = None

    mlp_n_layers: int = None
    mlp_n_units: int = None
    mlp_normalization: str = "batch"
    mlp_activation: str = "selu"
    mlp_dropout_rate: float = 0.0

    multiple_scales: bool = False

    def __post_init__(self):
        self.validate_model_parameters()
        self.validate_alpha_parameters()
        self.validate_kl_annealing_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_model_parameters(self):
        if self.wn_n_layers is None or self.wn_n_filters is None:
            raise ValueError(
                "Please pass WaveNet parameters: wn_n_layers and wn_n_filters."
            )

        if self.mlp_n_layers is None or self.mlp_n_units is None:
            raise ValueError(
                "Please pass MLP parameters: mlp_n_layers and mlp_n_units."
            )


class Model(InferenceModelBase):
    """WaveNet inference/model network and a Multi-Layer Perceptron observation model
    (WNINNO).

    Parameters
    ----------
    config : dynemo.models.wninno.Config
    """

    def __init__(self, config):
        super().__init__(config)

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
    #     - inf_mu    ~ affine(WaveNet(inputs_<=t))
    #     - inf_sigma ~ softplus(WaveNet(inputs_<=t))

    # Definition of layers
    inf_wn_layer = WaveNetLayer(
        config.n_modes,
        config.wn_n_filters,
        config.wn_n_layers,
        name="inf_wn",
    )
    inf_mu_layer = layers.Dense(config.n_modes, name="inf_mu")
    inf_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="inf_sigma"
    )
    theta_layer = SampleNormalDistributionLayer(name="theta")
    alpha_layer = NormalizationLayer("batch", name="alpha")

    # Data flow
    inf_wn = inf_wn_layer(inputs)
    inf_mu = inf_mu_layer(inf_wn)
    inf_sigma = inf_sigma_layer(inf_wn)
    theta = theta_layer([inf_mu, inf_sigma])
    alpha = alpha_layer(theta)

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each mode as the observation model.
    # - We use a multi-layer perceptron to calculate the mean vector and
    #   covariance matrix from the alphas.
    # - We calculate the likelihood of generating the training data with alpha
    #   and the observation model.

    # Definition of layers
    mlp_layer = MultiLayerPerceptronLayer(
        config.mlp_n_layers,
        config.mlp_n_units,
        config.mlp_normalization,
        config.mlp_activation,
        config.mlp_dropout_rate,
        name="mlp",
    )
    mean_layer = layers.Dense(config.n_channels, name="mean")
    cov_layer = layers.Dense(config.n_channels, activation="softplus", name="cov")
    ll_loss_layer = LogLikelihoodLossLayer(diag_cov=True, name="ll_loss")

    # Data flow
    mlp = mlp_layer(alpha)
    m = mean_layer(mlp)
    C = cov_layer(mlp)
    ll_loss = ll_loss_layer([inputs, m, C])

    # Model RNN:
    # - Learns p(theta_t |theta_<t) ~ N(theta_t | mod_mu, mod_sigma), where
    #     - mod_mu    ~ affine(WaveNet(theta_<t))
    #     - mod_sigma ~ softplus(WaveNet(theta_<t))

    # Definition of layers
    mod_wn_layer = WaveNetLayer(
        config.n_channels,
        config.wn_n_filters,
        config.wn_n_layers,
        name="mod_wn",
    )
    mod_mu_layer = layers.Dense(config.n_modes, name="mod_mu")
    mod_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="mod_sigma"
    )
    kl_div_layer = KLDivergenceLayer(name="kl_div")
    kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")

    # Data flow
    mod_wn = mod_wn_layer(alpha)
    mod_mu = mod_mu_layer(mod_wn)
    mod_sigma = mod_sigma_layer(mod_wn)
    kl_div = kl_div_layer([inf_mu, inf_sigma, mod_mu, mod_sigma])
    kl_loss = kl_loss_layer(kl_div)

    return tf.keras.Model(
        inputs=inputs, outputs=[ll_loss, kl_loss, alpha], name="WNINNO"
    )
