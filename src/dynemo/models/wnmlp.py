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
    MultiLayerPerceptronLayer,
    WaveNetLayer,
    SampleNormalDistributionLayer,
    LogLikelihoodLossLayer,
    KLDivergenceLayer,
    KLLossLayer,
)


@dataclass
class Config(BaseModelConfig, InferenceModelConfig):
    """Settings for WNMLP.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.

    inf_mlp_n_layers : int
        Number of layers in the inference multi-layer perceptron.
    inf_mlp_n_units : int
        Number of units in the inference multi-layer perceptron.
    inf_mlp_normalization : str
        Normalization layer type for the inference multi-layer perceptron.
    inf_mlp_activation : str
        Activation function for the inference multi-layer perceptron.
    inf_mlp_dropout_rate : float
        Dropout rate for the inference multi-layer perceptron.

    obs_mlp_n_layers : int
        Number of layers in the observation model multi-layer perceptron.
    obs_mlp_n_units : int
        Number of units in the observation model multi-layer perceptron.
    obs_mlp_normalization : str
        Normalization layer type for the observation model multi-layer perceptron.
    obs_mlp_activation : str
        Activation function for the observation model multi-layer perceptron.
    obs_mlp_dropout_rate : float
        Dropout rate for the observation model multi-layer perceptron.

    mod_wn_n_layers : int
        Number of layers for the model WaveNet.
    mod_wn_n_filters : int
        Number of filters for the model WaveNet.

    alpha_xform : str
        Functional form of alpha. Either 'gumbel-softmax', 'softmax' or 'softplus'.
    learn_alpha_temperature : bool
        Should we learn the alpha temperature when alpha_xform='softmax' or
        'gumbel-softmax'?
    initial_alpha_temperature : float
        Initial value for the alpha temperature.

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

    inf_mlp_n_layers: int = None
    inf_mlp_n_units: int = None
    inf_mlp_normalization: str = "batch"
    inf_mlp_activation: str = "selu"
    inf_mlp_dropout_rate: float = 0.0

    obs_mlp_n_layers: int = None
    obs_mlp_n_units: int = None
    obs_mlp_normalization: str = "batch"
    obs_mlp_activation: str = "selu"
    obs_mlp_dropout_rate: float = 0.0

    mod_wn_n_layers: int = None
    mod_wn_n_filters: int = None

    multiple_scales: bool = False

    def __post_init__(self):
        self.validate_alpha_parameters()
        self.validate_kl_annealing_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()


class Model(InferenceModelBase):
    """WaveNet+Multi-Layer Perceptron observation model and Multi-Layer Perceptron
    inference network (WNMLP).

    Parameters
    ----------
    config : dynemo.models.wnmlp.Config
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
    #     - inf_mu    = affine(MLP(inputs_t))
    #     - inf_sigma = softplus(MLP(inputs_t))

    # Definition of layers
    inf_mlp_layer = MultiLayerPerceptronLayer(
        config.inf_mlp_n_layers,
        config.inf_mlp_n_units,
        config.inf_mlp_normalization,
        config.inf_mlp_activation,
        config.inf_mlp_dropout_rate,
        name="inf_mlp",
    )
    inf_mu_layer = layers.Dense(config.n_modes, name="inf_mu")
    inf_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="inf_sigma"
    )
    alpha_layer = SampleNormalDistributionLayer(name="alpha")

    # Data flow
    inf_mlp = inf_mlp_layer(inputs)
    inf_mu = inf_mu_layer(inf_mlp)
    inf_sigma = inf_sigma_layer(inf_mlp)
    alpha = alpha_layer([inf_mu, inf_sigma])

    # Observation model:
    # - Learns p(X_t | alpha_t) = N(X_t | m, C), where
    #     - m = MLP(alpha)
    #     - C  = softplus(MLP(alpha))

    # Definition of layers
    obs_mlp_layer = MultiLayerPerceptronLayer(
        config.obs_mlp_n_layers,
        config.obs_mlp_n_units,
        config.obs_mlp_normalization,
        config.obs_mlp_activation,
        config.obs_mlp_dropout_rate,
        name="obs_mlp",
    )
    mean_layer = layers.Dense(config.n_channels, name="mean")
    cov_layer = layers.Dense(config.n_channels, activation="softplus", name="cov")
    ll_loss_layer = LogLikelihoodLossLayer(
        diag_cov=True, clip_start=config.sequence_length // 2 + 1, name="ll_loss"
    )

    # Data flow
    obs_mlp = obs_mlp_layer(alpha)
    m = mean_layer(obs_mlp)
    C = cov_layer(obs_mlp)
    ll_loss = ll_loss_layer([inputs, m, C])

    # Model RNN:
    # - Learns p(theta_t | theta_<t) ~ N(theta_t | mod_mu, mod_sigma), where
    #     - mod_mu = affine(WaveNet(alpha_<t))
    #     - mod_sigma = softplus(WaveNet(alpha_<t))

    # Definition of layers
    mod_wn_layer = WaveNetLayer(
        config.mod_wn_n_filters,
        config.mod_wn_n_filters,
        config.mod_wn_n_layers,
        name="mod_wn",
    )
    mod_mu_layer = layers.Dense(config.n_modes, name="mod_mu")
    mod_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="mod_sigma"
    )
    kl_div_layer = KLDivergenceLayer(
        clip_start=config.sequence_length // 2, name="kl_div"
    )
    kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")

    # Data flow
    mod_wn = mod_wn_layer(alpha)
    mod_mu = mod_mu_layer(mod_wn)
    mod_sigma = mod_sigma_layer(mod_wn)
    kl_div = kl_div_layer([inf_mu, inf_sigma, mod_mu, mod_sigma])
    kl_loss = kl_loss_layer(kl_div)

    return tf.keras.Model(
        inputs=inputs, outputs=[ll_loss, kl_loss, alpha], name="WNMLP"
    )
