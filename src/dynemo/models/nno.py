"""Class for a Gaussian observation model with means/covariances generated with
a neural network.

"""

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import layers
from dynemo.models.mod_base import BaseConfig, ModelBase
from dynemo.inference.layers import MultiLayerPerceptronLayer, LogLikelihoodLossLayer


@dataclass
class Config(BaseConfig):
    """Settings for NNO.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the generative model.

    mlp_n_layers : int
        Number of layers.
    mlp_n_units : int
        Number of hidden units.
    mlp_normalization : str
        Type of normalization layer.
    mlp_activation : str
        Activation function.
    mlp_dropout_rate : str
        Dropout rate.

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

    # Observation model parameters
    mlp_n_layers: int = 1
    mlp_n_units: int = None
    mlp_normalization: str = None
    mlp_activation: str = None
    mlp_dropout_rate: float = 0.0

    def __post_init__(self):
        self.validate_observation_model_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_observation_model_parameters(self):
        if self.mlp_n_units is None:
            raise ValueError("Please pass mlp_n_units.")


class Model(ModelBase):
    """Multi-layer perceptron observation model (NNO).

    Parameters
    ----------
    config : dynemo.models.nno.Config
    """

    def __init__(self, config):
        ModelBase.__init__(self, config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)


def _model_structure(config):

    # Layers for inputs
    data = layers.Input(shape=(config.sequence_length, config.n_channels), name="data")
    alpha = layers.Input(shape=(config.sequence_length, config.n_modes), name="alpha")

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each mode as the observation model.
    # - We use a multi-layer perceptron to calculate the mean vector and
    #   covariance matrix.
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
    ll_loss = ll_loss_layer([data, m, C])

    return tf.keras.Model(inputs=[data, alpha], outputs=[ll_loss], name="NNO")
