"""Simplified DyNeMo.

See the `documentation <https://osl-dynamics.readthedocs.io/en/latest/models\
/dynemo.html>`_ for a description of this model.

See Also
--------
- C. Gohil, et al., "Mixtures of large-scale functional brain network modes".
  `Neuroimage 263, 119595 (2022) <https://www.sciencedirect.com/science\
  /article/pii/S1053811922007108>`_.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from osl_dynamics.inference.layers import (
    StaticLossScalingFactorLayer,
    ModelRNNLayer,
    SoftmaxLayer,
    ShiftForForecastingLayer,
    VectorsLayer,
    MixMatricesLayer,
    MixVectorsLayer,
    LogLikelihoodLossLayer,
    ZeroLayer,
)
from osl_dynamics.models.dynemo import Model as DyNeMo
from osl_dynamics.models.inf_mod_base import VariationalInferenceModelConfig
from osl_dynamics.models.mod_base import BaseModelConfig


@dataclass
class Config(BaseModelConfig, VariationalInferenceModelConfig):
    """Settings for Simplified DyNeMo.

    Parameters
    ----------
    model_name : str
        Model name.
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.

    model_rnn : str
        RNN to use, either :code:`'gru'` or :code:`'lstm'`.
    model_n_layers : int
        Number of layers.
    model_n_units : int
        Number of units.
    model_normalization : str
        Type of normalization to use. Either None, :code:`'batch'` or
        :code:`'layer'`.
    model_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. :code:`'relu'`, :code:`'elu'`, etc.
    model_dropout : float
        Dropout rate.
    model_regularizer : str
        Regularizer.

    learn_alpha_temperature : bool
        Should we learn :code:`alpha_temperature`?
    initial_alpha_temperature : float
        Initial value for :code:`alpha_temperature`.

    learn_means : bool
        Should we make the mean vectors for each mode trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each mode trainable?
    initial_means : np.ndarray
        Initialisation for mean vectors.
    initial_covariances : np.ndarray
        Initialisation for state covariances.
        If :code:`diagonal_covariances=True` and full matrices are passed,
        the diagonal is extracted.
    covariances_epsilon : float
        Error added to mode covariances for numerical stability.
    diagonal_covariances : bool
        Should we learn diagonal mode covariances?
    means_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for mean vectors.
    covariances_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for covariance matrices.

    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
    lr_decay : float
        Decay for learning rate. Default is 0.1. We use
        :code:`lr = learning_rate * exp(-lr_decay * epoch)`.
    gradient_clip : float
        Value to clip gradients by. This is the :code:`clipnorm` argument
        passed to the Keras optimizer. Cannot be used if :code:`multi_gpu=True`.
    n_epochs : int
        Number of training epochs.
    optimizer : str or tf.keras.optimizers.Optimizer
        Optimizer to use. :code:`'adam'` is recommended.
    loss_calc : str
        How should we collapse the time dimension in the loss?
        Either :code:`'mean'` or :code:`'sum'`.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    """

    model_name: str = "Simplified-DyNeMo"

    model_rnn: str = "lstm"
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: str = None
    model_activation: str = None
    model_dropout: float = 0.0
    model_regularizer: str = None

    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None
    diagonal_covariances: bool = False
    covariances_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

    def __post_init__(self):
        self.validate_rnn_parameters()
        self.validate_observation_model_parameters()
        self.validate_alpha_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_rnn_parameters(self):
        if self.model_n_units is None:
            raise ValueError("Please pass model_n_units.")

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")

        if self.covariances_epsilon is None:
            if self.learn_covariances:
                self.covariances_epsilon = 1e-6
            else:
                self.covariances_epsilon = 0.0


class Model(DyNeMo):
    """Simplified DyNeMo model class.

    Parameters
    ----------
    config : osl_dynamics.models.simplified_dynemo.Config
    """

    config_type = Config

    def sample_alpha(self):
        raise NotImplementedError

    def fine_tuning(self):
        raise NotImplementedError

    def dual_estimation(self):
        raise NotImplementedError

    def _model_structure(self):
        """Build the model structure."""

        config = self.config

        # Inputs
        inputs = layers.Input(
            shape=(config.sequence_length, config.n_channels), name="data"
        )

        static_loss_scaling_factor_layer = StaticLossScalingFactorLayer(
            config.sequence_length,
            config.loss_calc,
            name="static_loss_scaling_factor",
        )
        static_loss_scaling_factor = static_loss_scaling_factor_layer(inputs)

        # Model RNN: predicts the next mixing coefficients based on historic
        # observed data
        data_drop_layer = layers.Dropout(config.model_dropout, name="data_drop")
        mod_rnn_layer = ModelRNNLayer(
            config.model_rnn,
            config.model_normalization,
            config.model_activation,
            config.model_n_layers,
            config.model_n_units,
            config.model_dropout,
            config.model_regularizer,
            name="mod_rnn",
        )
        theta_layer = layers.Dense(config.n_modes, name="theta")
        alpha_layer = SoftmaxLayer(
            config.initial_alpha_temperature,
            config.learn_alpha_temperature,
            name="alpha",
        )

        data_drop = data_drop_layer(inputs)
        mod_rnn = mod_rnn_layer(data_drop)
        theta = theta_layer(mod_rnn)
        alpha = alpha_layer(theta)

        # Shift data and inferred mixing coefficients to ensure the model is
        # forecasting future values
        shift_for_forecasting_layer = ShiftForForecastingLayer(clip=1, name="shift")

        shifted_alpha, shifted_inputs = shift_for_forecasting_layer([alpha, inputs])

        # Observation model: calculate the probability of the observed
        # data given the mixing coefficients
        means_layer = VectorsLayer(
            config.n_modes,
            config.n_channels,
            config.learn_means,
            config.initial_means,
            config.means_regularizer,
            name="means",
        )
        covs_layer = self._select_covariance_layer()
        mix_means_layer = MixVectorsLayer(name="mix_means")
        mix_covs_layer = MixMatricesLayer(name="mix_covs")

        mu = means_layer(
            inputs, static_loss_scaling_factor=static_loss_scaling_factor
        )  # inputs not used
        D = covs_layer(
            inputs, static_loss_scaling_factor=static_loss_scaling_factor
        )  # inputs not used
        m = mix_means_layer([shifted_alpha, mu])
        C = mix_covs_layer([shifted_alpha, D])

        # Calculate losses
        ll_loss_layer = LogLikelihoodLossLayer(
            config.covariances_epsilon,
            config.loss_calc,
            name="ll_loss",
        )
        zero_layer = ZeroLayer(shape=(1,))

        ll_loss = ll_loss_layer([shifted_inputs, m, C])
        zero_loss = zero_layer(inputs)  # inputs not used

        return tf.keras.Model(
            inputs=inputs,
            outputs=[ll_loss, zero_loss, theta],
            name=config.model_name,
        )
