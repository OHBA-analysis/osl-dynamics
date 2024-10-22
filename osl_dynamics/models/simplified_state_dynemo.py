"""Simplified State-Dynamic Network Modelling (State-DyNeMo).

"""

import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tqdm.auto import trange

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference.layers import (
    StaticLossScalingFactorLayer,
    ModelRNNLayer,
    CategoricalLogLikelihoodLossLayer,
    SoftmaxLayer,
    ShiftForForecastingLayer,
    VectorsLayer,
    ZeroLayer,
)
from osl_dynamics.models.simplified_dynemo import Model as SimplifiedDyNeMo
from osl_dynamics.models.inf_mod_base import VariationalInferenceModelConfig
from osl_dynamics.models.mod_base import BaseModelConfig

_logger = logging.getLogger("osl-dynamics")


@dataclass
class Config(BaseModelConfig, VariationalInferenceModelConfig):
    """Settings for Simplified State-DyNeMo.

    Parameters
    ----------
    model_name : str
        Model name.
    n_states : int
        Number of states.
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
        Type of normalization to use. Either :code:`None`, :code:`'batch'`
        or :code:`'layer'`.
    model_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. :code:`'relu'`, :code:`'elu'`, etc.
    model_dropout : float
        Dropout rate.
    model_regularizer : str
        Regularizer.

    learn_means : bool
        Should we make the mean vectors for each mode trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each mode trainable?
    initial_means : np.ndarray
        Initialisation for mean vectors.
    initial_covariances : np.ndarray
        Initialisation for mode covariances.
    covariances_epsilon : float
        Error added to standard deviations for numerical stability.
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

    model_name: str = "Simplified-State-DyNeMo"

    # Model network parameters
    model_rnn: str = "lstm"
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: str = None
    model_activation: str = None
    model_dropout: float = 0.0
    model_regularizer: str = None

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None
    diagonal_covariances: bool = False
    covariances_epsilon: float = 1e-6
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

    def __post_init__(self):
        self.validate_rnn_parameters()
        self.validate_observation_model_parameters()
        self.validate_kl_annealing_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_rnn_parameters(self):
        if self.model_n_units is None:
            raise ValueError("Please pass model_n_units.")

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")


class Model(SimplifiedDyNeMo):
    """Simplified State-DyNeMo model class.

    Parameters
    ----------
    config : osl_dynamics.models.simplified_state_dynemo.Config
    """

    config_type = Config

    def sample_alpha(self, n_samples, input_mean=0, input_std=1):
        """Uses the model RNN to sample a state probability time course, :code:`alpha`.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.
        input_mean : float
            Mean of the normal distribution to sample the initial input data from.
            Defaults to 0.
        input_std : float
            Standard deviation of the normal distribution to sample the initial input
            data from. Defaults to 1.

        Returns
        -------
        alpha : np.ndarray
            Sampled alpha.
        """
        # Get layers
        mod_rnn_layer = self.get_layer("mod_rnn")
        theta_layer = self.get_layer("theta")
        alpha_layer = self.get_layer("alpha")

        # Sequence of the input data
        input_data = np.zeros(
            [self.config.sequence_length, self.config.n_channels],
            dtype=np.float32,
        )

        # Randomly sample the first time step
        input_data[-1] = np.random.normal(
            loc=input_mean,
            scale=input_std,
            size=self.config.n_channels,
        )

        # Get observation model
        mu = self.get_means()  # shape: (n_states, n_channels)
        D = self.get_covariances()  # shape: (n_states, n_channels, n_channels)
        mvns = [
            tfp.distributions.MultivariateNormalTriL(
                loc=tf.gather(mu, n, axis=-2),
                scale_tril=tf.linalg.cholesky(tf.gather(D, n, axis=-3)),
                allow_nan_stats=False,
            )
            for n in range(self.config.n_states)
        ]

        # Sample the state probability time course
        alpha = np.empty([n_samples, self.config.n_states], dtype=np.float32)
        for i in trange(n_samples, desc="Sampling state probability time course"):
            # If there are leading zeros we trim the state probabilities so that
            # we don't pass the zeros
            trimmed_input = input_data[~np.all(input_data == 0, axis=1)][
                np.newaxis, :, :
            ]

            # Predict the probability distribution function for theta one time
            # step in the future
            mod_rnn = mod_rnn_layer(trimmed_input)
            theta = theta_layer(mod_rnn)[0, -1]

            # Calculate the state probability time course
            alpha[i] = alpha_layer(theta[np.newaxis, np.newaxis, :])[0, 0]

            # Shift the input data one time step to the left
            input_data = np.roll(input_data, -1, axis=0)

            # Generate the next input data by sampling from the corresponding
            # state-specific observation model
            state = np.argmax(alpha[i])
            input_data[-1] = mvns[state].sample()  # shape: (n_channels,)

        return alpha

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
        theta_layer = layers.Dense(config.n_states, name="theta")
        alpha_layer = SoftmaxLayer(
            initial_temperature=1.0,
            learn_temperature=False,
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
            config.n_states,
            config.n_channels,
            config.learn_means,
            config.initial_means,
            config.means_regularizer,
            name="means",
        )
        covs_layer = self._select_covariance_layer()

        mu = means_layer(
            inputs, static_loss_scaling_factor=static_loss_scaling_factor
        )  # inputs not used
        D = covs_layer(
            inputs, static_loss_scaling_factor=static_loss_scaling_factor
        )  # inputs not used

        # Calculate losses
        ll_loss_layer = CategoricalLogLikelihoodLossLayer(
            config.n_states,
            config.covariances_epsilon,
            config.loss_calc,
            name="ll_loss",
        )
        zero_layer = ZeroLayer(shape=(1,))

        ll_loss = ll_loss_layer([shifted_inputs, mu, D, shifted_alpha, None])
        zero_loss = zero_layer(inputs)  # inputs not used

        return tf.keras.Model(
            inputs=inputs,
            outputs=[ll_loss, zero_loss, theta],
            name=config.model_name,
        )
