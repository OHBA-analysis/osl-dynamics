"""State-Dynamic Network Modelling (State-DyNeMo).

"""

import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm.auto import trange

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference.layers import (
    CategoricalKLDivergenceLayer,
    CategoricalLogLikelihoodLossLayer,
    CovarianceMatricesLayer,
    DiagonalMatricesLayer,
    InferenceRNNLayer,
    KLLossLayer,
    ModelRNNLayer,
    SampleOneHotCategoricalDistributionLayer,
    SoftmaxLayer,
    VectorsLayer,
    StaticLossScalingFactorLayer,
)
from osl_dynamics.models.dynemo import Model as DyNeMo
from osl_dynamics.models.inf_mod_base import VariationalInferenceModelConfig
from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.simulation import HMM

_logger = logging.getLogger("osl-dynamics")


@dataclass
class Config(BaseModelConfig, VariationalInferenceModelConfig):
    """Settings for State-DyNeMo.

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

    inference_rnn : str
        RNN to use, either :code:`'gru'` or :code:`'lstm'`.
    inference_n_layers : int
        Number of layers.
    inference_n_units : int
        Number of units.
    inference_normalization : str
        Type of normalization to use. Either :code:`None`, :code:`'batch'`
        or :code:`'layer'`.
    inference_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. :code:`'relu'`, :code:`'elu'`, etc.
    inference_dropout : float
        Dropout rate.
    inference_regularizer : str
        Regularizer.

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

    do_kl_annealing : bool
        Should we use KL annealing during training?
    kl_annealing_curve : str
        Type of KL annealing curve. Either :code:`'linear'` or :code:`'tanh'`.
    kl_annealing_sharpness : float
        Parameter to control the shape of the annealing curve if
        :code:`kl_annealing_curve='tanh'`.
    n_kl_annealing_epochs : int
        Number of epochs to perform KL annealing.

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
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    """

    model_name: str = "State-DyNeMo"

    # Inference network parameters
    inference_rnn: str = "lstm"
    inference_n_layers: int = 1
    inference_n_units: int = None
    inference_normalization: str = None
    inference_activation: str = None
    inference_dropout: float = 0.0
    inference_regularizer: str = None

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
        if self.inference_n_units is None:
            raise ValueError("Please pass inference_n_units.")

        if self.model_n_units is None:
            raise ValueError("Please pass model_n_units.")

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")


class Model(DyNeMo):
    """State-DyNeMo model class.

    Parameters
    ----------
    config : osl_dynamics.models.state_dynemo.Config
    """

    config_type = Config

    def sample_alpha(self, n_samples, states=None):
        """Uses the model RNN to sample a state probability time course, :code:`alpha`.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.
        states : np.ndarray, optional
            One-hot state vectors to initialise the sampling with.
            Shape must be (sequence_length, n_states).

        Returns
        -------
        alpha : np.ndarray
            Sampled alpha.
        """
        # Get layers
        mod_rnn_layer = self.model.get_layer("mod_rnn")
        mod_theta_layer = self.model.get_layer("mod_theta")
        alpha_layer = self.model.get_layer("alpha")
        states_layer = self.model.get_layer("states")

        if states is None:
            # Sequence of the underlying state time course
            states = np.zeros(
                [self.config.sequence_length, self.config.n_states],
                dtype=np.float32,
            )

            # Randomly sample the first time step
            states[-1] = states_layer(np.random.normal(size=self.config.n_states))

        # Sample the state probability time course
        alpha = np.empty([n_samples, self.config.n_states], dtype=np.float32)
        for i in trange(n_samples, desc="Sampling state time course"):
            # If there are leading zeros we trim the state time course so that
            # we don't pass the zeros
            trimmed_states = states[~np.all(states == 0, axis=1)][np.newaxis, :, :]

            # Predict the probability distribution function for theta one time
            # step in the future, p(theta|state_<t)
            mod_rnn = mod_rnn_layer(trimmed_states)
            mod_theta = mod_theta_layer(mod_rnn)[0, -1]

            # Shift the state time course one time step to the left
            states = np.roll(states, -1, axis=0)

            # Sample from the probability distribution function
            states[-1] = states_layer(mod_theta[np.newaxis, np.newaxis, :][0])

            # Calculate the state time courses
            alpha[i] = alpha_layer(mod_theta[np.newaxis, np.newaxis, :])[0, 0]

        return alpha

    def _model_structure(self):
        """Build the model structure."""

        config = self.config

        # Inputs
        inputs = layers.Input(
            shape=(config.sequence_length, config.n_channels), name="data"
        )

        # Static loss scaling factor
        static_loss_scaling_factor_layer = StaticLossScalingFactorLayer(
            config.sequence_length,
            config.loss_calc,
            name="static_loss_scaling_factor",
        )
        static_loss_scaling_factor = static_loss_scaling_factor_layer(inputs)

        # Inference RNN:
        # - q(state_t) = softmax(theta_t), where theta_t is a set of logits
        inf_rnn_layer = InferenceRNNLayer(
            config.inference_rnn,
            config.inference_normalization,
            config.inference_activation,
            config.inference_n_layers,
            config.inference_n_units,
            config.inference_dropout,
            config.inference_regularizer,
            name="inf_rnn",
        )
        inf_theta_layer = layers.Dense(config.n_states, name="inf_theta")
        alpha_layer = SoftmaxLayer(
            initial_temperature=1.0,
            learn_temperature=False,
            name="alpha",
        )
        states_layer = SampleOneHotCategoricalDistributionLayer(name="states")

        inf_rnn = inf_rnn_layer(inputs)
        inf_theta = inf_theta_layer(inf_rnn)
        alpha = alpha_layer(inf_theta)
        states = states_layer(inf_theta)

        # Observation model:
        # - p(x_t) = N(m_t, C_t), where m_t and C_t are state dependent
        #   means/covariances
        means_layer = VectorsLayer(
            config.n_states,
            config.n_channels,
            config.learn_means,
            config.initial_means,
            config.means_regularizer,
            name="means",
        )
        if config.diagonal_covariances:
            covs_layer = DiagonalMatricesLayer(
                config.n_states,
                config.n_channels,
                config.learn_covariances,
                config.initial_covariances,
                config.covariances_epsilon,
                config.covariances_regularizer,
                name="covs",
            )
        else:
            covs_layer = CovarianceMatricesLayer(
                config.n_states,
                config.n_channels,
                config.learn_covariances,
                config.initial_covariances,
                config.covariances_epsilon,
                config.covariances_regularizer,
                name="covs",
            )
        ll_loss_layer = CategoricalLogLikelihoodLossLayer(
            config.n_states,
            config.covariances_epsilon,
            config.loss_calc,
            name="ll_loss",
        )

        mu = means_layer(
            inputs, static_loss_scaling_factor=static_loss_scaling_factor
        )  # data not used
        D = covs_layer(
            inputs, static_loss_scaling_factor=static_loss_scaling_factor
        )  # data not used
        ll_loss = ll_loss_layer([inputs, mu, D, alpha, None])

        # Model RNN:
        # - p(theta_t | state_<t), predicts logits for the next state based
        #   on a history of states.
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
        mod_theta_layer = layers.Dense(config.n_states, name="mod_theta")
        kl_div_layer = CategoricalKLDivergenceLayer(config.loss_calc, name="kl_div")
        kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")

        mod_rnn = mod_rnn_layer(states)
        mod_theta = mod_theta_layer(mod_rnn)
        kl_div = kl_div_layer([inf_theta, mod_theta])
        kl_loss = kl_loss_layer(kl_div)

        return tf.keras.Model(
            inputs=inputs,
            outputs=[ll_loss, kl_loss, inf_theta],
            name=config.model_name,
        )
