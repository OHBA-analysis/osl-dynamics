"""Hidden Markov Model (HMM) with a Possion observation model."""

import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from osl_dynamics.inference.layers import (
    VectorsLayer,
    SeparatePoissonLogLikelihoodLayer,
    HiddenMarkovStateInferenceLayer,
    SumLogLikelihoodLossLayer,
)
from osl_dynamics.models import obs_mod
from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.models.inf_mod_base import (
    MarkovStateInferenceModelConfig,
    MarkovStateInferenceModelBase,
)

_logger = logging.getLogger("osl-dynamics")


@dataclass
class Config(BaseModelConfig, MarkovStateInferenceModelConfig):
    """Settings for HMM-Poisson.

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

    learn_log_rates : bool
        Should we make :code:`log_rate` for each state trainable?
    initial_log_rates : np.ndarray
        Initialisation for state :code:`log_rates`.

    initial_trans_prob : np.ndarray
        Initialisation for the transition probability matrix.
    learn_trans_prob : bool
        Should we make the transition probability matrix trainable?
    trans_prob_update_delay : float
        We update the transition probability matrix as
        :code:`trans_prob = (1-rho) * trans_prob + rho * trans_prob_update`,
        where :code:`rho = (100 * epoch / n_epochs + 1 +
        trans_prob_update_delay) ** -trans_prob_update_forget`.
        This is the delay parameter.
    trans_prob_update_forget : float
        We update the transition probability matrix as
        :code:`trans_prob = (1-rho) * trans_prob + rho * trans_prob_update`,
        where :code:`rho = (100 * epoch / n_epochs + 1 +
        trans_prob_update_delay) ** -trans_prob_update_forget`.
        This is the forget parameter.
    initial_state_probs : np.ndarray
        State probabilities at :code:`time=0`.
    learn_initial_state_probs : bool
        Should we make the initial state probabilities trainable?
    baum_welch_implementation : str
        Which implementation of the Baum-Welch algorithm should we use?
        Either :code:`'log'` (default) or :code:`'rescale'`.

    init_method : str
        Initialization method. Defaults to 'random_state_time_course'.
    n_init : int
        Number of initializations. Defaults to 3.
    n_init_epochs : int
        Number of epochs for each initialization. Defaults to 1.
    init_take : float
        Fraction of dataset to use in the initialization.
        Defaults to 1.0.

    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
    lr_decay : float
        Decay for learning rate. Default is 0.1. We use
        :code:`lr = learning_rate * exp(-lr_decay * epoch)`.
    n_epochs : int
        Number of training epochs.
    optimizer : str or tf.keras.optimizers.Optimizer
        Optimizer to use.
    loss_calc : str
        How should we collapse the time dimension in the loss?
        Either :code:`'mean'` or :code:`'sum'`.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    best_of : int
        Number of full training runs to perform. A single run includes
        its own initialization and fitting from scratch.
    """

    model_name: str = "HMM-Poisson"

    # Observation model parameters
    learn_log_rates: bool = None
    initial_log_rates: np.ndarray = None

    # Initialization
    init_method: str = "random_state_time_course"
    n_init: int = 3
    n_init_epochs: int = 1
    init_take: float = 1.0

    def __post_init__(self):
        self.validate_observation_model_parameters()
        self.validate_hmm_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_observation_model_parameters(self):
        if self.learn_log_rates is None:
            raise ValueError("learn_log_rates must be passed.")


class Model(MarkovStateInferenceModelBase):
    """HMM-Poisson class.

    Parameters
    ----------
    config : osl_dynamics.models.hmm_poi.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""

        config = self.config

        # Inputs
        data = layers.Input(
            shape=(config.sequence_length, config.n_channels),
            name="data",
        )

        # Observation model
        log_rates_layer = VectorsLayer(
            config.n_states,
            config.n_channels,
            config.learn_log_rates,
            config.initial_log_rates,
            name="log_rates",
        )
        log_rates = log_rates_layer(data)  # data not used

        # Log-likelihood
        ll_layer = SeparatePoissonLogLikelihoodLayer(config.n_states, name="ll")
        ll = ll_layer([data, log_rates])

        # Hidden state inference
        hidden_state_inference_layer = HiddenMarkovStateInferenceLayer(
            config.n_states,
            config.sequence_length,
            config.initial_trans_prob,
            config.initial_state_probs,
            config.learn_trans_prob,
            config.learn_initial_state_probs,
            implementation=config.baum_welch_implementation,
            dtype="float64",
            name="hid_state_inf",
        )
        gamma, xi = hidden_state_inference_layer(ll)

        # Loss
        ll_loss_layer = SumLogLikelihoodLossLayer(config.loss_calc, name="ll_loss")
        ll_loss = ll_loss_layer([ll, gamma])

        # Create model
        inputs = {"data": data}
        outputs = {"ll_loss": ll_loss, "gamma": gamma, "xi": xi}
        name = config.model_name
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    def get_log_rates(self):
        """Get the state :code:`log_rates`.

        Returns
        -------
        log_rates : np.ndarray
            State :code:`log_rates`. Shape is (n_states, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "log_rates")

    def get_rates(self):
        """Get the state rates.

        Returns
        -------
        rates : np.ndarray
            State rates. Shape is (n_states, n_channels).
        """
        return np.exp(self.get_log_rates())

    def get_observation_model_parameters(self):
        """Wrapper for :code:`get_log_rates`."""
        return self.get_log_rates()

    def get_log_likelihood(self, x):
        """Get log-likelihood.

        Parameters
        ----------
        data : np.ndarray
            Data to calculate log-likelihood for.
            Shape must be (batch_size, sequence_length, n_channels).

        Returns
        -------
        log_likelihood : np.ndarray
            Log-likelihood. Shape is (batch_size,).
        """
        log_rate = self.get_log_rates()
        ll_layer = self.model.get_layer("ll")
        return ll_layer([x, [log_rate]]).numpy()

    def set_log_rates(self, log_rates, update_initializer=True):
        """Set the state :code:`log_rates`.

        Parameters
        ----------
        log_rates : np.ndarray
            State :code:`log_rates`. Shape is (n_states, n_channels).
        update_initializer : bool, optional
            Do we want to use the passed :code:`log_rates` when we
            re-initialize the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            log_rates,
            layer_name="log_rates",
            update_initializer=update_initializer,
        )

    def set_rates(self, log_rates, epsilon=1e-6, update_initializer=True):
        """Set the state rates.

        Parameters
        ----------
        rates : np.ndarray
            State rates. Shape is (n_states, n_channels).
        update_initializer : bool, optional
            Do we want to use the passed :code:`log_rates` when we
            re-initialize the model?
        """
        log_rates = np.log(log_rates + epsilon)
        self.set_log_rates(log_rates, update_initializer=update_initializer)

    def set_observation_model_parameters(
        self, observation_model_parameters, update_initializer=True
    ):
        """Wrapper for :code:`set_log_rates`."""
        self.set_log_rates(
            observation_model_parameters,
            update_initializer=update_initializer,
        )

    def set_regularizers(self, training_dataset):
        """Set regularizers."""
        raise NotImplementedError

    def set_random_state_time_course_initialization(self, training_dataset):
        """Sets the initial :code:`log_rates` based on a random state time course.

        Parameters
        ----------
        training_dataset : tf.data.Dataset
            Training datas.
        """
        _logger.info("Setting random log_rates")

        # Log_rate for each state
        rates = np.zeros(
            [self.config.n_states, self.config.n_channels], dtype=np.float32
        )

        n_batches = 0
        for batch in training_dataset:
            # Concatenate all the sequences in this batch
            data = np.concatenate(batch["data"])

            # Sample a state time course using the initial transition
            # probability matrix
            stc = self.sample_state_time_course(data.shape[0])

            # Calculate the mean for each state for this batch as log_rate
            rate = []
            for j in range(self.config.n_states):
                x = data[stc[:, j] == 1]
                mu = np.mean(x, axis=0)
                rate.append(mu)
            rates += rate
            n_batches += 1

        # Calculate the average from the running total
        rates /= n_batches

        if self.config.learn_log_rates:
            # Set initial log_rates
            self.set_rates(rates, update_initializer=True)
