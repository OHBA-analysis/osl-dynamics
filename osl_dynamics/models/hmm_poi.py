"""Hidden Markov Model (HMM) with a Possion observation model.

"""

import os
import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tqdm.auto import trange
from pqdm.threads import pqdm

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference.layers import (
    CategoricalPoissonLogLikelihoodLossLayer,
    VectorsLayer,
)
from osl_dynamics.models import obs_mod
from osl_dynamics.models.hmm import Model as HMM
from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.utils.misc import set_logging_level

_logger = logging.getLogger("osl-dynamics")


@dataclass
class Config(BaseModelConfig):
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
    state_probs_t0: np.ndarray
        State probabilities at :code:`time=0`. Not trainable.
    observation_update_decay : float
        Decay rate for the learning rate of the observation model.
        We update the learning rate (:code:`lr`) as
        :code:`lr = config.learning_rate * exp(-observation_update_decay *
        epoch)`.

    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
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
    """

    model_name: str = "HMM-Poisson"

    # Observation model parameters
    learn_log_rates: bool = None
    initial_log_rates: np.ndarray = None

    initial_trans_prob: np.ndarray = None
    learn_trans_prob: bool = True
    state_probs_t0: np.ndarray = None

    # Learning rate schedule parameters
    trans_prob_update_delay: float = 5  # alpha
    trans_prob_update_forget: float = 0.7  # beta
    observation_update_decay: float = 0.1

    def __post_init__(self):
        self.validate_observation_model_parameters()
        self.validate_trans_prob_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_observation_model_parameters(self):
        if self.learn_log_rates is None:
            raise ValueError("learn_log_rates must be passed.")

    def validate_trans_prob_parameters(self):
        if self.initial_trans_prob is not None:
            if (
                not isinstance(self.initial_trans_prob, np.ndarray)
                or self.initial_trans_prob.ndim != 2
            ):
                raise ValueError("initial_trans_prob must be a 2D numpy array.")

            if not all(np.isclose(np.sum(self.initial_trans_prob, axis=1), 1)):
                raise ValueError("rows of initial_trans_prob must sum to one.")


class Model(HMM):
    """HMM-Poisson class.

    Parameters
    ----------
    config : osl_dynamics.models.hmm_poi.Config
    """

    config_type = Config

    def get_likelihood(self, x):
        """Get the likelihood, :math:`p(x_t | s_t)`.

        Parameters
        ----------
        x : np.ndarray
            Observed data. Shape is (batch_size, sequence_length, n_channels).

        Returns
        -------
        likelihood : np.ndarray
            Likelihood. Shape is (n_states, batch_size*sequence_length).
        """
        # Get the current observation model parameters
        log_rates = self.get_log_rates()
        n_states = log_rates.shape[0]

        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        # Calculate the log-likelihood for each state to have generated the
        # observed data
        log_likelihood = np.empty([n_states, batch_size, sequence_length])
        for state in range(n_states):
            poi = tf.stop_gradient(
                tfp.distributions.Poisson(
                    log_rate=tf.gather(log_rates, state, axis=-2),
                    allow_nan_stats=False,
                )
            )
            log_likelihood[state] = tf.reduce_sum(poi.log_prob(x), axis=-1)
        log_likelihood = log_likelihood.reshape(n_states, batch_size * sequence_length)

        # We add a constant to the log-likelihood for time points where all
        # states have a negative log-likelihood. This is critical for numerical
        # stability.
        time_points_with_all_states_negative = np.all(log_likelihood < 0, axis=0)
        if np.any(time_points_with_all_states_negative):
            log_likelihood[:, time_points_with_all_states_negative] -= np.max(
                log_likelihood[:, time_points_with_all_states_negative], axis=0
            )

        # Return the likelihood
        return np.exp(log_likelihood)

    def get_log_likelihood(self, data):
        r"""Get the log-likelihood of data, :math:`\log p(x_t | s_t)`.

        Parameters
        ----------
        data : np.ndarray
            Data. Shape is (batch_size, ..., n_channels).

        Returns
        -------
        log_likelihood : np.ndarray
            Log-likelihood. Shape is (batch_size, ..., n_states)
        """
        log_rates = self.get_log_rates()
        poi = tf.stop_gradient(
            tfp.distributions.Poisson(
                log_rate=log_rates,
                allow_nan_stats=False,
            )
        )
        log_likelihood = tf.reduce_sum(
            poi.log_prob(tf.expand_dims(data, axis=-2)), axis=-1
        )
        return log_likelihood.numpy()

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

    def get_n_params_generative_model(self):
        """Get the number of trainable parameters in the generative model.

        This includes the transition probabiltity matrix, state :code:`log_rates`.

        Returns
        -------
        n_params : int
            Number of parameters in the generative model.
        """
        n_params = 0
        if self.config.learn_trans_prob:
            n_params += self.config.n_states * (self.config.n_states - 1)

        for var in self.trainable_weights:
            var_name = var.name
            if "log_rates" in var_name:
                n_params += np.prod(var.shape)

        return int(n_params)

    def fine_tuning(
        self, training_data, n_epochs=None, learning_rate=None, store_dir="tmp"
    ):
        """Fine tuning the model for each session.

        Here, we estimate the posterior distribution (state probabilities)
        and observation model using the data from a single session with the
        group-level transition probability matrix held fixed.

        Parameters
        ----------
        training_data : osl_dynamics.data.Data
            Training dataset.
        n_epochs : int, optional
            Number of epochs to train for. Defaults to the value in the
            :code:`config` used to create the model.
        learning_rate : float, optional
            Learning rate. Defaults to the value in the :code:`config` used
            to create the model.
        store_dir : str, optional
            Directory to temporarily store the model in.

        Returns
        -------
        alpha : list of np.ndarray
            Session-specific mixing coefficients.
            Each element has shape (n_samples, n_states).
        log_rates : np.ndarray
            Session-specific :code:`log_rates`.
            Shape is (n_sessions, n_states, n_channels).
        """
        # Save group-level model parameters
        os.makedirs(store_dir, exist_ok=True)
        self.save_weights(f"{store_dir}/weights.h5")

        # Temporarily change hyperparameters
        original_n_epochs = self.config.n_epochs
        original_learning_rate = self.config.learning_rate
        original_learn_trans_prob = self.config.learn_trans_prob
        self.config.n_epochs = n_epochs or self.config.n_epochs
        self.config.learning_rate = learning_rate or self.config.learning_rate
        self.config.learn_trans_prob = False

        # Reset the optimiser
        self.compile()

        # Fine tune the model for each session
        alpha = []
        log_rates = []
        with set_logging_level(_logger, logging.WARNING):
            for i in trange(training_data.n_sessions, desc="Fine tuning"):
                # Train on this session
                with training_data.set_keep(i):
                    self.fit(training_data, verbose=0)
                    a = self.get_alpha(training_data, concatenate=True)

                # Get the inferred parameters
                m = self.get_log_rates()
                alpha.append(a)
                log_rates.append(m)

                # Reset back to group-level model parameters
                self.load_weights(f"{store_dir}/weights.h5")
                self.compile()

        # Reset hyperparameters
        self.config.n_epochs = original_n_epochs
        self.config.learning_rate = original_learning_rate
        self.config.learn_trans_prob = original_learn_trans_prob

        return alpha, np.array(log_rates)

    def dual_estimation(self, training_data, alpha=None, n_jobs=1):
        """Dual estimation to get session-specific observation model parameters.

        Here, we estimate the state :code:`log_rates` for sessions
        with the posterior distribution of the states held fixed.

        Parameters
        ----------
        training_data : osl_dynamics.data.Data
            Prepared training data object.
        alpha : list of np.ndarray, optional
            Posterior distribution of the states. Shape is
            (n_sessions, n_samples, n_states).
        n_jobs : int, optional
            Number of jobs to run in parallel.

        Returns
        -------
        log_rates : np.ndarray
            Session-specific :code:`log_rates`.
            Shape is (n_sessions, n_states, n_channels).
        """
        if alpha is None:
            # Get the posterior
            alpha = self.get_alpha(training_data, concatenate=False)

        # Validation
        if isinstance(alpha, np.ndarray):
            alpha = [alpha]

        # Get the session-specific data
        data = training_data.time_series(prepared=True, concatenate=False)

        if len(alpha) != len(data):
            raise ValueError(
                "len(alpha) and training_data.n_sessions must be the same."
            )

        # Make sure the data and alpha have the same number of samples
        data = [d[: a.shape[0]] for d, a in zip(data, alpha)]

        n_states = self.config.n_states
        n_channels = self.config.n_channels

        # Helper function for dual estimation for a single session
        def _single_dual_estimation(a, x):
            sum_a = np.sum(a, axis=0)
            if self.config.learn_log_rates:
                session_log_rates = np.empty((n_states, n_channels))
                for state in range(n_states):
                    session_log_rates[state] = (
                        np.sum(x * a[:, state, None], axis=0) / sum_a[state]
                    )
            else:
                session_log_rates = self.get_log_rates()

            return session_log_rates

        # Setup keyword arguments to pass to the helper function
        kwargs = []
        for a, x in zip(alpha, data):
            kwargs.append({"a": a, "x": x})

        if len(data) == 1:
            _logger.info("Dual estimation")
            results = [_single_dual_estimation(**kwargs[0])]

        elif n_jobs == 1:
            results = []
            for i in trange(len(data), desc="Dual estimation"):
                results.append(_single_dual_estimation(**kwargs[i]))

        else:
            _logger.info("Dual estimation")
            results = pqdm(
                kwargs,
                _single_dual_estimation,
                argument_type="kwargs",
                n_jobs=n_jobs,
            )

        return np.squeeze(results)

    def _model_structure(self):
        """Build the model structure."""

        config = self.config

        # Inputs
        inputs = layers.Input(
            shape=(config.sequence_length, config.n_channels + config.n_states),
            name="inputs",
        )
        data, gamma = tf.split(inputs, [config.n_channels, config.n_states], axis=2)

        # Definition of layers
        log_rates_layer = VectorsLayer(
            config.n_states,
            config.n_channels,
            config.learn_log_rates,
            config.initial_log_rates,
            name="log_rates",
        )
        ll_loss_layer = CategoricalPoissonLogLikelihoodLossLayer(
            config.n_states, config.loss_calc, name="ll_loss"
        )

        # Data flow
        log_rates = log_rates_layer(data)  # data not used
        ll_loss = ll_loss_layer([data, log_rates, gamma, None])

        return tf.keras.Model(inputs=inputs, outputs=[ll_loss], name=config.model_name)

    def set_regularizers(self, training_dataset):
        raise NotImplementedError
