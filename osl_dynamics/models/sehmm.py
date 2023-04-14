"""Subject Embedding HMM.
"""

import logging
import os.path as op
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from numba.core.errors import NumbaWarning
from scipy.special import logsumexp, xlogy
from tensorflow.keras import backend, layers, utils, initializers
from tqdm.auto import trange

import osl_dynamics.data.tf as dtf
from osl_dynamics import inference
from osl_dynamics.inference.layers import (
    CategoricalLogLikelihoodLossLayer,
    LearnableTensorLayer,
    VectorsLayer,
    CovarianceMatricesLayer,
    ConcatEmbeddingsLayer,
    SubjectMapLayer,
    TFRangeLayer,
    ZeroLayer,
    InverseCholeskyLayer,
    SampleGammaDistributionLayer,
    StaticKLDivergenceLayer,
    KLLossLayer,
    MultiLayerPerceptronLayer,
)
from osl_dynamics.models import dynemo_obs, sedynemo_obs
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.simulation import HMM

_logger = logging.getLogger("osl-dynamics")

warnings.filterwarnings("ignore", category=NumbaWarning)

EPS = sys.float_info.epsilon


@dataclass
class Config(BaseModelConfig):
    """Settings for Subject Embedding HMM.

    Parameters
    ----------
    model_name : str
        Name of the model.
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of the sequences passed to the generative model.
    learn_means : bool
        Should we make the group mean vectors for each state trainable?
    learn_covariances : bool
        Should we make the group covariance matrix for each state trainable?
    initial_means : np.ndarray
        Initialisation for group level state means.
    initial_covariances : np.ndarray
        Initialisation for group level state covariances.
    covariances_epsilon : float
        Error added to mode covariances for numerical stability.
    means_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for group mean vectors.
    covariances_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for group covariance matrices.

    n_subjects : int
        Number of subjects.
    subject_embeddings_dim : int
        Number of dimensions for subject embeddings.
    mode_embeddings_dim : int
        Number of dimensions for mode embeddings.

    dev_n_layers : int
        Number of layers for the MLP for deviations.
    dev_n_units : int
        Number of units for the MLP for deviations.
    dev_normalization : str
        Type of normalization for the MLP for deviations.
        Either None, 'batch' or 'layer'.
    dev_activation : str
        Type of activation to use for the MLP for deviations.
        E.g. 'relu', 'sigmoid', 'tanh', etc.
    dev_dropout : float
        Dropout rate for the MLP for deviations.

    initial_trans_prob : np.ndarray
        Initialisation for trans prob matrix
    learn_trans_prob : bool
        Should we make the trans prob matrix trainable?
    state_probs_t0: np.ndarray
        State probabilities at time=0. Not trainable.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
    trans_prob_update_delay : float
        We update the transition probability matrix as
        trans_prob = (1-rho) * trans_prob + rho * trans_prob_update,
        where rho = (100 * epoch / n_epochs + 1 + trans_prob_update_delay)
        ** -trans_prob_update_forget. This is the delay parameter.
    trans_prob_update_forget : float
        We update the transition probability matrix as
        trans_prob = (1-rho) * trans_prob + rho * trans_prob_update,
        where rho = (100 * epoch / n_epochs + 1 + trans_prob_update_delay)
        ** -trans_prob_update_forget. This is the forget parameter.
    observation_update_decay : float
        Decay rate for the learning rate of the observation model.
        We update the learning rate (lr) as
        lr = config.learning_rate * exp(-observation_update_decay * epoch).
    n_epochs : int
        Number of training epochs.
    optimizer : str or tensorflow.keras.optimizers.Optimizer
        Optimizer to use.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    """

    model_name: str = "SE-HMM"

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None
    covariances_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

    # Parameters specific to subject embedding model
    n_subjects: int = None
    subject_embeddings_dim: int = None
    mode_embeddings_dim: int = None

    dev_n_layers: int = 0
    dev_n_units: int = None
    dev_normalization: str = None
    dev_activation: str = None
    dev_dropout: float = 0.0

    initial_trans_prob: np.ndarray = None
    learn_trans_prob: bool = True
    state_probs_t0: np.ndarray = None

    # Learning rate schedule parameters
    trans_prob_update_delay: float = 5  # alpha
    trans_prob_update_forget: float = 0.7  # beta
    observation_update_decay: float = 0.1

    def __post_init__(self):
        self.validate_observation_model_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()
        self.validate_subject_embedding_parameters()

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")

        if self.covariances_epsilon is None:
            if self.learn_covariances:
                self.covariances_epsilon = 1e-6
            else:
                self.covariances_epsilon = 0.0

    def validate_subject_embedding_parameters(self):
        if (
            self.n_subjects is None
            or self.subject_embeddings_dim is None
            or self.mode_embeddings_dim is None
        ):
            raise ValueError(
                "n_subjects, subject_embeddings_dim and mode_embeddings_dim must be passed."
            )

        if self.dev_n_layers != 0 and self.dev_n_units is None:
            raise ValueError("Please pass dev_inf_n_units.")


class Model(ModelBase):
    """Subject Embedding HMM class.

    Parameters
    ----------
    config : osl_dynamics.models.hmm.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

        self.rho = 1
        self.set_trans_prob(self.config.initial_trans_prob)
        self.set_state_probs_t0(self.config.state_probs_t0)

    def fit(self, dataset, epochs=None, use_tqdm=False, **kwargs):
        """Fit model to a dataset.

        Iterates between:

        - Baum-Welch updates of latent variable time courses and transition
          probability matrix.
        - TensorFlow updates of observation model parameters.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        epochs : int
            Number of epochs.
        kwargs : keyword arguments
            Keyword arguments for the TensorFlow observation model training.
            These keywords arguments will be passed to self.model.fit().

        Returns
        -------
        history : dict
            Dictionary with history of the loss and learning rates (lr and rho).
        """
        if epochs is None:
            epochs = self.config.n_epochs

        # Make a TensorFlow dataset
        dataset = self.make_dataset(
            dataset, shuffle=True, concatenate=True, subj_id=True
        )

        # Training curves
        history = {"loss": [], "rho": [], "lr": []}

        # Loop through epochs
        if use_tqdm:
            _range = trange(epochs)
        else:
            _range = range(epochs)
        for n in _range:
            # Setup a progress bar for this epoch
            if not use_tqdm:
                print("Epoch {}/{}".format(n + 1, epochs))
                pb_i = utils.Progbar(len(dataset))

            # Update rho
            self._update_rho(n)

            # Set learning rate for the observation model
            lr = self.config.learning_rate * np.exp(
                -self.config.observation_update_decay * n
            )
            backend.set_value(self.model.optimizer.lr, lr)

            # Loop through batches
            loss = []
            for data in dataset:
                x = data["data"]
                subj_id = data["subj_id"]

                # Update state probabilities
                gamma, xi = self._get_state_probs(x, subj_id)

                # Update transition probability matrix
                if self.config.learn_trans_prob:
                    self._update_trans_prob(gamma, xi)

                # Reshape gamma: (batch_size*sequence_length, n_states)
                # -> (batch_size, sequence_length, n_states)
                gamma = gamma.reshape(x.shape[0], x.shape[1], -1)

                # Update observation model parameters
                x_gamma_and_subj_id = np.concatenate(
                    [x, gamma, np.expand_dims(subj_id, -1)], axis=2
                )
                h = self.model.fit(x_gamma_and_subj_id, epochs=1, verbose=0, **kwargs)

                # Get the new loss
                l = h.history["loss"][0]
                if np.isnan(l):
                    _logger.error("Training failed!")
                    return
                loss.append(l)

                # Update progress bar
                if use_tqdm:
                    _range.set_postfix(rho=self.rho, lr=lr, loss=l)
                else:
                    pb_i.add(1, values=[("rho", self.rho), ("lr", lr), ("loss", l)])

            history["loss"].append(np.mean(loss))
            history["rho"].append(self.rho)
            history["lr"].append(lr)

        if use_tqdm:
            _range.close()

        return history

    def random_subset_initialization(
        self, training_data, n_epochs, n_init, take, **kwargs
    ):
        """Random subset initialization.

        The model is trained for a few epochs with different random subsets
        of the training dataset. The model with the best free energy is kept.

        Parameters
        ----------
        training_data : tensorflow.data.Dataset or osl_dynamics.data.Data
            Dataset to use for training.
        n_epochs : int
            Number of epochs to train the model.
        n_init : int
            Number of initializations.
        take : float
            Fraction of total batches to take.
        kwargs : keyword arguments
            Keyword arguments for the fit method.

        Returns
        -------
        history : history
            The training history of the best initialization.
        """
        if n_init is None or n_init == 0:
            _logger.info(
                "Number of initializations was set to zero. "
                + "Skipping initialization."
            )
            return

        _logger.info("Random subset initialization")

        # Make a TensorFlow Dataset
        training_data = self.make_dataset(training_data, concatenate=True, subj_id=True)

        # Calculate the number of batches to use
        n_total_batches = dtf.get_n_batches(training_data)
        n_batches = max(round(n_total_batches * take), 1)
        _logger.info(f"Using {n_batches} out of {n_total_batches} batches")

        # Pick the initialization with the lowest free energy
        best_loss = np.Inf
        for n in range(n_init):
            _logger.info(f"Initialization {n}")
            self.reset()
            training_data_subset = training_data.shuffle(100000).take(n_batches)
            history = self.fit(training_data_subset, epochs=n_epochs, **kwargs)
            if history is None:
                continue
            loss = history["loss"][-1]
            if loss < best_loss:
                best_initialization = n
                best_loss = loss
                best_history = history
                best_weights = self.get_weights()
                best_trans_prob = self.trans_prob

        if best_loss == np.Inf:
            _logger.error("Initialization failed")
            return

        _logger.info(f"Using initialization {best_initialization}")
        self.set_weights(best_weights, best_trans_prob)

        return best_history

    def random_state_time_course_initialization(
        self, training_data, n_epochs, n_init, take=1, **kwargs
    ):
        """Random state time course initialization.

        The model is trained for a few epochs with a sampled state time course
        initialization. The model with the best free energy is kept.

        Parameters
        ----------
        training_data : tensorflow.data.Dataset or osl_dynamics.data.Data
            Dataset to use for training.
        n_epochs : int
            Number of epochs to train the model.
        n_init : int
            Number of initializations.
        take : float
            Fraction of total batches to take.
        kwargs : keyword arguments
            Keyword arguments for the fit method.

        Returns
        -------
        history : history
            The training history of the best initialization.
        """
        if n_init is None or n_init == 0:
            _logger.info(
                "Number of initializations was set to zero. "
                + "Skipping initialization."
            )
            return

        _logger.info("Random state time course initialization")

        # Make a TensorFlow Dataset
        training_dataset = self.make_dataset(
            training_data, concatenate=True, subj_id=True
        )

        # Calculate the number of batches to use
        n_total_batches = dtf.get_n_batches(training_dataset)
        n_batches = max(round(n_total_batches * take), 1)
        _logger.info(f"Using {n_batches} out of {n_total_batches} batches")

        # Pick the initialization with the lowest free energy
        best_loss = np.Inf
        for n in range(n_init):
            _logger.info(f"Initialization {n}")
            self.reset()
            training_data_subset = training_dataset.shuffle(100000).take(n_batches)
            self.set_random_state_time_course_initialization(training_data_subset)
            history = self.fit(training_data_subset, epochs=n_epochs, **kwargs)
            if history is None:
                continue
            loss = history["loss"][-1]
            if loss < best_loss:
                best_initialization = n
                best_loss = loss
                best_history = history
                best_weights = self.get_weights()
                best_trans_prob = self.trans_prob

        if best_loss == np.Inf:
            _logger.error("Initialization failed")
            return

        _logger.info(f"Using initialization {best_initialization}")
        self.set_weights(best_weights, best_trans_prob)

        return best_history

    def _get_state_probs(self, x, subj_id):
        """Get state probabilities.

        Parameters
        ----------
        x : np.ndarray
            Observed data. Shape is (batch_size, sequence_length, n_channels).
        subj_id : np.ndarray
            Subject ID. Shape is (batch_size, sequence_length).

        Returns
        -------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data, q(s_t).
            Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive time points,
            q(s_t, s_t+1). Shape is (batch_size*sequence_length-1, n_states*n_states).
        """

        # Use Baum-Welch algorithm to calculate gamma, xi
        B = self._get_likelihood(x, subj_id)
        Pi_0 = self.state_probs_t0
        P = self.trans_prob

        gamma, xi = self._baum_welch(B, Pi_0, P)

        return gamma, xi

    @numba.jit
    def _baum_welch(self, B, Pi_0, P):
        """Hidden state inference using the Baum-Welch algorithm.

        Parameters
        ----------
        B : np.ndarray
            Probability of individual data points, under observation model for
            each state. Shape is (n_states, n_samples).
        Pi_0 : np.ndarray
            Initial state probabilities. Shape is (n_states,).
        P : np.ndarray
            State transition probabilities. Shape is (n_states, n_states).

        Returns
        -------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data, q(s_t).
            Shape is (n_samples, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive time points,
            q(s_t, s_t+1). Shape is (n_samples-1, n_states*n_states).
        """
        n_samples = B.shape[1]
        n_states = B.shape[0]

        # Memory allocation
        alpha = np.empty((n_samples, n_states))
        beta = np.empty((n_samples, n_states))
        scale = np.empty(n_samples)

        # Forward pass
        alpha[0] = Pi_0 * B[:, 0]
        scale[0] = np.sum(alpha[0])
        alpha[0] /= scale[0] + EPS
        for i in range(1, n_samples):
            alpha[i] = (alpha[i - 1] @ P) * B[:, i]
            scale[i] = np.sum(alpha[i])
            alpha[i] /= scale[i] + EPS

        # Backward pass
        beta[-1] = 1.0 / (scale[-1] + EPS)
        for i in range(2, n_samples + 1):
            beta[-i] = (beta[-i + 1] * B[:, -i + 1]) @ P.T
            beta[-i] /= scale[-i] + EPS

        # Marginal probabilities
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        # Joint probabilities
        b = beta[1:] * B[:, 1:].T
        t = P * np.expand_dims(alpha[:-1], axis=2) * np.expand_dims(b, axis=1)
        xi = t.reshape(n_samples - 1, -1, order="F")
        xi /= np.expand_dims(np.sum(xi, axis=1), axis=1) + EPS

        return gamma, xi

    def _get_likelihood(self, x, subj_id):
        """Get the likelihood, p(x_t | s_t).

        Parameters
        ----------
        x : np.ndarray
            Observed data. Shape is (batch_size, sequence_length, n_channels).
        subj_id : np.ndarray
            Subject ID. Shape is (batch_size, sequence_length).

        Returns
        -------
        likelihood : np.ndarray
            Likelihood. Shape is (n_states, batch_size*sequence_length).
        """
        # Get the current observation model parameters
        means, covs = self.get_subject_means_covs()

        n_states = means.shape[1]
        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        subj_id = tf.cast(subj_id, tf.int32)
        means = tf.gather(means, subj_id)
        covs = tf.gather(covs, subj_id)

        # Calculate the log-likelihood for each state to have generated the
        # observed data
        log_likelihood = np.empty([n_states, batch_size, sequence_length])
        for state in range(n_states):
            mvn = tfp.distributions.MultivariateNormalTriL(
                loc=means[:, :, state, :],
                scale_tril=tf.linalg.cholesky(covs[:, :, state, :, :]),
                allow_nan_stats=False,
            )
            log_likelihood[state] = mvn.log_prob(x)
        log_likelihood = log_likelihood.reshape(n_states, batch_size * sequence_length)

        # We add a constant to the log-likelihood for time points where all states
        # have a negative log-likelihood. This is critical for numerical stability.
        time_points_with_all_states_negative = np.all(log_likelihood < 0, axis=0)
        if np.any(time_points_with_all_states_negative):
            log_likelihood[:, time_points_with_all_states_negative] -= np.max(
                log_likelihood[:, time_points_with_all_states_negative], axis=0
            )

        # Return the likelihood
        return np.exp(log_likelihood)

    def _update_trans_prob(self, gamma, xi):
        """Update transition probability matrix.

        Parameters
        ----------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data, q(s_t).
            Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive time points,
            q(s_t, s_t+1). Shape is (batch_size*sequence_length-1, n_states*n_states).
        """
        # Calculate the new transition probability matrix using the posterior from
        # the Baum-Welch algorithm:
        #
        # p(s_t+1 | s_t) = E{q(s_t, s_t+1)} / E{q(s_t)}
        #                = sum^{T-1}_{t=1} xi(t, t+1) / sum^{T-1}_{t=1} gamma(t)
        #
        # where E{.} denotes the expectation.
        phi_interim = np.sum(xi, axis=0).reshape(
            self.config.n_states, self.config.n_states
        ).T / np.sum(gamma[:-1], axis=0).reshape(self.config.n_states, 1)

        # We use stochastic updates on trans_prob as per Eqs. (1) and (2) in the paper:
        # https://www.sciencedirect.com/science/article/pii/S1053811917305487
        self.trans_prob = (1 - self.rho) * self.trans_prob + self.rho * phi_interim

    def _update_rho(self, ind):
        """Update rho.

        Parameters
        ---------
        ind : int
            Index of iteration.
        """
        # Calculate new value, using modified version of Eq. (2) to account for
        # total number of iterations:
        # https://www.sciencedirect.com/science/article/pii/S1053811917305487
        self.rho = np.power(
            100 * ind / self.config.n_epochs + 1 + self.config.trans_prob_update_delay,
            -self.config.trans_prob_update_forget,
        )

    def _get_posterior_entropy(self, gamma, xi):
        """Posterior entropy.

        Calculate the entropy of the posterior distribution:

        .. math::
            E &= \int q(s_{1:T}) \log q(s_{1:T}) ds_{1:T}

              &= \displaystyle\sum_{t=1}^{T-1} \int q(s_t, s_{t+1}) \log q(s_t, s_{t+1}) ds_t ds_{t+1} - \displaystyle\sum_{t=2}^{T-1} \int q(s_t) \log q(s_t) ds_t

        Parameters
        ----------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data, q(s_t).
            Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive time points,
            q(s_t, s_t+1). Shape is (batch_size*sequence_length-1, n_states*n_states).

        Returns
        -------
        entropy : float
            Entropy.
        """
        # first_term = sum^{T-1}_t=1 int q(s_t, s_t+1) log(q(s_t, s_t+1)) ds_t ds_t+1
        first_term = np.sum(xlogy(xi, xi))

        # second_term = sum^{T-1}_t=2 int q(s_t) log q(s_t) ds_t
        second_term = np.sum(xlogy(gamma, gamma)[1:-1])

        return first_term - second_term

    def _get_posterior_expected_log_likelihood(self, x, gamma, subj_id):
        """Expected log-likelihood.

        Calculates the expected log-likelihood with respect to the posterior
        distribution of the states:

        .. math::
            LL &= \int q(s_{1:T}) \log \prod_{t=1}^T p(x_t | s_t) ds_{1:T}

               &= \sum_{t=1}^T \int q(s_t) \log p(x_t | s_t) ds_t

        Parameters
        ----------
        x : np.ndarray
            Data. Shape is (batch_size, sequence_length, n_channels).
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data, q(s_t).
            Shape is (batch_size*sequence_length, n_states).
        subj_id : np.ndarray
            Subject ID. Shape is (batch_size, sequence_length).

        Returns
        -------
        log_likelihood : float
            Posterior expected log-likelihood.
        """
        gamma = np.reshape(gamma, (x.shape[0], x.shape[1], -1))
        log_likelihood = self._get_log_likelihood(x, subj_id)
        return tf.reduce_sum(log_likelihood * gamma)

    def _get_posterior_expected_prior(self, gamma, xi):
        """Posterior expected prior.

        Calculates the expected prior probability of states with respect to the
        posterior distribution of the states:

        .. math::
            P &= \int q(s_{1:T}) \log p(s_{1:T}) ds

              &= \int q(s_1) \log p(s_1) ds_1 + \displaystyle\sum_{t=1}^{T-1} \int q(s_t, s_{t+1}) \log p(s_{t+1} | s_t) ds_t ds_{t+1}

        Parameters
        ----------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data, q(s_t).
            Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive time points,
            q(s_t, s_t+1). Shape is (batch_size*sequence_length-1, n_states*n_states).

        Returns
        -------
        prior : float
            Posterior expected prior probability.
        """
        n_samples, n_states = gamma.shape

        # first_term = int q(s_1) log p(s_1) ds_1
        first_term = np.sum(xlogy(gamma[0], self.state_probs_t0))

        # remaining_terms =
        # sum^{T-1}_t=1 int q(s_t, s_t+1) log p(s_t+1 | s_t}) ds_t ds_t+1
        remaining_terms = np.sum(
            xlogy(
                xi.reshape(n_samples - 1, n_states, n_states, order="F"),
                np.expand_dims(self.trans_prob, 0),
            )
        )

        return first_term + remaining_terms

    def _evidence_predict_step(self, log_smoothing_distribution):
        """Predict step for calculating the evidence.

        .. math::
            p(s_t=j | x_{1:t-1}) = \displaystyle\sum_i p(s_t = j | s_{t-1} = i) p(s_{t-1} = i | x_{1:t-1})

        Parameters
        ----------
        log_smoothing_distribution : np.ndarray
            log p(s_t-1 | x_1:t-1). Shape is (batch_size, n_states).

        Returns
        -------
        log_prediction_distribution : np.ndarray
            log p(s_t | x_1:t-1). Shape is (batch_size, n_states).
        """
        log_trans_prob = np.expand_dims(np.log(self.trans_prob), 0)
        log_smoothing_distribution = np.expand_dims(log_smoothing_distribution, -1)
        log_prediction_distribution = logsumexp(
            log_trans_prob + log_smoothing_distribution, -2
        )
        return log_prediction_distribution

    def _get_log_likelihood(self, data, subj_id):
        """Get the log-likelihood of data, log p(x_t | s_t).

        Parameters
        ----------
        data : np.ndarray
            Data. Shape is (batch_size, ..., n_channels).
        subj_id : np.ndarray
            Subject ID. Shape is (batch_size, ...).

        Returns
        -------
        log_likelihood : np.ndarray
            Log-likelihood. Shape is (batch_size, ..., n_states)
        """
        means, covs = self.get_subject_means_covs()
        subj_id = tf.cast(subj_id, tf.int32)
        means = tf.gather(means, subj_id)
        covs = tf.gather(covs, subj_id)

        mvn = tfp.distributions.MultivariateNormalTriL(
            loc=means,
            scale_tril=tf.linalg.cholesky(covs),
            allow_nan_stats=False,
        )
        log_likelihood = mvn.log_prob(tf.expand_dims(data, axis=-2))
        return log_likelihood.numpy()

    def _evidence_update_step(self, data, log_prediction_distribution, subj_id):
        """Update step for calculating the evidence.

        .. math::
            p(s_t = j | x_{1:t}) &= \displaystyle\\frac{p(x_t | s_t = j) p(s_t = j | x_{1:t-1})}{p(x_t | x_{1:t-1})}

            p(x_t | x_{1:t-1}) &= \displaystyle\sum_i p(x_t | s_t = j) p(s_t = i | x_{1:t-1})

        Parameters
        ----------
        data : np.ndarray
            Data for the update step. Shape is (batch_size, n_channels).
        log_prediction_distribution : np.ndarray
            log p(s_t | x_1:t-1). Shape is (batch_size, n_states).
        subj_id : np.ndarray
            Subject ID. Shape is (batch_size,).

        Returns
        -------
        log_smoothing_distribution : np.ndarray
            log p(s_t | x_1:t). Shape is (batch_size, n_states).
        predictive_log_likelihood : np.ndarray
            log p(x_t | x_1:t-1). Shape is (batch_size).
        """
        log_likelihood = self._get_log_likelihood(data, subj_id)
        log_smoothing_distribution = log_likelihood + log_prediction_distribution
        predictive_log_likelihood = logsumexp(log_smoothing_distribution, -1)

        # Normalise the log smoothing distribution
        log_smoothing_distribution -= np.expand_dims(predictive_log_likelihood, -1)
        return log_smoothing_distribution, predictive_log_likelihood

    def get_stationary_distribution(self):
        """Get the stationary distribution of the Markov chain.

        This is the left eigenvector of the transition probability matrix
        corresponding to eigenvalue = 1.

        Returns
        -------
        stationary_distribution : np.ndarray
            Stationary distribution of the Markov chain. Shape is (n_states,).
        """
        eigval, eigvec = np.linalg.eig(self.trans_prob.T)
        stationary_distribution = np.squeeze(eigvec[:, np.isclose(eigval, 1)]).real
        stationary_distribution /= np.sum(stationary_distribution)
        return stationary_distribution

    def sample_state_time_course(self, n_samples):
        """Sample a state time course.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        stc : np.ndarray
            State time course with shape (n_samples, n_states).
        """
        sim = HMM(self.trans_prob)
        stc = sim.generate_states(n_samples)
        return stc

    def get_trans_prob(self):
        """Get the transition probability matrix.

        Returns
        -------
        trans_prob : np.ndarray
            Transition probability matrix. Shape is (n_states, n_states).
        """
        return self.trans_prob

    def get_group_covariances(self):
        """Get the covariances of each state.

        Returns
        -------
        covariances : np.ndarray
            State covariances. Shape is (n_states, n_channels, n_channels).
        """
        return sedynemo_obs.get_group_means_covs(self.model)[1]

    def get_group_means_covariances(self):
        """Get the means and covariances of each state.

        Returns
        -------
        means : np.ndarary
            Group level state means.
        covariances : np.ndarray
            Group level state covariances.
        """
        return sedynemo_obs.get_group_means_covs(self.model)

    def get_subject_means_covs(self, subject_embeddings=None, n_neighbours=2):
        """Get the subject means and covariances.

        Parameters
        ----------
        subject_embeddings : np.ndarray
            Input embedding vectors for subjects. Shape is (n_subjects, subject_embeddings_dim).
        n_neighbours : int
            Number of nearest neighbours. Ignored if subject_embedding=None.

        Returns
        -------
        means : np.ndarray
            Subject means. Shape is (n_subjects, n_states, n_channels).
        covs : np.ndarray
            Subject covariances. Shape is (n_subjects, n_states, n_channels, n_channels).
        """
        return sedynemo_obs.get_subject_means_covs(
            self.model,
            self.config.learn_means,
            self.config.learn_covariances,
            subject_embeddings,
            n_neighbours,
        )

    def get_subject_embeddings(self):
        """Get the subject embedding vectors

        Returns
        -------
        subject_embeddings : np.ndarray
            Embedding vectors for subjects.
            Shape is (n_subjects, subject_embedding_dim).
        """
        return sedynemo_obs.get_subject_embeddings(self.model)

    def set_group_means(self, group_means, update_initializer=True):
        """Set the group means of each state.

        Parameters
        ----------
        group_means : np.ndarray
            State covariances.
        update_initializer : bool
            Do we want to use the passed means when we re-initialize
            the model?
        """
        dynemo_obs.set_means(
            self.model, group_means, update_initializer, layer_name="group_means"
        )

    def set_group_covariances(self, group_covariances, update_initializer=True):
        """Set the group covariances of each state.

        Parameters
        ----------
        group_covariances : np.ndarray
            State covariances.
        update_initializer : bool
            Do we want to use the passed covariances when we re-initialize
            the model?
        """
        dynemo_obs.set_covariances(
            self.model,
            group_covariances,
            update_initializer=update_initializer,
            layer_name="group_covs",
        )

    def set_trans_prob(self, trans_prob):
        """Sets the transition probability matrix.

        Parameters
        ----------
        trans_prob : np.ndarray
            State transition probabilities. Shape must be (n_states, n_states).
        """
        if trans_prob is None:
            trans_prob = (
                np.ones((self.config.n_states, self.config.n_states))
                * 0.1
                / (self.config.n_states - 1)
            )
            np.fill_diagonal(trans_prob, 0.9)
        self.trans_prob = trans_prob

    def set_state_probs_t0(self, state_probs_t0):
        """Set the initial state probabilities.

        Parameters
        ----------
        state_probs_t0 : np.ndarray
            Initial state probabilities. Shape is (n_states,).
        """

        if state_probs_t0 is None:
            state_probs_t0 = np.ones((self.config.n_states,)) / self.config.n_states
        self.state_probs_t0 = state_probs_t0

    def set_random_state_time_course_initialization(self, training_data):
        """Sets the initial means/covariances based on a random state time course.

        Parameters
        ----------
        training_data : tensorflow.data.Dataset or osl_dynamics.data.Data
            Training data.
        """
        _logger.info("Setting random means and covariances")

        # Make a TensorFlow Dataset
        training_dataset = self.make_dataset(training_data, concatenate=True)

        # Mean and covariance for each state
        means = np.zeros(
            [self.config.n_states, self.config.n_channels], dtype=np.float32
        )
        covariances = np.zeros(
            [self.config.n_states, self.config.n_channels, self.config.n_channels],
            dtype=np.float32,
        )

        for batch in training_dataset:
            # Concatenate all the sequences in this batch
            data = np.concatenate(batch["data"])

            # Sample a state time course using the initial transition
            # probability matrix
            stc = self.sample_state_time_course(data.shape[0])

            # Calculate the mean/covariance for each state for this batch
            m = []
            C = []
            for j in range(self.config.n_states):
                x = data[stc[:, j] == 1]
                mu_j = np.mean(x, axis=0)
                sigma_j = np.cov(x, rowvar=False)
                m.append(mu_j)
                C.append(sigma_j)
            means += m
            covariances += C

        # Calculate the average from the running total
        n_batches = dtf.get_n_batches(training_dataset)
        means /= n_batches
        covariances /= n_batches

        if self.config.learn_means:
            # Set initial means
            self.set_group_means(means, update_initializer=True)

        if self.config.learn_covariances:
            # Set initial covariances
            self.set_group_covariances(covariances, update_initializer=True)

    def set_regularizers(self, training_dataset):
        """Set the means and covariances regularizer based on the training data.

        A multivariate normal prior is applied to the mean vectors with mu = 0,
        sigma=diag((range / 2)**2). If config.diagonal_covariances is True, a log
        normal prior is applied to the diagonal of the covariances matrices with mu=0,
        sigma=sqrt(log(2 * (range))), otherwise an inverse Wishart prior is applied
        to the covariances matrices with nu=n_channels - 1 + 0.1 and psi=diag(1 / range).

        Parameters
        ----------
        training_dataset : tensorflow.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        """
        training_dataset = self.make_dataset(training_dataset, concatenate=True)

        if self.config.learn_means:
            dynemo_obs.set_means_regularizer(
                self.model, training_dataset, layer_name="group_means"
            )

        if self.config.learn_covariances:
            dynemo_obs.set_covariances_regularizer(
                self.model,
                training_dataset,
                self.config.covariances_epsilon,
                layer_name="group_covs",
            )

    def set_bayesian_kl_scaling(self, training_dataset):
        """Set the correct scaling for KL loss between deviation posterior and prior.

        Parameters
        ----------
        training_dataset : tensorflow.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        """
        training_dataset = self.make_dataset(training_dataset, concatenate=True)
        n_batches = dtf.get_n_batches(training_dataset)
        learn_means = self.config.learn_means
        learn_covariances = self.config.learn_covariances
        sedynemo_obs.set_bayesian_kl_scaling(
            self.model, n_batches, learn_means, learn_covariances
        )

    def free_energy(self, dataset):
        """Get the variational free energy.

        This calculates:

        .. math::
            \mathcal{F} = \int q(s_{1:T}) \log \left[ \\frac{q(s_{1:T})}{p(x_{1:T}, s_{1:T})} \\right] ds_{1:T}

        Parameters
        ----------
        dataset : tensorflow.data.Dataset or osl_dynamics.data.Data
            Dataset to evaluate the free energy for.

        Returns
        -------
        free_energy : float
            Variational free energy.
        """
        _logger.info("Getting free energy")

        # Convert to a TensorFlow dataset if not already
        dataset = self.make_dataset(dataset, concatenate=True, subj_id=True)

        # Calculate variational free energy for each batch
        free_energy = []
        for data in dataset:
            x = data["data"]
            subj_id = data["subj_id"]
            batch_size = x.shape[0]

            # Get the marginal and join posterior to calculate the free energy
            gamma, xi = self._get_state_probs(x, subj_id)

            # Calculate the free energy:
            #
            # F = int q(s) log[q(s) / p(x, s)] ds
            #   = int q(s) log[q(s) / p(x | s) p(s)] ds
            #   = - int q(s) log p(x | s) ds    [log_likelihood]
            #     + int q(s) log q(s) ds        [entropy]
            #     - int q(s) log p(s) ds        [prior]

            log_likelihood = self._get_posterior_expected_log_likelihood(
                x, gamma, subj_id
            )
            entropy = self._get_posterior_entropy(gamma, xi)
            prior = self._get_posterior_expected_prior(gamma, xi)

            # Average free energy for a sequence in this batch
            seq_fe = (-log_likelihood + entropy - prior) / batch_size
            free_energy.append(seq_fe)

        # Return average over batches
        return np.mean(free_energy)

    def evidence(self, dataset):
        """Calculate the model evidence, p(x), of HMM on a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset or osl_dynamics.data.Data
            Dataset to evaluate the model evidence on.

        Returns
        -------
        evidence : float
            Model evidence.
        """
        _logger.info("Getting model evidence")
        dataset = self.make_dataset(dataset, concatenate=True, subj_id=True)
        n_batches = dtf.get_n_batches(dataset)

        evidence = 0
        for n, data in enumerate(dataset):
            x = data["data"]
            subj_id = data["subj_id"]
            print("Batch {}/{}".format(n + 1, n_batches))
            pb_i = utils.Progbar(self.config.sequence_length)
            batch_size = tf.shape(x)[0]
            batch_evidence = np.zeros((batch_size))
            for t in range(self.config.sequence_length):
                # Prediction step
                if t == 0:
                    initial_distribution = self.get_stationary_distribution()
                    log_prediction_distribution = np.broadcast_to(
                        np.expand_dims(initial_distribution, axis=0),
                        (batch_size, self.config.n_states),
                    )
                else:
                    log_prediction_distribution = self._evidence_predict_step(
                        log_smoothing_distribution
                    )

                # Update step
                (
                    log_smoothing_distribution,
                    predictive_log_likelihood,
                ) = self._evidence_update_step(
                    x[:, t, :], log_prediction_distribution, subj_id[:, t]
                )

                # Update the batch evidence
                batch_evidence += predictive_log_likelihood
                pb_i.add(1)
            evidence += np.mean(batch_evidence)

        return evidence / n_batches

    def get_alpha(self, dataset, concatenate=False):
        """Get state probabilities.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset or osl_dynamics.data.Data
            Prediction dataset. This can be a list of datasets, one for each subject.
        concatenate : bool
            Should we concatenate alpha for each subject?

        Returns
        -------
        alpha : list or np.ndarray
            State probabilities with shape (n_subjects, n_samples, n_states)
            or (n_samples, n_states).
        """
        dataset = self.make_dataset(dataset, subj_id=True)

        _logger.info("Getting alpha")
        alpha = []
        for ds in dataset:
            gamma = []
            for data in ds:
                x = data["data"]
                subj_id = data["subj_id"]
                g, _ = self._get_state_probs(x, subj_id)
                gamma.append(g)
            alpha.append(np.concatenate(gamma).astype(np.float32))

        if concatenate or len(alpha) == 1:
            alpha = np.concatenate(alpha)

        return alpha

    def get_training_time_series(self, training_data, prepared=True, concatenate=False):
        """Get the time series used for training from a Data object.

        Parameters
        ----------
        training_data : osl_dynamics.data.Data
            Data object.
        prepared : bool
            Should we return the prepared data? If not, we return the raw data.
        concatenate : bool
            Should we concatenate the data for each subject?

        Returns
        -------
        training_data : np.ndarray or list
            Training data time series.
        """
        return training_data.trim_time_series(
            self.config.sequence_length, prepared=prepared, concatenate=concatenate
        )

    def save_weights(self, filepath):
        """Save all model weights.

        Parameters
        ----------
        filepath : str
            Location to save model weights to.
        """
        self.model.save_weights(filepath)
        np.save(op.join(str(Path(filepath).parent), "trans_prob.npy"), self.trans_prob)

    def load_weights(self, filepath):
        """Load all model parameters.

        Parameters
        ----------
        filepath : str
            Location to load model weights from.
        """
        self.trans_prob = np.load(op.join(str(Path(filepath).parent), "trans_prob.npy"))
        return self.model.load_weights(filepath)

    def set_weights(self, weights, trans_prob):
        """Set model parameter weights.

        Parameters
        ----------
        weights : tensorflow weights
            TensorFlow weights for the observation model.
        trans_prob : np.ndarray
            Transition probability matrix.
        """
        self.model.set_weights(weights)
        self.set_trans_prob(trans_prob)

    def reset_weights(self):
        """Resets trainable variables in the model to their initial value."""
        inference.initializers.reinitialize_model_weights(self.model)
        self.set_trans_prob(self.config.initial_trans_prob)


def _model_structure(config):
    # Inputs
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels + config.n_states + 1),
        name="input",
    )
    data, gamma, subj_id = tf.split(
        inputs, [config.n_channels, config.n_states, 1], axis=2
    )
    subj_id = tf.squeeze(subj_id, axis=2)

    # Subject embedding layers
    subjects_layer = TFRangeLayer(config.n_subjects, name="subjects")
    subject_embeddings_layer = layers.Embedding(
        config.n_subjects, config.subject_embeddings_dim, name="subject_embeddings"
    )

    # Group level observation model parameters
    group_means_layer = VectorsLayer(
        config.n_states,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        config.means_regularizer,
        name="group_means",
    )
    group_covs_layer = CovarianceMatricesLayer(
        config.n_states,
        config.n_channels,
        config.learn_covariances,
        config.initial_covariances,
        config.covariances_epsilon,
        config.covariances_regularizer,
        name="group_covs",
    )

    subjects = subjects_layer(data)
    subject_embeddings = subject_embeddings_layer(subjects)

    group_mu = group_means_layer(data)
    group_D = group_covs_layer(data)

    # ---------------
    # Mean deviations

    # Layer definitions
    if config.learn_means:
        means_mode_embeddings_layer = layers.Dense(
            config.mode_embeddings_dim,
            name="means_mode_embeddings",
        )
        means_concat_embeddings_layer = ConcatEmbeddingsLayer(
            name="means_concat_embeddings",
        )

        means_dev_map_input_layer = MultiLayerPerceptronLayer(
            config.dev_n_layers,
            config.dev_n_units,
            config.dev_normalization,
            config.dev_activation,
            config.dev_dropout,
            name="means_dev_map_input",
        )
        means_dev_map_layer = layers.Dense(config.n_channels, name="means_dev_map")
        norm_means_dev_map_layer = layers.LayerNormalization(
            axis=-1, scale=False, name="norm_means_dev_map"
        )

        means_dev_mag_inf_alpha_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_states, 1),
            learn=config.learn_means,
            initializer=initializers.TruncatedNormal(mean=0, stddev=0.02),
            name="means_dev_mag_inf_alpha_input",
        )
        means_dev_mag_inf_alpha_layer = layers.Activation(
            "softplus", name="means_dev_mag_inf_alpha"
        )
        means_dev_mag_inf_beta_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_states, 1),
            learn=config.learn_means,
            initializer=initializers.TruncatedNormal(mean=50, stddev=0.02),
            name="means_dev_mag_inf_beta_input",
        )
        means_dev_mag_inf_beta_layer = layers.Activation(
            "softplus", name="means_dev_mag_inf_beta"
        )
        means_dev_mag_layer = SampleGammaDistributionLayer(
            config.covariances_epsilon, name="means_dev_mag"
        )

        means_dev_layer = layers.Multiply(name="means_dev")

        # Data flow to get the subject specific deviations of means

        # Get the concatenated embeddings
        means_mode_embeddings = means_mode_embeddings_layer(group_mu)
        means_concat_embeddings = means_concat_embeddings_layer(
            [subject_embeddings, means_mode_embeddings]
        )

        # Get the mean deviation maps (no global magnitude information)
        means_dev_map_input = means_dev_map_input_layer(means_concat_embeddings)
        means_dev_map = means_dev_map_layer(means_dev_map_input)
        norm_means_dev_map = norm_means_dev_map_layer(means_dev_map)

        # Get the deviation magnitudes (scale deviation maps globally)

        means_dev_mag_inf_alpha_input = means_dev_mag_inf_alpha_input_layer(data)
        means_dev_mag_inf_alpha = means_dev_mag_inf_alpha_layer(
            means_dev_mag_inf_alpha_input
        )
        means_dev_mag_inf_beta_input = means_dev_mag_inf_beta_input_layer(data)
        means_dev_mag_inf_beta = means_dev_mag_inf_beta_layer(
            means_dev_mag_inf_beta_input
        )
        means_dev_mag = means_dev_mag_layer(
            [means_dev_mag_inf_alpha, means_dev_mag_inf_beta]
        )
        means_dev = means_dev_layer([means_dev_mag, norm_means_dev_map])
    else:
        means_dev_layer = ZeroLayer(
            shape=(config.n_subjects, config.n_states, config.n_channels),
            name="means_dev",
        )
        means_dev = means_dev_layer(data)

    # ----------------------
    # Covariances deviations

    # Layer definitions
    if config.learn_covariances:
        covs_mode_embeddings_layer = layers.Dense(
            config.mode_embeddings_dim,
            name="covs_mode_embeddings",
        )
        covs_concat_embeddings_layer = ConcatEmbeddingsLayer(
            name="covs_concat_embeddings",
        )

        covs_dev_map_input_layer = MultiLayerPerceptronLayer(
            config.dev_n_layers,
            config.dev_n_units,
            config.dev_normalization,
            config.dev_activation,
            config.dev_dropout,
            name="covs_dev_map_input",
        )
        covs_dev_map_layer = layers.Dense(
            config.n_channels * (config.n_channels + 1) // 2, name="covs_dev_map"
        )
        norm_covs_dev_map_layer = layers.LayerNormalization(
            axis=-1, scale=False, name="norm_covs_dev_map"
        )

        covs_dev_mag_inf_alpha_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_states, 1),
            learn=config.learn_covariances,
            initializer=initializers.TruncatedNormal(mean=0, stddev=0.02),
            name="covs_dev_mag_inf_alpha_input",
        )
        covs_dev_mag_inf_alpha_layer = layers.Activation(
            "softplus", name="covs_dev_mag_inf_alpha"
        )
        covs_dev_mag_inf_beta_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_states, 1),
            learn=config.learn_covariances,
            initializer=initializers.TruncatedNormal(mean=50, stddev=0.02),
            name="covs_dev_mag_inf_beta_input",
        )
        covs_dev_mag_inf_beta_layer = layers.Activation(
            "softplus", name="covs_dev_mag_inf_beta"
        )
        covs_dev_mag_layer = SampleGammaDistributionLayer(
            config.covariances_epsilon, name="covs_dev_mag"
        )
        covs_dev_layer = layers.Multiply(name="covs_dev")

        # Data flow to get subject specific deviations of covariances

        # Get the concatenated embeddings
        covs_mode_embeddings = covs_mode_embeddings_layer(
            InverseCholeskyLayer(config.covariances_epsilon)(group_D)
        )
        covs_concat_embeddings = covs_concat_embeddings_layer(
            [subject_embeddings, covs_mode_embeddings]
        )

        # Get the covariance deviation maps (no global magnitude information)
        covs_dev_map_input = covs_dev_map_input_layer(covs_concat_embeddings)
        covs_dev_map = covs_dev_map_layer(covs_dev_map_input)
        norm_covs_dev_map = norm_covs_dev_map_layer(covs_dev_map)

        # Get the deviation magnitudes (scale deviation maps globally)
        covs_dev_mag_inf_alpha_input = covs_dev_mag_inf_alpha_input_layer(data)
        covs_dev_mag_inf_alpha = covs_dev_mag_inf_alpha_layer(
            covs_dev_mag_inf_alpha_input
        )
        covs_dev_mag_inf_beta_input = covs_dev_mag_inf_beta_input_layer(data)
        covs_dev_mag_inf_beta = covs_dev_mag_inf_beta_layer(covs_dev_mag_inf_beta_input)
        covs_dev_mag = covs_dev_mag_layer(
            [covs_dev_mag_inf_alpha, covs_dev_mag_inf_beta]
        )
        covs_dev = covs_dev_layer([covs_dev_mag, norm_covs_dev_map])
    else:
        covs_dev_layer = ZeroLayer(
            shape=(
                config.n_subjects,
                config.n_states,
                config.n_channels * (config.n_channels + 1) // 2,
            ),
            name="covs_dev",
        )
        covs_dev = covs_dev_layer(data)

    # ----------------------------------------
    # Add deviations to group level parameters

    # Layer definitions
    subject_means_layer = SubjectMapLayer(
        "means", config.covariances_epsilon, name="subject_means"
    )
    subject_covs_layer = SubjectMapLayer(
        "covariances", config.covariances_epsilon, name="subject_covs"
    )

    # Data flow
    mu = subject_means_layer([group_mu, means_dev])
    D = subject_covs_layer([group_D, covs_dev])

    # -----------------------------------
    # Mix the subject specific paraemters
    # and get the conditional likelihood

    # Layer definitions
    ll_loss_layer = CategoricalLogLikelihoodLossLayer(
        config.n_states, config.covariances_epsilon, name="ll_loss"
    )

    # Data flow
    ll_loss = ll_loss_layer([data, mu, D, gamma, subj_id])

    # ---------
    # KL losses

    # For the observation model (static KL loss)
    if config.learn_means:
        # Layer definitions
        means_dev_mag_mod_beta_input_layer = MultiLayerPerceptronLayer(
            config.dev_n_layers,
            config.dev_n_units,
            config.dev_normalization,
            config.dev_activation,
            config.dev_dropout,
            name="means_dev_mag_mod_beta_input",
        )
        means_dev_mag_mod_beta_layer = layers.Dense(
            1,
            activation="softplus",
            name="means_dev_mag_mod_beta",
        )

        means_dev_mag_kl_loss_layer = StaticKLDivergenceLayer(
            config.covariances_epsilon, name="means_dev_mag_kl_loss"
        )

        # Data flow
        means_dev_mag_mod_beta_input = means_dev_mag_mod_beta_input_layer(
            means_concat_embeddings
        )
        means_dev_mag_mod_beta = means_dev_mag_mod_beta_layer(
            means_dev_mag_mod_beta_input
        )
        means_dev_mag_kl_loss = means_dev_mag_kl_loss_layer(
            [
                data,
                means_dev_mag_inf_alpha,
                means_dev_mag_inf_beta,
                means_dev_mag_mod_beta,
            ]
        )
    else:
        means_dev_mag_kl_loss_layer = ZeroLayer((), name="means_dev_mag_kl_loss")
        means_dev_mag_kl_loss = means_dev_mag_kl_loss_layer(data)

    if config.learn_covariances:
        # Layer definitions
        covs_dev_mag_mod_beta_input_layer = MultiLayerPerceptronLayer(
            config.dev_n_layers,
            config.dev_n_units,
            config.dev_normalization,
            config.dev_activation,
            config.dev_dropout,
            name="covs_dev_mag_mod_beta_input",
        )
        covs_dev_mag_mod_beta_layer = layers.Dense(
            1,
            activation="softplus",
            name="covs_dev_mag_mod_beta",
        )

        covs_dev_mag_kl_loss_layer = StaticKLDivergenceLayer(
            config.covariances_epsilon, name="covs_dev_mag_kl_loss"
        )

        # Data flow
        covs_dev_mag_mod_beta_input = covs_dev_mag_mod_beta_input_layer(
            covs_concat_embeddings
        )
        covs_dev_mag_mod_beta = covs_dev_mag_mod_beta_layer(covs_dev_mag_mod_beta_input)
        covs_dev_mag_kl_loss = covs_dev_mag_kl_loss_layer(
            [
                data,
                covs_dev_mag_inf_alpha,
                covs_dev_mag_inf_beta,
                covs_dev_mag_mod_beta,
            ]
        )
    else:
        covs_dev_mag_kl_loss_layer = ZeroLayer((), name="covs_dev_mag_kl_loss")
        covs_dev_mag_kl_loss = covs_dev_mag_kl_loss_layer(data)

    # Total KL loss
    # Layer definitions
    kl_loss_layer = KLLossLayer(do_annealing=False, name="kl_loss")

    # Data flow
    kl_loss = kl_loss_layer([means_dev_mag_kl_loss, covs_dev_mag_kl_loss])

    return tf.keras.Model(inputs=inputs, outputs=[ll_loss, kl_loss], name="SE-HMM-Obs")
