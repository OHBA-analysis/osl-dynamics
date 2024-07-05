"""Hidden Markov Model (HMM) with a Multivariate Normal observation model.

See the `documentation <https://osl-dynamics.readthedocs.io/en/latest/models\
/hmm.html>`_ for a description of this model.

See Also
--------
- D. Vidaurre, et al., "Spectrally resolved fast transient brain states in
  electrophysiological data". `Neuroimage 126, 81-95 (2016)
  <https://www.sciencedirect.com/science/article/pii/S1053811915010691>`_.
- D. Vidaurre, et al., "Discovering dynamic brain networks from big data in
  rest and task". `Neuroimage 180, 646-656 (2018)
  <https://www.sciencedirect.com/science/article/pii/S1053811917305487>`_.
- `MATLAB HMM-MAR Toolbox <https://github.com/OHBA-analysis/HMM-MAR>`_.
"""

import logging
import os
import os.path as op
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend, layers, utils
from numba.core.errors import NumbaWarning
from scipy.special import logsumexp, xlogy
from tqdm.auto import trange
from pqdm.threads import pqdm

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference import initializers
from osl_dynamics.inference.layers import (
    CategoricalLogLikelihoodLossLayer,
    CovarianceMatricesLayer,
    DiagonalMatricesLayer,
    VectorsLayer,
    StaticLossScalingFactorLayer,
)
from osl_dynamics.models import obs_mod
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.simulation import HMM
from osl_dynamics.utils.misc import set_logging_level
from osl_dynamics import array_ops

_logger = logging.getLogger("osl-dynamics")

warnings.filterwarnings("ignore", category=NumbaWarning)
logging.getLogger("numba.core.transforms").setLevel(logging.ERROR)

EPS = sys.float_info.epsilon


@dataclass
class Config(BaseModelConfig):
    """Settings for HMM.

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

    learn_means : bool
        Should we make the mean vectors for each state trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each staet trainable?
    initial_means : np.ndarray
        Initialisation for state means.
    initial_covariances : np.ndarray
        Initialisation for state covariances. If
        :code:`diagonal_covariances=True` and full matrices are passed,
        the diagonal is extracted.
    diagonal_covariances : bool
        Should we learn diagonal covariances?
    covariances_epsilon : float
        Error added to state covariances for numerical stability.

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

    model_name: str = "HMM"

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None
    diagonal_covariances: bool = False
    covariances_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

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
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")

        if self.covariances_epsilon is None:
            if self.learn_covariances:
                self.covariances_epsilon = 1e-6
            else:
                self.covariances_epsilon = 0.0

    def validate_trans_prob_parameters(self):
        if self.initial_trans_prob is not None:
            if (
                not isinstance(self.initial_trans_prob, np.ndarray)
                or self.initial_trans_prob.ndim != 2
            ):
                raise ValueError("initial_trans_prob must be a 2D numpy array.")

            if not all(np.isclose(np.sum(self.initial_trans_prob, axis=1), 1)):
                raise ValueError("rows of initial_trans_prob must sum to one.")


class Model(ModelBase):
    """HMM class.

    Parameters
    ----------
    config : osl_dynamics.models.hmm.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""
        self.model = self._model_structure()

        self.rho = 1
        self.set_trans_prob(self.config.initial_trans_prob)
        self.set_state_probs_t0(self.config.state_probs_t0)

    def fit(self, dataset, epochs=None, use_tqdm=False, verbose=1, **kwargs):
        """Fit model to a dataset.

        Iterates between:

        - Baum-Welch updates of latent variable time courses and transition
          probability matrix.
        - TensorFlow updates of observation model parameters.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        epochs : int, optional
            Number of epochs.
        use_tqdm : bool, optional
            Should we use :code:`tqdm` to display a progress bar?
        verbose : int, optional
            Verbosity level. :code:`0=silent`.
        kwargs : keyword arguments, optional
            Keyword arguments for the TensorFlow observation model training.
            These keywords arguments will be passed to :code:`self.model.fit()`.

        Returns
        -------
        history : dict
            Dictionary with history of the loss and learning rates (:code:`lr`
            and :code:`rho`).
        """
        if epochs is None:
            epochs = self.config.n_epochs

        # Make a TensorFlow Dataset
        dataset = self.make_dataset(dataset, shuffle=True, concatenate=True)

        # Set static loss scaling factor
        self.set_static_loss_scaling_factor(dataset)

        # Training curves
        history = {"loss": [], "rho": [], "lr": []}

        # Loop through epochs
        if use_tqdm:
            _range = trange(epochs)
        else:
            _range = range(epochs)
        for n in _range:
            # Setup a progress bar for this epoch
            if verbose > 0 and not use_tqdm:
                print("Epoch {}/{}".format(n + 1, epochs))
                pb_i = utils.Progbar(dtf.get_n_batches(dataset))

            # Update rho
            self._update_rho(n)

            # Set learning rate for the observation model
            lr = self.config.learning_rate * np.exp(
                -self.config.observation_update_decay * n
            )
            backend.set_value(self.model.optimizer.lr, lr)

            # Loop over batches
            loss = []
            for data in dataset:
                x = data["data"]

                # Update state probabilities
                gamma, xi = self.get_posterior(x)

                # Update transition probability matrix
                if self.config.learn_trans_prob:
                    self.update_trans_prob(gamma, xi)

                # Reshape gamma: (batch_size*sequence_length, n_states)
                # -> (batch_size, sequence_length, n_states)
                gamma = gamma.reshape(x.shape[0], x.shape[1], -1)

                # Update observation model
                x_and_gamma = np.concatenate([x, gamma], axis=2)
                h = self.model.fit(x_and_gamma, epochs=1, verbose=0, **kwargs)

                # Get new loss
                l = h.history["loss"][0]
                if np.isnan(l):
                    _logger.error("Training failed!")
                    return
                loss.append(l)

                if verbose > 0:
                    # Update progress bar
                    if use_tqdm:
                        _range.set_postfix(rho=self.rho, lr=lr, loss=l)
                    else:
                        pb_i.add(
                            1,
                            values=[("rho", self.rho), ("lr", lr), ("loss", l)],
                        )

            history["loss"].append(np.mean(loss))
            history["rho"].append(self.rho)
            history["lr"].append(lr)

        if use_tqdm:
            _range.close()

        return history

    def set_static_loss_scaling_factor(self, dataset):
        """Set the :code:`n_batches` attribute of the
        :code:`"static_loss_scaling_factor"` layer.

        Parameters
        ----------
        dataset : tf.data.Dataset
            TensorFlow dataset.
        """
        layer_names = [layer.name for layer in self.model.layers]
        if "static_loss_scaling_factor" in layer_names:
            n_batches = dtf.get_n_batches(dataset)
            self.model.get_layer("static_loss_scaling_factor").n_batches = n_batches

    def random_subset_initialization(
        self, training_data, n_epochs, n_init, take, **kwargs
    ):
        """Random subset initialization.

        The model is trained for a few epochs with different random subsets
        of the training dataset. The model with the best free energy is kept.

        Parameters
        ----------
        training_data : tf.data.Dataset or osl_dynamics.data.Data
            Dataset to use for training.
        n_epochs : int
            Number of epochs to train the model.
        n_init : int
            Number of initializations.
        take : float
            Fraction of total batches to take.
        kwargs : keyword arguments, optional
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
        training_dataset = self.make_dataset(
            training_data, shuffle=True, concatenate=True
        )

        # Calculate the number of batches to use
        if take < 1:
            n_total_batches = dtf.get_n_batches(training_dataset)
            n_batches = max(round(n_total_batches * take), 1)
            _logger.info(f"Using {n_batches} out of {n_total_batches} batches")

        # Pick the initialization with the lowest free energy
        best_loss = np.Inf
        for n in range(n_init):
            _logger.info(f"Initialization {n}")
            self.reset()
            if take < 1:
                training_data_subset = training_dataset.take(n_batches)
            else:
                training_data_subset = training_dataset
            history = self.fit(training_data_subset, epochs=n_epochs, **kwargs)
            if history is None:
                continue
            loss = history["loss"][-1]
            if loss < best_loss:
                best_initialization = n
                best_loss = loss
                best_history = history
                best_weights, best_trans_prob = self.get_weights()

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
        training_data : tf.data.Dataset or osl_dynamics.data.Data
            Dataset to use for training.
        n_epochs : int
            Number of epochs to train the model.
        n_init : int
            Number of initializations.
        take : float, optional
            Fraction of total batches to take.
        kwargs : keyword arguments, optional
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
            training_data, shuffle=True, concatenate=True
        )

        # Calculate the number of batches to use
        if take < 1:
            n_total_batches = dtf.get_n_batches(training_dataset)
            n_batches = max(round(n_total_batches * take), 1)
            _logger.info(f"Using {n_batches} out of {n_total_batches} batches")

        # Pick the initialization with the lowest free energy
        best_loss = np.Inf
        for n in range(n_init):
            _logger.info(f"Initialization {n}")
            self.reset()
            if take < 1:
                training_data_subset = training_dataset.take(n_batches)
            else:
                training_data_subset = training_dataset
            self.set_random_state_time_course_initialization(training_data_subset)
            history = self.fit(training_data_subset, epochs=n_epochs, **kwargs)
            if history is None:
                continue
            loss = history["loss"][-1]
            if loss < best_loss:
                best_initialization = n
                best_loss = loss
                best_history = history
                best_weights, best_trans_prob = self.get_weights()

        if best_loss == np.Inf:
            _logger.error("Initialization failed")
            return

        _logger.info(f"Using initialization {best_initialization}")
        self.set_weights(best_weights, best_trans_prob)

        return best_history

    def get_posterior(self, x):
        """Get marginal and joint posterior.

        Parameters
        ----------
        x : np.ndarray
            Observed data. Shape is (batch_size, sequence_length, n_channels).

        Returns
        -------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data,
            :math:`q(s_t)`. Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive
            time points, :math:`q(s_t, s_{t+1})`. Shape is
            (batch_size*sequence_length-1, n_states*n_states).
        """
        B = self.get_likelihood(x)
        Pi_0 = self.state_probs_t0
        P = self.trans_prob
        return self.baum_welch(B, Pi_0, P)

    @numba.jit
    def baum_welch(self, B, Pi_0, P):
        """Hidden state inference using the Baum-Welch algorithm.

        Parameters
        ----------
        B : np.ndarray
            Probability of array data points, under observation model for
            each state. Shape is (n_states, n_samples).
        Pi_0 : np.ndarray
            Initial state probabilities. Shape is (n_states,).
        P : np.ndarray
            State transition probabilities. Shape is (n_states, n_states).

        Returns
        -------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data,
            :math:`q(s_t)`. Shape is (n_samples, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive
            time points, :math:`q(s_t, s_{t+1})`. Shape is (n_samples-1,
            n_states*n_states).
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
        means, covs = self.get_means_covariances()
        n_states = means.shape[0]

        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        # Calculate the log-likelihood for each state to have generated the
        # observed data
        log_likelihood = np.empty([n_states, batch_size, sequence_length])
        for state in range(n_states):
            mvn = tf.stop_gradient(
                tfp.distributions.MultivariateNormalTriL(
                    loc=tf.gather(means, state, axis=-2),
                    scale_tril=tf.linalg.cholesky(tf.gather(covs, state, axis=-3)),
                    allow_nan_stats=False,
                )
            )
            log_likelihood[state] = mvn.log_prob(x)
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

    def update_trans_prob(self, gamma, xi):
        """Update transition probability matrix.

        Parameters
        ----------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data,
            :math:`q(s_t)`. Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive
            time points, :math:`q(s_t, s_{t+1})`. Shape is
            (batch_size*sequence_length-1, n_states*n_states).
        """
        # Calculate the new transition probability matrix using the posterior
        # from the Baum-Welch algorithm:
        #
        # p(s_t+1 | s_t) = E{q(s_t, s_t+1)} / E{q(s_t)}
        #                = sum^{T-1}_{t=1} xi(t, t+1) / sum^{T-1}_{t=1} gamma(t)
        #
        # where E{.} denotes the expectation.
        phi_interim = np.sum(xi, axis=0).reshape(
            self.config.n_states, self.config.n_states
        ).T / np.sum(gamma[:-1], axis=0).reshape(self.config.n_states, 1)

        # We use stochastic updates on trans_prob as per Eqs. (1) and (2) in the
        # paper:
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

    def get_posterior_entropy(self, gamma, xi):
        """Posterior entropy.

        Calculate the entropy of the posterior distribution:

        .. math::
            E &= \int q(s_{1:T}) \log q(s_{1:T}) ds_{1:T}

              &= \displaystyle\sum_{t=1}^{T-1} \int q(s_t, s_{t+1}) \
                 \log q(s_t, s_{t+1}) ds_t ds_{t+1} - \
                 \displaystyle\sum_{t=2}^{T-1} \
                 \int q(s_t) \log q(s_t) ds_t

        Parameters
        ----------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data,
            :math:`q(s_t)`. Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive
            time points, :math:`q(s_t, s_{t+1})`. Shape is
            (batch_size*sequence_length-1, n_states*n_states).

        Returns
        -------
        entropy : float
            Entropy.
        """
        # first_term = sum^{T-1}_t=1 int q(s_t, s_t+1)
        # log(q(s_t, s_t+1)) ds_t ds_t+1
        first_term = np.sum(xlogy(xi, xi))

        # second_term = sum^{T-1}_t=2 int q(s_t) log q(s_t) ds_t
        second_term = np.sum(xlogy(gamma, gamma)[1:-1])

        return first_term - second_term

    def get_posterior_expected_log_likelihood(self, x, gamma):
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
            Marginal posterior distribution of hidden states given the data,
            :math:`q(s_t)`. Shape is (batch_size*sequence_length, n_states).

        Returns
        -------
        log_likelihood : float
            Posterior expected log-likelihood.
        """
        gamma = np.reshape(gamma, (x.shape[0], x.shape[1], -1))
        log_likelihood = self.get_log_likelihood(x)
        return tf.stop_gradient(tf.reduce_sum(log_likelihood * gamma))

    def get_posterior_expected_prior(self, gamma, xi):
        """Posterior expected prior.

        Calculates the expected prior probability of states with respect to the
        posterior distribution of the states:

        .. math::
            P &= \int q(s_{1:T}) \log p(s_{1:T}) ds

              &= \int q(s_1) \log p(s_1) ds_1 + \displaystyle\sum_{t=1}^{T-1} \
                 \int q(s_t, s_{t+1}) \log p(s_{t+1} | s_t) ds_t ds_{t+1}

        Parameters
        ----------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data,
            :math:`q(s_t)`. Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive
            time points, :math:`q(s_t, s_{t+1})`. Shape is
            (batch_size*sequence_length-1, n_states*n_states).

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
        means, covs = self.get_means_covariances()
        mvn = tf.stop_gradient(
            tfp.distributions.MultivariateNormalTriL(
                loc=means,
                scale_tril=tf.linalg.cholesky(covs),
                allow_nan_stats=False,
            )
        )
        log_likelihood = mvn.log_prob(tf.expand_dims(data, axis=-2))
        return log_likelihood.numpy()

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

    def get_means(self):
        """Get the state means.

        Returns
        -------
        means : np.ndarray
            State means. Shape is (n_states, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "means")

    def get_covariances(self):
        """Get the state covariances.

        Returns
        -------
        covariances : np.ndarray
            State covariances. Shape is (n_states, n_channels, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "covs")

    def get_means_covariances(self):
        """Get the state means and covariances.

        This is a wrapper for :code:`get_means` and :code:`get_covariances`.

        Returns
        -------
        means : np.ndarary
            State means.
        covariances : np.ndarray
            State covariances.
        """
        return self.get_means(), self.get_covariances()

    def get_observation_model_parameters(self):
        """Wrapper for :code:`get_means_covariances`."""
        return self.get_means_covariances()

    def set_means(self, means, update_initializer=True):
        """Set the state means.

        Parameters
        ----------
        means : np.ndarray
            State means. Shape is (n_states, n_channels).
        update_initializer : bool, optional
            Do we want to use the passed means when we re-initialize the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            means,
            layer_name="means",
            update_initializer=update_initializer,
        )

    def set_covariances(self, covariances, update_initializer=True):
        """Set the state covariances.

        Parameters
        ----------
        covariances : np.ndarray
            State covariances. Shape is (n_states, n_channels, n_channels).
        update_initializer : bool, optional
            Do we want to use the passed covariances when we re-initialize
            the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            covariances,
            layer_name="covs",
            update_initializer=update_initializer,
            diagonal_covariances=self.config.diagonal_covariances,
        )

    def set_means_covariances(
        self,
        means,
        covariances,
        update_initializer=True,
    ):
        """This is a wrapper for :code:`set_means` and
        :code:`set_covariances`."""
        self.set_means(means, update_initializer=update_initializer)
        self.set_covariances(covariances, update_initializer=update_initializer)

    def set_observation_model_parameters(
        self, observation_model_parameters, update_initializer=True
    ):
        """Wrapper for :code:`set_means_covariances`."""
        self.set_means_covariances(
            observation_model_parameters[0],
            observation_model_parameters[1],
            update_initializer=update_initializer,
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

    def set_random_state_time_course_initialization(self, training_dataset):
        """Sets the initial means/covariances based on a random state time course.

        Parameters
        ----------
        training_dataset : tf.data.Dataset
            Training data.
        """
        _logger.info("Setting random means and covariances")

        # Mean and covariance for each state
        means = np.zeros(
            [self.config.n_states, self.config.n_channels], dtype=np.float32
        )
        covariances = np.zeros(
            [self.config.n_states, self.config.n_channels, self.config.n_channels],
            dtype=np.float32,
        )

        n_batches = 0
        for batch in training_dataset:
            # Concatenate all the sequences in this batch
            data = np.concatenate(batch["data"])

            if data.shape[0] < 2 * self.config.n_channels:
                raise ValueError(
                    "Not enough time points in batch, "
                    "increase batch_size or sequence_length"
                )

            # Sample a state time course using the initial transition
            # probability matrix
            stc = self.sample_state_time_course(data.shape[0])

            # Make sure each state activates
            non_active_states = np.sum(stc, axis=0) < 2 * self.config.n_channels
            while np.any(non_active_states):
                new_stc = self.sample_state_time_course(data.shape[0])
                new_active_states = np.sum(new_stc, axis=0) != 0
                for j in range(self.config.n_states):
                    if non_active_states[j] and new_active_states[j]:
                        stc[:, j] = new_stc[:, j]
                non_active_states = np.sum(stc, axis=0) < 2 * self.config.n_channels

            # Calculate the mean/covariance for each state for this batch
            m = []
            C = []
            for j in range(self.config.n_states):
                x = data[stc[:, j] == 1]
                mu = np.mean(x, axis=0)
                sigma = np.cov(x, rowvar=False)
                m.append(mu)
                C.append(sigma)
            means += m
            covariances += C
            n_batches += 1

        # Calculate the average from the running total
        means /= n_batches
        covariances /= n_batches

        if self.config.learn_means:
            # Set initial means
            self.set_means(means, update_initializer=True)

        if self.config.learn_covariances:
            # Set initial covariances
            self.set_covariances(covariances, update_initializer=True)

    def set_regularizers(self, training_dataset):
        """Set the means and covariances regularizer based on the training data.

        A multivariate normal prior is applied to the mean vectors with
        :code:`mu=0`, :code:`sigma=diag((range/2)**2)`. If
        :code:`config.diagonal_covariances=True`, a log normal prior is applied
        to the diagonal of the covariances matrices with :code:`mu=0`,
        :code:`sigma=sqrt(log(2*range))`, otherwise an inverse Wishart prior is
        applied to the covariances matrices with :code:`nu=n_channels-1+0.1`
        and :code:`psi=diag(1/range)`.

        Parameters
        ----------
        training_dataset : tf.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        """
        training_dataset = self.make_dataset(training_dataset, concatenate=True)

        if self.config.learn_means:
            obs_mod.set_means_regularizer(self.model, training_dataset)

        if self.config.learn_covariances:
            obs_mod.set_covariances_regularizer(
                self.model,
                training_dataset,
                self.config.covariances_epsilon,
                self.config.diagonal_covariances,
            )

    def free_energy(self, dataset):
        """Get the variational free energy.

        This calculates:

        .. math::
            \mathcal{F} = \int q(s_{1:T}) \log \left[ \
                          \\frac{q(s_{1:T})}{p(x_{1:T}, s_{1:T})} \\right] \
                          ds_{1:T}

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Dataset to evaluate the free energy for.

        Returns
        -------
        free_energy : float
            Variational free energy.
        """
        _logger.info("Getting free energy")

        # Convert to a TensorFlow dataset if not already
        dataset = self.make_dataset(dataset, concatenate=True)

        # Calculate variational free energy for each batch
        free_energy = []
        for data in dataset:
            x = data["data"]
            batch_size = x.shape[0]

            # Get the marginal and join posterior to calculate the free energy
            gamma, xi = self.get_posterior(x)

            # Calculate the free energy:
            #
            # F = int q(s) log[q(s) / p(x, s)] ds
            #   = int q(s) log[q(s) / p(x | s) p(s)] ds
            #   = - int q(s) log p(x | s) ds    [log_likelihood]
            #     + int q(s) log q(s) ds        [entropy]
            #     - int q(s) log p(s) ds        [prior]

            log_likelihood = self.get_posterior_expected_log_likelihood(x, gamma)
            entropy = self.get_posterior_entropy(gamma, xi)
            prior = self.get_posterior_expected_prior(gamma, xi)

            # Average free energy for a sequence in this batch
            seq_fe = (-log_likelihood + entropy - prior) / batch_size
            free_energy.append(seq_fe)

        # Return average over batches
        return np.mean(free_energy)

    def evidence(self, dataset):
        """Calculate the model evidence, :math:`p(x)`, of HMM on a dataset.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Dataset to evaluate the model evidence on.

        Returns
        -------
        evidence : float
            Model evidence.
        """

        # Helper functions
        def _evidence_predict_step(log_smoothing_distribution=None):
            """Predict step for calculating the evidence.

            .. math::
                p(s_t=j | x_{1:t-1}) = \displaystyle\sum_i p(s_t = j | s_{t-1} = i)\
                                                        p(s_{t-1} = i | x_{1:t-1})

            Parameters
            ----------
            log_smoothing_distribution : np.ndarray
                :math:`\log p(s_{t-1} | x_{1:t-1})`.
                Shape is (batch_size, n_states).

            Returns
            -------
            log_prediction_distribution : np.ndarray
                :math:`\log p(s_t | x_{1:t-1})`. Shape is (batch_size, n_states).
            """
            if log_smoothing_distribution is None:
                initial_distribution = self.get_stationary_distribution()
                log_prediction_distribution = np.broadcast_to(
                    np.expand_dims(initial_distribution, axis=0),
                    (batch_size, self.config.n_states),
                )
            else:
                log_trans_prob = np.expand_dims(np.log(self.trans_prob), 0)
                log_smoothing_distribution = np.expand_dims(
                    log_smoothing_distribution,
                    axis=-1,
                )
                log_prediction_distribution = logsumexp(
                    log_trans_prob + log_smoothing_distribution, -2
                )
            return log_prediction_distribution

        def _evidence_update_step(data, log_prediction_distribution):
            """Update step for calculating the evidence.

            .. math::
                p(s_t = j | x_{1:t}) &= \displaystyle\\frac{p(x_t | s_t = j) \
                                        p(s_t = j | x_{1:t-1})}{p(x_t | x_{1:t-1})}

                p(x_t | x_{1:t-1}) &= \displaystyle\sum_i p(x_t | s_t = j) \
                                                        p(s_t = i | x_{1:t-1})

            Parameters
            ----------
            data : np.ndarray
                Data for the update step. Shape is (batch_size, n_channels).
            log_prediction_distribution : np.ndarray
                :math:`\log p(s_t | x_{1:t-1})`. Shape is (batch_size, n_states).

            Returns
            -------
            log_smoothing_distribution : np.ndarray
                :math:`\log p(s_t | x_{1:t})`. Shape is (batch_size, n_states).
            predictive_log_likelihood : np.ndarray
                :math:`\log p(x_t | x_{1:t-1})`. Shape is (batch_size,).
            """
            log_likelihood = self.get_log_likelihood(data)
            log_smoothing_distribution = log_likelihood + log_prediction_distribution
            predictive_log_likelihood = logsumexp(log_smoothing_distribution, -1)

            # Normalise the log smoothing distribution
            log_smoothing_distribution -= np.expand_dims(
                predictive_log_likelihood,
                axis=-1,
            )
            return log_smoothing_distribution, predictive_log_likelihood

        _logger.info("Getting model evidence")
        dataset = self.make_dataset(dataset, concatenate=True)
        n_batches = dtf.get_n_batches(dataset)

        evidence = 0
        for n, data in enumerate(dataset):
            x = data["data"]
            print("Batch {}/{}".format(n + 1, n_batches))
            pb_i = utils.Progbar(self.config.sequence_length)
            batch_size = tf.shape(x)[0]
            batch_evidence = np.zeros((batch_size))
            log_smoothing_distribution = None
            for t in range(self.config.sequence_length):
                # Prediction step
                log_prediction_distribution = _evidence_predict_step(
                    log_smoothing_distribution
                )

                # Update step
                (
                    log_smoothing_distribution,
                    predictive_log_likelihood,
                ) = _evidence_update_step(x[:, t, :], log_prediction_distribution)

                # Update the batch evidence
                batch_evidence += predictive_log_likelihood
                pb_i.add(1)
            evidence += np.mean(batch_evidence)

        return evidence / n_batches

    def get_alpha(self, dataset, concatenate=False, remove_edge_effects=False):
        """Get state probabilities.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Prediction dataset. This can be a list of datasets, one for
            each session.
        concatenate : bool, optional
            Should we concatenate alpha for each session?
        remove_edge_effects : bool, optional
            Edge effects can arise due to separating the data into sequences.
            We can remove these by predicting overlapping :code:`alpha` and
            disregarding the :code:`alpha` near the ends. Passing :code:`True`
            does this by using sequences with 50% overlap and throwing away the
            first and last 25% of predictions.

        Returns
        -------
        alpha : list or np.ndarray
            State probabilities with shape (n_sessions, n_samples, n_states)
            or (n_samples, n_states).
        """
        if remove_edge_effects:
            step_size = self.config.sequence_length // 2  # 50% overlap
            trim = step_size // 2  # throw away 25%
        else:
            step_size = None

        dataset = self.make_dataset(dataset, step_size=step_size)

        n_datasets = len(dataset)
        if len(dataset) > 1:
            iterator = trange(n_datasets, desc="Getting alpha")
        else:
            iterator = range(n_datasets)
            _logger.info("Getting alpha")

        alpha = []
        for i in iterator:
            gamma = []
            for j, data in enumerate(dataset[i]):
                n_batches = dtf.get_n_batches(dataset[i])
                x = data["data"]
                g, _ = self.get_posterior(x)
                if remove_edge_effects:
                    batch_size, sequence_length, _ = x.shape
                    n_states = g.shape[-1]
                    g = g.reshape(batch_size, sequence_length, n_states)
                    if j == 0:
                        g = [
                            g[0, :-trim],
                            g[1:, trim:-trim].reshape(-1, n_states),
                        ]
                    elif j == n_batches - 1:
                        g = [
                            g[:-1, trim:-trim].reshape(-1, n_states),
                            g[-1, trim:],
                        ]
                    else:
                        g = [g[:, trim:-trim].reshape(-1, n_states)]
                    g = np.concatenate(g).reshape(-1, n_states)
                gamma.append(g)
            alpha.append(np.concatenate(gamma).astype(np.float32))

        if concatenate or len(alpha) == 1:
            alpha = np.concatenate(alpha)

        return alpha

    def get_viterbi_path(self, dataset, concatenate=False):
        """Get the Viterbi path with the Viterbi algorithm.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Prediction dataset. This can be a list of datasets, one for
            each session.
        concatenate : bool, optional
            Should we concatenate the Viterbi path for each session?

        Returns
        -------
        viterbi_path : list or np.ndarray
            Viterbi path with shape (n_sessions, n_samples) or (n_samples,).
        """
        P_0 = self.state_probs_t0
        P = self.trans_prob
        n_states = P.shape[0]

        log_P_0 = np.log(P_0 + EPS)
        log_P = np.log(P + EPS)

        def _viterbi_path(x):
            # x.shape = (batch_size, sequence_length, n_channels)
            batch_size = x.shape[0]
            sequence_length = x.shape[1]

            log_b = self.get_log_likelihood(x)
            log_b = log_b.reshape(batch_size * sequence_length, n_states)

            log_prob = np.empty((batch_size * sequence_length, n_states), dtype=float)
            prev = np.empty((batch_size * sequence_length, n_states), dtype=int)

            log_prob[0] = log_P_0 + log_b[0]
            for t in range(1, batch_size * sequence_length):
                place_holder = log_prob[t - 1] + log_P.T + log_b[t]
                log_prob[t] = np.max(place_holder, axis=1)
                prev[t] = np.argmax(place_holder, axis=1)

            # Backtrack
            path = np.empty(batch_size * sequence_length, dtype=int)
            path[-1] = np.argmax(log_prob[-1])
            for t in range(batch_size * sequence_length - 2, -1, -1):
                path[t] = prev[t + 1, path[t + 1]]

            return path

        dataset = self.make_dataset(dataset)
        n_datasets = len(dataset)
        if len(dataset) > 1:
            iterator = trange(n_datasets, desc="Getting Viterbi path")
        else:
            iterator = range(n_datasets)
            _logger.info("Getting Viterbi path")

        viterbi_path = []
        for i in iterator:
            path = []
            for data in dataset[i]:
                x = data["data"]
                path.append(_viterbi_path(x))
            viterbi_path.append(array_ops.get_one_hot(np.concatenate(path), n_states))

        if concatenate or len(viterbi_path) == 1:
            viterbi_path = np.concatenate(viterbi_path)

        return viterbi_path

    def get_n_params_generative_model(self):
        """Get the number of trainable parameters in the generative model.

        This includes the transition probabiltity matrix, state means and
        covariances.

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
            if "means" in var_name or "covs" in var_name:
                n_params += np.prod(var.shape)

        return int(n_params)

    def bayesian_information_criterion(self, dataset, loss_type="free_energy"):
        """Calculate the Bayesian Information Criterion (BIC) for the model.

        Parameters
        ----------
        dataset : osl_dynamics.data.Data
            Dataset to calculate the BIC for.
        loss_type : str, optional
            Which loss to use for the BIC. Can be :code:`"free_energy"`
            or :code:`"evidence"`.

        Returns
        -------
        bic : float
            Bayesian Information Criterion for the model (for each sequence).
        """
        if loss_type == "free_energy":
            loss = self.free_energy(dataset)
        elif loss_type == "evidence":
            loss = -self.evidence(dataset)
        else:
            raise ValueError("loss_type must be 'free_energy' or 'evidence'")

        n_params = self.get_n_params_generative_model()
        n_sequences = dtf.n_batches(
            dataset.time_series(concatenate=True), self.config.sequence_length
        )

        bic = (
            2 * loss
            + (np.log(self.config.sequence_length) + np.log(n_sequences))
            * n_params
            / n_sequences
        )
        return bic

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
            Each element has shape (n_samples, n_modes).
        means : np.ndarray
            Session-specific means. Shape is (n_sessions, n_modes, n_channels).
        covariances : np.ndarray
            Session-specific covariances.
            Shape is (n_sessions, n_modes, n_channels, n_channels).
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
        means = []
        covariances = []
        with set_logging_level(_logger, logging.WARNING):
            for i in trange(training_data.n_sessions, desc="Fine tuning"):
                # Train on this session
                with training_data.set_keep(i):
                    self.fit(training_data, verbose=0)
                    a = self.get_alpha(training_data, concatenate=True)

                # Get the inferred parameters
                m, c = self.get_means_covariances()
                alpha.append(a)
                means.append(m)
                covariances.append(c)

                # Reset back to group-level model parameters
                self.load_weights(f"{store_dir}/weights.h5")
                self.compile()

        # Reset hyperparameters
        self.config.n_epochs = original_n_epochs
        self.config.learning_rate = original_learning_rate
        self.config.learn_trans_prob = original_learn_trans_prob

        return alpha, np.array(means), np.array(covariances)

    def dual_estimation(self, training_data, alpha=None, n_jobs=1):
        """Dual estimation to get session-specific observation model parameters.

        Here, we estimate the state means and covariances for sessions
        with the posterior distribution of the states held fixed.

        Parameters
        ----------
        training_data : osl_dynamics.data.Data or list of tf.data.Dataset
            Prepared training data object.
        alpha : list of np.ndarray, optional
            Posterior distribution of the states. Shape is
            (n_sessions, n_samples, n_states).
        n_jobs : int, optional
            Number of jobs to run in parallel.

        Returns
        -------
        means : np.ndarray
            Session-specific means. Shape is (n_sessions, n_states, n_channels).
        covariances : np.ndarray
            Session-specific covariances.
            Shape is (n_sessions, n_states, n_channels, n_channels).
        """
        if alpha is None:
            # Get the posterior
            alpha = self.get_alpha(training_data, concatenate=False)

        # Validation
        if isinstance(alpha, np.ndarray):
            alpha = [alpha]

        # Get the session-specific data
        if isinstance(training_data, list):
            data = []
            for d in training_data:
                subject_data = []
                for batch in d:
                    subject_data.append(np.concatenate(batch["data"]))
                data.append(np.concatenate(subject_data))
        else:
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
            if self.config.learn_means:
                session_means = np.empty((n_states, n_channels))
                for state in range(n_states):
                    session_means[state] = (
                        np.sum(x * a[:, state, None], axis=0) / sum_a[state]
                    )
            else:
                session_means = self.get_means()

            if self.config.learn_covariances:
                session_covariances = np.empty((n_states, n_channels, n_channels))
                for state in range(n_states):
                    diff = x - session_means[state]
                    session_covariances[state] = (
                        np.sum(
                            diff[:, :, None]
                            * diff[:, None, :]
                            * a[:, state, None, None],
                            axis=0,
                        )
                        / sum_a[state]
                    )
                    session_covariances[
                        state
                    ] += self.config.covariances_epsilon * np.eye(
                        self.config.n_channels
                    )
            else:
                session_covariances = self.get_covariances()

            return session_means, session_covariances

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

        # Unpack the results
        means = []
        covariances = []
        for result in results:
            m, c = result
            means.append(m)
            covariances.append(c)

        return np.squeeze(means), np.squeeze(covariances)

    def save_weights(self, filepath):
        """Save all model weights.

        Parameters
        ----------
        filepath : str
            Location to save model weights to.
        """
        self.model.save_weights(filepath)
        np.save(
            op.join(str(Path(filepath).parent), "trans_prob.npy"),
            self.trans_prob,
        )

    def load_weights(self, filepath):
        """Load all model parameters.

        Parameters
        ----------
        filepath : str
            Location to load model weights from.
        """
        self.trans_prob = np.load(
            op.join(str(Path(filepath).parent), "trans_prob.npy"),
        )
        return self.model.load_weights(filepath)

    def get_weights(self):
        """Get model parameter weights.

        Returns
        -------
        weights : tensorflow weights
            TensorFlow weights for the observation model.
        trans_prob : np.ndarray
            Transition probability matrix.
        """
        return self.model.get_weights(), self.trans_prob

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
        initializers.reinitialize_model_weights(self.model)
        self.set_trans_prob(self.config.initial_trans_prob)

    def _model_structure(self):
        """Build the model structure."""

        config = self.config

        # Inputs
        inputs = layers.Input(
            shape=(config.sequence_length, config.n_channels + config.n_states),
            name="inputs",
        )
        data, gamma = tf.split(inputs, [config.n_channels, config.n_states], axis=2)

        # Static loss scaling factor
        static_loss_scaling_factor_layer = StaticLossScalingFactorLayer(
            config.sequence_length,
            config.loss_calc,
            name="static_loss_scaling_factor",
        )
        static_loss_scaling_factor = static_loss_scaling_factor_layer(data)

        # Definition of layers
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

        # Data flow
        mu = means_layer(
            data, static_loss_scaling_factor=static_loss_scaling_factor
        )  # data not used
        D = covs_layer(
            data, static_loss_scaling_factor=static_loss_scaling_factor
        )  # data not used
        ll_loss = ll_loss_layer([data, mu, D, gamma, None])

        return tf.keras.Model(inputs=inputs, outputs=[ll_loss], name="HMM")
