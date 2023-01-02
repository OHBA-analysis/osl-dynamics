"""Hidden Markov Model (HMM).

"""

import sys
import warnings
import os.path as op
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend, layers, utils
from numba.core.errors import NumbaWarning

import osl_dynamics.data.tf as dtf
from osl_dynamics.simulation import HMM
from osl_dynamics.inference import initializers
from osl_dynamics.models import dynemo_obs
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.inference.layers import (
    VectorsLayer,
    CovarianceMatricesLayer,
    CategoricalLogLikelihoodLossLayer,
)

warnings.filterwarnings("ignore", category=NumbaWarning)

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
        Should we make the mean vectors for each mode trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each mode trainable?
    initial_means : np.ndarray
        Initialisation for mean vectors.
    initial_covariances : np.ndarray
        Initialisation for mode covariances.
    covariances_epsilon : float
        Error added to mode covariances for numerical stability.
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

    model_name: str = "HMM"

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None
    covariances_epsilon: float = None

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

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")

        if self.covariances_epsilon is None:
            if self.learn_covariances:
                self.covariances_epsilon = 1e-6
            else:
                self.covariances_epsilon = 0.0


class Model(ModelBase):
    """HMM class.

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

    def fit(self, dataset, epochs=None, take=1, **kwargs):
        """Fit model to a dataset.

        Iterates between:
        - Baum-Welch updates of latent variable time courses and transition probability matrix.
        - TensorFlow updates of observation model parameters.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        epochs : int
            Number of epochs.
        take : float
            Fraction of total batches to take.
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

        # Make a TensorFlow Dataset
        dataset = self.make_dataset(dataset, shuffle=True, concatenate=True)

        if take != 1:
            # Print a message stating how many batches we'll use
            n_total_batches = dtf.get_n_batches(dataset)
            n_batches = max(round(n_total_batches * take), 1)
            print(f"Using {n_batches} out of {n_total_batches} batches")

        history = {"loss": [], "rho": [], "lr": []}
        for n in range(epochs):
            if n == epochs - 1:
                # If it's the last epoch, we train on the full dataset
                take = 1

            # Get the training data for this epoch
            if take != 1:
                dataset.shuffle(100000)
                n_batches = max(round(n_total_batches * take), 1)
                data_subset = dataset.take(n_batches)
            else:
                data_subset = dataset

            # Setup a progress bar
            print("Epoch {}/{}".format(n + 1, epochs))
            pb_i = utils.Progbar(len(data_subset))

            # Update rho
            self._update_rho(n)

            # Set learning rate for the observation model
            lr = self.config.learning_rate * np.exp(
                -self.config.observation_update_decay * n
            )
            backend.set_value(self.model.optimizer.lr, lr)

            # Loop over batches
            loss = []
            for data in data_subset:
                x = data["data"]

                # Update state probabilities
                gamma, xi = self._get_state_probs(x)

                # Update transition probability matrix
                if self.config.learn_trans_prob:
                    self._update_trans_prob(gamma, xi)

                # Reshape gamma: (batch_size*sequence_length, n_states)
                # -> (batch_size, sequence_length, n_states)
                gamma = gamma.reshape(x.shape[0], x.shape[1], -1)

                # Update observation model
                training_data = np.concatenate([x, gamma], axis=2)
                h = self.model.fit(training_data, epochs=1, verbose=0, **kwargs)

                l = h.history["loss"][0]
                if np.isnan(l):
                    print("\nTraining failed!")
                    return
                loss.append(l)
                pb_i.add(1, values=[("rho", self.rho), ("lr", lr), ("loss", l)])

            history["loss"].append(np.mean(loss))
            history["rho"].append(self.rho)
            history["lr"].append(lr)

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
            print(
                "Number of initializations was set to zero. "
                + "Skipping initialization."
            )
            return

        print("Random subset initialization:")

        # Make a TensorFlow Dataset
        training_data = self.make_dataset(training_data, shuffle=True, concatenate=True)

        # Calculate the number of batches to use
        n_total_batches = dtf.get_n_batches(training_data)
        n_batches = max(round(n_total_batches * take), 1)
        print(f"Using {n_batches} out of {n_total_batches} batches")

        # Pick the initialization with the lowest free energy
        best_loss = np.Inf
        for n in range(n_init):
            print(f"Initialization {n}")
            self.reset()
            training_data.shuffle(100000)
            training_data_subset = training_data.take(n_batches)
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
            print("Initialization failed")
            return

        print(f"Using initialization {best_initialization}")
        self.reset()
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
            print(
                "Number of initializations was set to zero. "
                + "Skipping initialization."
            )
            return

        print("Random state time course initialization:")

        # Make a TensorFlow Dataset
        training_dataset = self.make_dataset(
            training_data, shuffle=True, concatenate=True
        )

        # Calculate the number of batches to use
        n_total_batches = dtf.get_n_batches(training_dataset)
        n_batches = max(round(n_total_batches * take), 1)
        print(f"Using {n_batches} out of {n_total_batches} batches")

        # Pick the initialization with the lowest free energy
        best_loss = np.Inf
        for n in range(n_init):
            print(f"Initialization {n}")
            self.reset()
            self.set_random_state_time_course_initialization(training_data)
            training_dataset.shuffle(100000)
            training_data_subset = training_dataset.take(n_batches)
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
            print("Initialization failed")
            return

        print(f"Using initialization {best_initialization}")
        self.reset()
        self.set_weights(best_weights, best_trans_prob)

        return best_history

    def _get_state_probs(self, x):
        """Get state probabilities.

        Parameters
        ----------
        x : np.ndarray
            Observed data. Shape is (batch_size, sequence_length, n_channels).
        Returns
        -------
        gamma : np.ndarray
            Probability of hidden state given data.
            Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Probability of hidden state given child and parent states, given data.
            Shape is (batch_size*sequence_length - 1, n_states, n_states).
        """

        # Use Baum-Welch algorithm to calculate gamma, xi
        B = self._get_likelihood(x)
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
            Probability of hidden state given data. Shape is (n_samples, n_states).
        xi : np.ndarray
            Probability of hidden state given child and parent states, given data.
            Shape is (n_samples - 1, n_states*n_states).
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

        b = beta[1:] * B[:, 1:].T
        t = P * np.expand_dims(alpha[:-1], axis=2) * np.expand_dims(b, axis=1)
        xi = t.reshape(n_samples - 1, -1, order="F")
        xi /= np.expand_dims(np.sum(xi, axis=1), axis=1) + EPS

        return gamma, xi

    def _get_likelihood(self, x):
        """Get likelihood time series.

        Parameters
        ----------
        x : np.ndarray
            Observed data. Shape is (batch_size, sequence_length, n_channels).

        Returns
        -------
        likelihood : np.ndarray
            Likelihood time series. Shape is (n_states, batch_size*sequence_length).
        """

        # Get the current observation model parameters
        means, covs = self.get_means_covariances()

        n_states = means.shape[0]
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        n_samples = batch_size * sequence_length

        # Calculate the log-likelihood for each state to have generated the
        # observed data
        log_likelihood = np.empty([n_states, n_samples])
        for state in range(n_states):
            mvn = tfp.distributions.MultivariateNormalTriL(
                loc=means[state],
                scale_tril=tf.linalg.cholesky(covs[state]),
                allow_nan_stats=False,
            )
            log_likelihood[state] = np.reshape(mvn.log_prob(x), n_samples)

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
            Probability of hidden state given data.
            Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Probability of hidden state given child and parent states, given data.
            Shape is (batch_size*sequence_length - 1, n_states*n_states).
        """
        # Use Baum-Welch algorithm
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

    def get_covariances(self):
        """Get the covariances of each mode.

        Returns
        -------
        covariances : np.ndarray
            State covariances. Shape is (n_states, n_channels, n_channels).
        """
        return dynemo_obs.get_covariances(self.model)

    def get_means_covariances(self):
        """Get the means and covariances of each mode.

        Returns
        -------
        means : np.ndarary
            Mode means.
        covariances : np.ndarray
            Mode covariances.
        """
        return dynemo_obs.get_means_covariances(self.model)

    def set_means(self, means, update_initializer=True):
        """Set the means of each mode.

        Parameters
        ----------
        means : np.ndarray
            Mode covariances.
        update_initializer : bool
            Do we want to use the passed means when we re-initialize
            the model?
        """
        dynemo_obs.set_means(self.model, means, update_initializer)

    def set_covariances(self, covariances, update_initializer=True):
        """Set the covariances of each mode.

        Parameters
        ----------
        covariances : np.ndarray
            Mode covariances.
        update_initializer : bool
            Do we want to use the passed covariances when we re-initialize
            the model?
        """
        dynemo_obs.set_covariances(self.model, covariances, update_initializer)

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
        training_data : osl_dynamics.data.Data
            Training data object.
        """

        # Loop over subjects
        subject_means = []
        subject_covariances = []
        for data in training_data.subjects:

            # Sample a state time course using the initial transition
            # probability matrix
            stc = self.sample_state_time_course(data.shape[0])

            # Calculate the mean/covariance for each state for this subject
            m = []
            C = []
            for j in range(self.config.n_states):
                x = data[stc[:, j] == 1]
                mu_j = np.mean(x, axis=0)
                sigma_j = np.cov(x, rowvar=False)
                m.append(mu_j)
                C.append(sigma_j)

            subject_means.append(m)
            subject_covariances.append(C)

        # Average over subjects
        initial_means = np.mean(subject_means, axis=0)
        initial_covariances = np.mean(subject_covariances, axis=0)

        if self.config.learn_means:
            # Set initial means
            self.set_means(initial_means, update_initializer=True)

        if self.config.learn_covariances:
            # Set initial covariances
            self.set_covariances(initial_covariances, update_initializer=True)

    def set_regularizers(self, training_dataset):
        """Set the means and covariances regularizer based on the training data.

        A multivariate normal prior is applied to the mean vectors with mu = 0,
        sigma=diag((range / 2)**2) and an inverse Wishart prior is applied to the
        covariances matrices with nu=n_channels - 1 + 0.1 and psi=diag(range).

        Parameters
        ----------
        training_dataset : tensorflow.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        """
        training_dataset = self.make_dataset(training_dataset, concatenate=True)

        if self.config.learn_means:
            dynemo_obs.set_means_regularizer(self.model, training_dataset)

        if self.config.learn_covariances:
            dynemo_obs.set_covariances_regularizer(self.model, training_dataset)

    def get_alpha(self, dataset, concatenate=False):
        """Get state probabilities.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Prediction dataset for each subject.
        concatenate : bool
            Should we concatenate alpha for each subject?

        Returns
        -------
        alpha : list or np.ndarray
            State probabilities with shape (n_subjects, n_samples, n_states)
            or (n_samples, n_states).
        """
        dataset = self.make_dataset(dataset, concatenate=False)

        print("Getting alpha")
        alpha = []
        for ds in dataset:
            gamma = []
            for data in ds:
                x = data["data"]
                g, _ = self._get_state_probs(x)
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
        initializers.reinitialize_model_weights(self.model)
        self.set_trans_prob(self.config.initial_trans_prob)


def _model_structure(config):
    # Inputs
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels + config.n_states),
        name="inputs",
    )
    data, gamma = tf.split(inputs, [config.n_channels, config.n_states], axis=2)

    # Definition of layers
    means_layer = VectorsLayer(
        config.n_states,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        name="means",
    )
    covs_layer = CovarianceMatricesLayer(
        config.n_states,
        config.n_channels,
        config.learn_covariances,
        config.initial_covariances,
        config.covariances_epsilon,
        name="covs",
    )
    ll_loss_layer = CategoricalLogLikelihoodLossLayer(
        config.n_states, config.covariances_epsilon, name="ll_loss"
    )

    # Data flow
    mu = means_layer(data)  # data not used
    D = covs_layer(data)  # data not used
    ll_loss = ll_loss_layer([data, mu, D, gamma])

    return tf.keras.Model(inputs=inputs, outputs=[ll_loss], name="HMM-Obs")
