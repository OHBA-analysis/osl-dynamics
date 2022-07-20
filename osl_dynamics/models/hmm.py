"""Hidden Markov Model (HMM).

NOTE:

- This model is still under development.
- This model requires a C library which has been compiled for BMRC, but will need to be recompiled if running on a different computer. The C library can be found on BMRC here: /well/woolrich/projects/software/hmm_inference_libc

"""

import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, utils

from osl_dynamics.simulation import HMM
from osl_dynamics.models import dynemo_obs
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.inference.layers import (
    MeanVectorsLayer,
    CovarianceMatricesLayer,
    CategoricalLogLikelihoodLossLayer,
)

from ctypes import c_void_p, c_double, c_int, CDLL
from numpy.ctypeslib import ndpointer

# Load a C library for hidden state inference.
# This library has has already been compiled for BMRC.
# Please re-compile if running on a different computer with:
# `python setup.py build`, and update the path.
libfile = "/well/woolrich/projects/software/hmm_inference_libc/build/lib.linux-x86_64-3.8/hidden_state_inference.so"
hidden_state_inference = CDLL(libfile)


@dataclass
class Config(BaseModelConfig):
    """Settings for HMM.

    Parameters
    ----------
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
    initial_transprob : np.ndarray
        Initialisation for trans prob matrix
    learn_transprob : bool
        Should we make the trans prob matrix trainable?
    state_probs_t0: np.ndarray
        State probabilities at time=0. Not trainable.
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
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None

    initial_transprob: np.ndarray = None
    learn_transprob: bool = True
    state_probs_t0: np.ndarray = None

    stochastic_update_delay: float = 0  # alpha
    stochastic_update_forget: float = 0.5  # beta

    def __post_init__(self):
        self.validate_observation_model_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")


class Model(ModelBase):
    """HMM class.

    Parameters
    ----------
    config : osl_dynamics.models.hmm.Config
    """

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

        self.rho = 1
        initial_transprob = self.config.initial_transprob
        if initial_transprob is None:
            initial_transprob = (
                np.ones((self.config.n_states, self.config.n_states))
                * 0.1
                / self.config.n_states
            )
            np.fill_diagonal(initial_transprob, 0.9)
        self.transprob = initial_transprob

        if self.config.state_probs_t0 is None:
            self.state_probs_t0 = (
                np.ones((self.config.n_states,)) / self.config.n_states
            )  # state probs at time 0

    def fit(self, dataset, epochs=None):
        """Fit model to a dataset.

        Iterates between:
        - Baum-Welch updates of latent variable time courses and transition probability matrix.
        - TensorFlow updates of observation model parameters.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Training dataset.
        epochs : int
            Number of epochs.

        Returns
        -------
        history : dict
            Dictionary with loss and rho history. Keys are 'loss' and 'rho'.
        """
        if epochs is None:
            epochs = self.config.n_epochs

        history = {"loss": [], "rho": []}
        for n in range(epochs):
            print("Epoch {}/{}".format(n + 1, epochs))
            pb_i = utils.Progbar(len(dataset))

            # Update rho
            self._update_rho(n)

            # Loop over batches
            loss = []
            for data in dataset:
                x = data["data"]

                # Update state probabilities
                gamma, xi = self._get_state_probs(x)

                # Update transition probability matrix
                if self.config.learn_transprob:
                    self._update_transprob(gamma, xi)

                # Update observation model
                training_data = np.concatenate([x, gamma], axis=2)
                h = self.model.fit(training_data, epochs=1, verbose=0)

                l = h.history["loss"][0]
                loss.append(l)
                pb_i.add(1, values=[("loss", l)])

            history["loss"].append(np.mean(loss))
            history["rho"].append(self.rho)

        return history

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
            Shape is (batch_size*sequence_length - 1, n_states*n_states).
        """
        # Would be order>0 if observation model is MAR for example,
        # for MVN observation model order=0
        order = 0

        likelihood = self._get_likelihood(x)
        n_samples = likelihood.shape[1] * likelihood.shape[2]
        likelihood = np.reshape(likelihood, [-1, n_samples])  # (n_states, n_samples)

        # Outputs
        gamma = np.zeros(((n_samples - order), self.config.n_states))
        xi = np.zeros(
            ((n_samples - 1 - order), self.config.n_states * self.config.n_states)
        )
        scale = np.zeros((n_samples, 1))

        # We need to have numpy contiguous arrays (items in array are in contiguous
        # locations in memory) to pass into the C++ update function
        gamma_cont = np.ascontiguousarray(gamma.T, np.double)
        xi_cont = np.ascontiguousarray(xi.T, np.double)
        scale_cont = np.ascontiguousarray(scale.T, np.double)

        # Inputs
        # We need to have numpy contiguous arrays (items in array are in contiguous
        # locations in memory) to pass into the C++ update function
        B_cont = np.ascontiguousarray(likelihood, np.double)
        Pi_0_cont = np.ascontiguousarray(self.state_probs_t0.T, np.double)
        transprob_cont = np.ascontiguousarray(self.transprob.T, np.double)

        # Define C++ class initialiser outputs types
        hidden_state_inference.new_inferer.restype = c_void_p

        # Define input types
        hidden_state_inference.new_inferer.argtypes = [c_int, c_int]

        inferer_obj = hidden_state_inference.new_inferer(self.config.n_states, order)

        hidden_state_inference.state_inference.argtypes = [
            c_void_p,
            ndpointer(dtype=c_double, shape=B_cont.shape),
            ndpointer(dtype=c_double, shape=Pi_0_cont.shape),
            ndpointer(dtype=c_double, shape=transprob_cont.shape),
            c_int,
            ndpointer(dtype=c_double, shape=gamma_cont.shape),
            ndpointer(dtype=c_double, shape=xi_cont.shape),
            ndpointer(dtype=c_double, shape=scale_cont.shape),
        ]

        # Uses Baum-Welch algorithm to update gamma_cont, xi_cont, scale_cont
        hidden_state_inference.state_inference(
            inferer_obj,
            B_cont,
            Pi_0_cont,
            transprob_cont,
            n_samples,
            gamma_cont,
            xi_cont,
            scale_cont,
        )

        # Reshape gamma: (n_states, batch_size*sequence_length)
        # -> (batch_size, sequence_length, n_states)
        gamma = np.reshape(gamma_cont, [-1, x.shape[0], x.shape[1]])
        gamma = np.transpose(gamma, [1, 2, 0])

        return gamma, xi_cont

    def _get_likelihood(self, x):
        """Get likelihood time series.

        Parameters
        ----------
        x : np.ndarray
            Observed data. Shape is (batch_size, sequence_length, n_channels).

        Returns
        -------
        np.ndarray
            Likelihood time series. Shape is (n_states, batch_size, sequence_length).
        """
        means, covs = self.get_means_covariances()

        n_states = means.shape[0]
        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        log_likelihood = np.empty([n_states, batch_size, sequence_length])
        for state in range(n_states):
            mvn = tfp.distributions.MultivariateNormalTriL(
                loc=means[state],
                scale_tril=tf.linalg.cholesky(covs[state]),
                allow_nan_stats=False,
            )
            log_likelihood[state] = mvn.log_prob(x)

        return np.exp(log_likelihood)

    def _update_transprob(self, gamma, xi):
        """Update transition probability matrix.

        Parameters
        ----------
        gamma : np.ndarray
            Probability of hidden state given data.
            Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Probability of hidden state given child and parent states, given data.
            Shape is (batch_size*sequence_length - 1, n_states, n_states).
        """
        # Reshape gamma: (batch_size, sequence_length, n_states)
        # -> (n_states, batch_size*sequence_length)
        gamma = gamma.reshape(-1, gamma.shape[-1]).T

        # Use Baum-Welch algorithm
        phi_interim = np.reshape(
            np.sum(xi, 1), [self.config.n_states, self.config.n_states]
        ).T / np.reshape(np.sum(gamma[:, :-1], 1), [self.config.n_states, 1])

        # We use stochastic updates on transprob as per Eqs. (1) and (2) in the paper:
        # https://www.sciencedirect.com/science/article/pii/S1053811917305487
        self.transprob = (1 - self.rho) * self.transprob + self.rho * phi_interim

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
            100 * ind / self.config.n_epochs + 1 + self.config.stochastic_update_delay,
            -self.config.stochastic_update_forget,
        )

    def sample_stc(self, n_samples):
        """Sample a state time course.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        np.ndarray
            State time course with shape (n_samples, n_states).
        """
        sim = HMM(self.transprob)
        stc = sim.generate_states(n_samples)
        return stc

    def get_transprob(self):
        """Get the transition probability matrix.

        Returns
        -------
        np.ndarray
        """
        return self.transprob

    def get_covariances(self):
        """Get the covariances of each mode.

        Returns
        -------
        np.ndarary
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
        list or np.ndarray
            State probabilities with shape (n_subjects, n_samples, n_states)
            or (n_samples, n_states).
        """
        dataset = self._make_dataset(dataset)

        alpha = []
        for ds in dataset:
            gamma = []
            for data in ds:
                g, _ = self._get_state_probs(data["data"])
                gamma.append(np.concatenate(g))
            alpha.append(np.concatenate(gamma))

        if concatenate or len(alpha) == 1:
            alpha = np.concatenate(alpha)

        return alpha

    def save_all(self, dirname):
        """Save all model weights.

        Parameters
        ----------
        dirname : str
            Location to save model parameters to.
        """
        os.system("mkdir {}".format(dirname))
        self.save_weights(os.path.join(dirname, "model_weights"))
        np.save(os.path.join(dirname, "model_transprob.npy"), self.transprob)

    def load_all(self, dirname):
        """Load all model parameters.

        Parameters
        ----------
        dirname : str
            Location to load model parameters from.
        """
        self.load_weights(os.path.join(dirname, "model_weights"))
        self.transprob = np.load(os.path.join(dirname, "model_transprob.npy"))


def _model_structure(config):
    # Inputs
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels + config.n_states),
        name="inputs",
    )
    data, gamma = tf.split(inputs, [config.n_channels, config.n_states], 2)

    # Definition of layers
    means_layer = MeanVectorsLayer(
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
        name="covs",
    )
    ll_loss_layer = CategoricalLogLikelihoodLossLayer(config.n_states, name="ll_loss")

    # Data flow
    mu = means_layer(data)  # data not used
    D = covs_layer(data)  # data not used
    ll_loss = ll_loss_layer([data, mu, D, gamma])

    return tf.keras.Model(inputs=inputs, outputs=[ll_loss], name="HMM-Obs")
