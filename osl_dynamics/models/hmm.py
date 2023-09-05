"""Hidden Markov Model (HMM).

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

import os
import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm.auto import trange

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference.layers import (
    CovarianceMatricesLayer,
    DiagonalMatricesLayer,
    VectorsLayer,
    StaticLossScalingFactorLayer,
    HiddenStateInferenceLayer,
    SeparateLogLikelihoodLayer,
    SumLogLikelihoodLossLayer,
)
from osl_dynamics.models import obs_mod
from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.models.inf_mod_base import (
    MarkovStateInferenceModelConfig,
    MarkovStateInferenceModelBase,
)
from osl_dynamics.utils.misc import set_logging_level

_logger = logging.getLogger("osl-dynamics")


@dataclass
class Config(BaseModelConfig, MarkovStateInferenceModelConfig):
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

    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
    n_epochs : int
        Number of training epochs.
    optimizer : str or tf.keras.optimizers.Optimizer
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
    diagonal_covariances: bool = False
    covariances_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

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


class Model(MarkovStateInferenceModelBase):
    """HMM class.

    Parameters
    ----------
    config : osl_dynamics.models.hmm.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

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

    def subject_fine_tuning(
        self, training_data, n_epochs=None, learning_rate=None, store_dir="tmp"
    ):
        """Fine tuning the model for each subject.

        Here, we estimate the posterior distribution (state probabilities)
        and observation model using the data from a single subject with the
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
            Subject specific state probabilities.
            Each element has shape (n_samples, n_modes).
        means : np.ndarray
            Subject specific means. Shape is (n_subjects, n_modes, n_channels).
        covariances : np.ndarray
            Subject specific covariances.
            Shape is (n_subjects, n_modes, n_channels, n_channels).
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

        # Fine tune the model for each subject
        alpha = []
        means = []
        covariances = []
        with set_logging_level(_logger, logging.WARNING), self.set_trainable(
            ["hid_state_inf"], False
        ):
            for subject in trange(training_data.n_arrays, desc="Subject fine tuning"):
                # Train on this subject
                with training_data.set_keep(subject):
                    self.fit(training_data, verbose=0)
                    a = self.get_alpha(training_data, concatenate=True, verbose=0)

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

    def dual_estimation(self, training_data):
        """Dual estimation for subject specific observation model parameters.

        Parameters
        ----------
        training_data : osl_dynamics.data.Data
            Training dataset.

        Returns
        -------
        means : np.ndarray
            Subject specific means. Shape is (n_subjects, n_modes, n_channels).
        covariances : np.ndarray
            Subject specific covariances.
            Shape is (n_subjects, n_modes, n_channels, n_channels).
        """
        _logger.info("Dual estimation")
        means = []
        covariances = []
        x = training_data.time_series()
        alpha = self.get_alpha(training_data)
        for subject in trange(training_data.n_arrays, desc="Dual estimation"):
            if self.config.learn_means:
                means_ = np.sum(
                    np.expand_dims(alpha[subject], axis=2)
                    * np.expand_dims(x[subject], axis=1),
                    axis=0,
                ) / np.expand_dims(np.sum(alpha[subject], axis=0), axis=1)
            else:
                means_ = np.zeros((self.config.n_states, self.config.n_channels))

            if self.config.learn_covariances:
                covariances_ = np.zeros(
                    (
                        self.config.n_states,
                        self.config.n_channels,
                        self.config.n_channels,
                    )
                )
                for k in range(self.config.n_states):
                    diff = x[subject] - means_[k]
                    covariances_[k] = np.sum(
                        alpha[subject][:, k][:, None, None]
                        * np.matmul(diff[:, :, None], diff[:, None, :]),
                        axis=0,
                    ) / np.sum(alpha[subject][:, k], axis=0)
            else:
                covariances_ = np.stack(
                    [np.eye(self.config.n_channels)] * self.config.n_states
                )

            means.append(means_)
            covariances.append(covariances_)

        return np.array(means), np.array(covariances)

    def get_training_time_series(
        self,
        training_data,
        prepared=True,
        concatenate=False,
    ):
        """Get the time series used for training from a Data object.

        Parameters
        ----------
        training_data : osl_dynamics.data.Data
            Data object.
        prepared : bool, optional
            Should we return the prepared data? If not, we return the raw data.
        concatenate : bool, optional
            Should we concatenate the data for each subject?

        Returns
        -------
        training_data : np.ndarray or list
            Training data time series.
        """
        return training_data.trim_time_series(
            self.config.sequence_length,
            prepared=prepared,
            concatenate=concatenate,
        )


def _model_structure(config):
    # Inputs
    data = layers.Input(
        shape=(config.sequence_length, config.n_channels),
        name="data",
    )

    # Static loss scaling factor
    static_loss_scaling_factor_layer = StaticLossScalingFactorLayer(
        name="static_loss_scaling_factor"
    )
    static_loss_scaling_factor = static_loss_scaling_factor_layer(data)

    # Observation model
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
    ll_layer = SeparateLogLikelihoodLayer(
        config.n_states, config.covariances_epsilon, name="ll"
    )

    mu = means_layer(data, static_loss_scaling_factor=static_loss_scaling_factor)
    D = covs_layer(data, static_loss_scaling_factor=static_loss_scaling_factor)
    ll = ll_layer([data, mu, D, None])

    # Hidden state inference
    hidden_state_inference_layer = HiddenStateInferenceLayer(
        config.n_states,
        config.initial_trans_prob,
        config.learn_trans_prob,
        name="hid_state_inf",
    )

    gamma, xi = hidden_state_inference_layer(ll)

    # Loss
    ll_loss_layer = SumLogLikelihoodLossLayer(name="ll_loss")

    ll_loss = ll_loss_layer([ll, gamma])

    return tf.keras.Model(inputs=data, outputs=[ll_loss, gamma, xi], name="HMM")
