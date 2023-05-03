"""Subject Embedding HMM.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf
from numba.core.errors import NumbaWarning
from tensorflow.keras import backend, layers, utils, initializers
from tqdm.auto import trange

import osl_dynamics.data.tf as dtf
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
from osl_dynamics.models.hmm import Config as HMMConfig, Model as HMMModel
from osl_dynamics.models import dynemo_obs, sedynemo_obs

_logger = logging.getLogger("osl-dynamics")

warnings.filterwarnings("ignore", category=NumbaWarning)


@dataclass
class Config(HMMConfig):
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

    do_kl_annealing : bool
        Should we use KL annealing during training?
    kl_annealing_curve : str
        Type of KL annealing curve. Either 'linear' or 'tanh'.
    kl_annealing_sharpness : float
        Parameter to control the shape of the annealing curve if
        kl_annealing_curve='tanh'.
    n_kl_annealing_epochs : int
        Number of epochs to perform KL annealing.
    """

    model_name: str = "SE-HMM"

    # Parameters specific to subject embedding model
    n_subjects: int = None
    subject_embeddings_dim: int = None
    mode_embeddings_dim: int = None

    dev_n_layers: int = 0
    dev_n_units: int = None
    dev_normalization: str = None
    dev_activation: str = None
    dev_dropout: float = 0.0

    # KL annealing parameters
    do_kl_annealing: bool = False
    kl_annealing_curve: Literal["linear", "tanh"] = None
    kl_annealing_sharpness: float = None
    n_kl_annealing_epochs: int = None

    def __post_init__(self):
        super().__post_init__()
        self.validate_subject_embedding_parameters()
        self.validate_kl_annealing_parameters()

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

    def validate_kl_annealing_parameters(self):
        if self.do_kl_annealing:
            if self.kl_annealing_curve is None:
                raise ValueError(
                    "If we are performing KL annealing, "
                    "kl_annealing_curve must be passed."
                )

            if self.kl_annealing_curve not in ["linear", "tanh"]:
                raise ValueError("KL annealing curve must be 'linear' or 'tanh'.")

            if self.kl_annealing_curve == "tanh":
                if self.kl_annealing_sharpness is None:
                    raise ValueError(
                        "kl_annealing_sharpness must be passed if "
                        + "kl_annealing_curve='tanh'."
                    )

                if self.kl_annealing_sharpness < 0:
                    raise ValueError("KL annealing sharpness must be positive.")

            if self.n_kl_annealing_epochs is None:
                raise ValueError(
                    "If we are performing KL annealing, "
                    + "n_kl_annealing_epochs must be passed."
                )

            if self.n_kl_annealing_epochs < 1:
                raise ValueError(
                    "Number of KL annealing epochs must be greater than zero."
                )


class Model(HMMModel):
    """Subject Embedding HMM class.

    Parameters
    ----------
    config : osl_dynamics.models.sehmm.Config
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

        # Set bayesian KL scaling
        self.set_bayesian_kl_scaling(dataset)

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

            # Update KL annealing factor
            if self.config.do_kl_annealing:
                self._update_kl_annealing_factor(n)

        if use_tqdm:
            _range.close()

        return history

    def _update_kl_annealing_factor(self, epoch, n_cycles=1):
        """Update the KL annealing factor."""
        n_epochs_one_cycle = self.config.n_kl_annealing_epochs // n_cycles
        if epoch < self.config.n_kl_annealing_epochs:
            epoch = epoch % n_epochs_one_cycle
            if self.config.kl_annealing_curve == "tanh":
                new_value = (
                    0.5
                    * np.tanh(
                        self.config.kl_annealing_sharpness
                        * (epoch - 0.5 * n_epochs_one_cycle)
                        / n_epochs_one_cycle
                    )
                    + 0.5
                )
            elif self.config.kl_annealing_curve == "linear":
                new_value = epoch / n_epochs_one_cycle
        else:
            new_value = 1.0

        kl_loss_layer = self.model.get_layer("kl_loss")
        kl_loss_layer.annealing_factor.assign(new_value)

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
        # Make a TensorFlow Dataset
        training_data = self.make_dataset(training_data, concatenate=True, subj_id=True)

        return super().random_subset_initialization(
            training_data, n_epochs, n_init, take, **kwargs
        )

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
        # Make a TensorFlow Dataset
        training_dataset = self.make_dataset(
            training_data, concatenate=True, subj_id=True
        )

        return super().random_state_time_course_initialization(
            training_dataset, n_epochs, n_init, take, **kwargs
        )

    def get_group_covariances(self):
        """Get the covariances of each state.

        Returns
        -------
        covariances : np.ndarray
            State covariances. Shape is (n_states, n_channels, n_channels).
        """
        return sedynemo_obs.get_group_means_covariances(self.model)[1]

    def get_group_means_covariances(self):
        """Get the means and covariances of each state.

        Returns
        -------
        means : np.ndarary
            Group level state means.
        covariances : np.ndarray
            Group level state covariances.
        """
        return sedynemo_obs.get_group_means_covariances(self.model)

    def get_subject_means_covariances(self, subject_embeddings=None, n_neighbours=2):
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
        return sedynemo_obs.get_subject_means_covariances(
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

    def set_means(self, means, update_initializer=True):
        """Wrapper of set_group_means."""
        self.set_group_means(means, update_initializer)

    def set_covariances(self, covariances, update_initializer=True):
        """Wrapper of set_group_covariances."""
        self.set_group_covariances(covariances, update_initializer)

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
            axis=-1, name="norm_means_dev_map"
        )

        means_dev_mag_inf_alpha_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_states, 1),
            learn=config.learn_means,
            initializer=initializers.TruncatedNormal(mean=20, stddev=10),
            name="means_dev_mag_inf_alpha_input",
        )
        means_dev_mag_inf_alpha_layer = layers.Activation(
            "softplus", name="means_dev_mag_inf_alpha"
        )
        means_dev_mag_inf_beta_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_states, 1),
            learn=config.learn_means,
            initializer=initializers.TruncatedNormal(mean=100, stddev=20),
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
            axis=-1, name="norm_covs_dev_map"
        )

        covs_dev_mag_inf_alpha_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_states, 1),
            learn=config.learn_covariances,
            initializer=initializers.TruncatedNormal(mean=20, stddev=10),
            name="covs_dev_mag_inf_alpha_input",
        )
        covs_dev_mag_inf_alpha_layer = layers.Activation(
            "softplus", name="covs_dev_mag_inf_alpha"
        )
        covs_dev_mag_inf_beta_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_states, 1),
            learn=config.learn_covariances,
            initializer=initializers.TruncatedNormal(mean=100, stddev=20),
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
    kl_loss_layer = KLLossLayer(do_annealing=True, name="kl_loss")

    # Data flow
    kl_loss = kl_loss_layer([means_dev_mag_kl_loss, covs_dev_mag_kl_loss])

    return tf.keras.Model(inputs=inputs, outputs=[ll_loss, kl_loss], name="SE-HMM-Obs")
