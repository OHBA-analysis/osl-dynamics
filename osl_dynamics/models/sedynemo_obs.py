"""Subject Embedding Dynamic Network Modes (Se-DyNeMo) observation model.

"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow.keras import layers

import osl_dynamics.data.tf as dtf
from osl_dynamics.models import dynemo_obs
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.inference.layers import (
    add_epsilon,
    LogLikelihoodLossLayer,
    VectorsLayer,
    CovarianceMatricesLayer,
    SubjectDevEmbeddingLayer,
    SubjectMapLayer,
    MixSubjectEmbeddingParametersLayer,
    TFRangeLayer,
    ZeroLayer,
    InverseCholeskyLayer,
    SampleNormalDistributionLayer,
    SubjectMapKLDivergenceLayer,
    KLLossLayer,
    MultiLayerPerceptronLayer,
)


@dataclass
class Config(BaseModelConfig):
    """Settings for DyNeMo observation model.

    Parameters
    ----------
    model_name : str
        Model name.
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the generative model.
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
    means_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for group mean vectors.
    covariances_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for group covariance matrices.

    n_subjects : int
        Number of subjects.
    subject_embedding_dim : int
        Number of dimensions for the subject embedding.
    mode_embedding_dim : int
        Number of dimensions for the mode embedding in the spatial maps encoder.

    dev_inf_n_layers : int
        Number of layers for the inference MLP for deviations.
    dev_inf_n_units : int
        Number of units for the inference MLP for deviations.
    dev_inf_normalization : str
        Type of normalization for the inference MLP for deviations.
        Either None, 'batch' or 'layer'.
    dev_inf_activation : str
        Type of activation to use for the inference MLP for deviations.
        E.g. 'relu', 'sigmoid', 'tanh', etc.
    dev_inf_dropout : float
        Dropout rate for the inference MLP for deviations.
    dev_mod_n_layers : int
        Number of layers for the model MLP for deviations.
    dev_mod_n_units : int
        Number of units for the model MLP for deviations.
    dev_mod_normalization : str
        Type of normalization for the model MLP for deviations.
    dev_mod_activation : str
        Type of activation to use for the model MLP for deviations.
    dev_mod_dropout : float
        Dropout rate for the model MLP for deviations.

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

    model_name: str = "SE-DyNeMo-Obs"

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
    subject_embedding_dim: int = None
    mode_embedding_dim: int = None

    dev_inf_n_layers: int = 0
    dev_inf_n_units: int = None
    dev_inf_normalization: str = None
    dev_inf_activation: str = None
    dev_inf_dropout: float = 0.0
    dev_mod_n_layers: int = 0
    dev_mod_n_units: int = None
    dev_mod_normalization: str = None
    dev_mod_activation: str = None
    dev_mod_dropout: float = 0.0

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
            or self.subject_embedding_dim is None
            or self.mode_embedding_dim is None
        ):
            raise ValueError(
                "n_subjects, subject_embedding_dim and mode_embedding_dim must be passed."
            )

        if self.dev_inf_n_layers != 0 and self.dev_inf_n_units is None:
            raise ValueError("Please pass dev_inf_n_units.")

        if self.dev_mod_n_layers != 0 and self.dev_mod_n_units is None:
            raise ValueError("Plase pass dev_mod_n_units.")


class Model(ModelBase):
    """SE-DyNeMo observation model class.

    Parameters
    ----------
    config : osl_dynamics.models.dynemo_obs.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_group_means_covariances(self):
        """Get the group means and covariances of each mode.

        Returns
        -------
        means : np.ndarray
            Mode means for the group. Shape is (n_modes, n_channels).
        covariances : np.ndarray
            Mode covariances for the group. Shape is (n_modes, n_channels, n_channels).
        """
        return get_group_means_covariances(self.model)

    def get_subject_embeddings(self):
        """Get the subject embedding vectors.

        Returns
        -------
        subject_embeddings : np.ndarray
            Embedding vectors for subjects. Shape is (n_subjects, subject_embedding_dim).
        """
        return get_subject_embeddings(self.model)

    def get_mode_embeddings(self):
        """Get the mode spatial map embeddings.

        Returns
        -------
        means_mode_embeddings : np.ndarray
            Mode embeddings for means. Shape is (n_modes, mode_embedding_dim).
        covs_mode_embeddings : np.ndarray
            Mode embeddings for covs. Shape is (n_modes, mode_embedding_dim).
        """
        return get_mode_embeddings(self.model)

    def get_concatenated_embeddings(self):
        """Get the concatenated embedding vectors of deviations.

        Returns
        -------
        means_embedding : np.ndarray
            Embedding vectors for the mean deviations.
            Shape is (n_subjects, n_modes, subject_embedding_dim + mode_embedding_dim).
        covs_embedding : np.ndarray
            Embedding vectors for the covs deviations.
            Shape is (n_subjects, n_modes, subject_embedding_dim + mode_embedding_dim).
        """
        return get_concatenated_embeddings(self.model)

    def get_subject_dev(self):
        """Get the subject specific deviations of means and covs from the group.

        Returns
        -------
        means_dev : np.ndarray
            Deviation of means from the group. Shape is (n_subjects, n_modes, n_channels).
        covs_dev : np.ndarray
            Deviation of Cholesky factor of covs from the group.
            Shaoe is (n_subjects, n_modes, n_channels * (n_channels + 1) // 2).
        """
        return get_subject_dev(self.model)

    def get_subject_means_covariances(self):
        """Get the means and covariances for each subject

        Returns
        -------
        subject_means : np.ndarray
            Mode means for each subject. Shape is (n_subjects, n_modes, n_channels).
        subject_covs : np.ndarray
            Mode covariances for each subject. Shape is (n_subjects, n_modes, n_channels, n_channels).
        """
        return get_subject_means_covariances(self.model)

    def set_regularizers(self, training_dataset):
        """Set the means and covariances regularizer based on the training data.

        A multivariate normal prior is applied to the mean vectors with mu = 0,
        sigma=diag((range / 2)**2) and an inverse Wishart prior is applied to the
        covariances matrices with nu=n_channels - 1 + 0.1 and psi=diag(range).

        Parameters
        ----------
        training_data : tensorflow.data.Dataset
            Training dataset.
        """
        if self.config.learn_means:
            dynemo_obs.set_means_regularizer(
                self.model, training_dataset, layer_name="group_means"
            )

        if self.config.learn_covariances:
            dynemo_obs.set_covariances_regularizer(
                self.model, training_dataset, layer_name="group_covs"
            )

    def set_bayesian_deviation_parameters(self, training_dataset):
        """Set the correct scaling for KL loss between deviation posterior and prior."""
        n_batches = dtf.get_n_batches(training_dataset)
        learn_means = self.config.learn_means
        learn_covariances = self.config.learn_covariances
        set_bayesian_kl_scaling(self.model, n_batches, learn_means, learn_covariances)

    def set_group_means(self, group_means, update_initializer=True):
        """Set the group means of each mode.

        Parameters
        ----------
        group_means : np.ndarray
            Mode means.
        update_initializer : bool
            Do we want to use the passed group means when we re-initialize the model?
        """
        dynemo_obs.set_means(
            self.model, group_means, update_initializer, layer_name="group_means"
        )

    def set_group_covariances(self, group_covariances, update_initializer=True):
        """Set the group covariances of each mode.

        Parameters
        ----------
        group_covariances : np.ndarray
            Mode covariances.
        update_initializer : bool
            Do we want to use the passed group covariances when we re-initialize
            the model?
        """
        dynemo_obs.set_covariances(
            self.model, group_covariances, update_initializer, layer_name="group_covs"
        )


def _model_structure(config):
    # Layers for inputs
    data = layers.Input(shape=(config.sequence_length, config.n_channels), name="data")
    alpha = layers.Input(shape=(config.sequence_length, config.n_modes), name="alpha")
    subj_id = layers.Input(shape=(config.sequence_length,), name="subj_id")

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each mode as the observation model.
    # - Each subject has their own mean vectors and covariance matrices for each mode.
    #   They are near the group means and covariances.
    # - We calculate the likelihood of generating the training data with alpha
    #   and the observation model.

    # Definition of layers

    # Subject embedding layer
    subjects_layer = TFRangeLayer(config.n_subjects, name="subjects")
    subject_embedding_layer = layers.Embedding(
        config.n_subjects, config.subject_embedding_dim, name="subject_embeddings"
    )

    group_means_layer = VectorsLayer(
        config.n_modes,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        config.means_regularizer,
        name="group_means",
    )
    group_covs_layer = CovarianceMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_covariances,
        config.initial_covariances,
        config.covariances_epsilon,
        config.covariances_regularizer,
        name="group_covs",
    )
    means_mode_embedding_layer = layers.Dense(
        config.mode_embedding_dim,
        name="means_mode_embedding",
    )
    covs_mode_embedding_layer = layers.Dense(
        config.mode_embedding_dim,
        name="covs_mode_embedding",
    )
    means_concat_embedding_layer = SubjectDevEmbeddingLayer(
        config.n_modes,
        config.n_channels,
        config.n_subjects,
        name="means_concat_embedding",
    )
    covs_concat_embedding_layer = SubjectDevEmbeddingLayer(
        config.n_modes,
        config.n_channels,
        config.n_subjects,
        name="covs_concat_embedding",
    )

    # Inference part of the deviation
    means_dev_inf_input_layer = MultiLayerPerceptronLayer(
        config.dev_inf_n_layers,
        config.dev_inf_n_units,
        config.dev_inf_normalization,
        config.dev_inf_activation,
        config.dev_inf_dropout,
        name="means_dev_inf_input",
    )
    means_dev_inf_mu_layer = layers.Dense(config.n_channels, name="means_dev_inf_mu")
    means_dev_inf_sigma_layer = layers.Dense(
        config.n_channels, activation="softplus", name="means_dev_inf_sigma"
    )
    if config.learn_means:
        means_dev_layer = SampleNormalDistributionLayer(
            config.theta_std_epsilon, name="means_dev"
        )
    else:
        means_dev_layer = ZeroLayer(
            shape=(config.n_subjects, config.n_modes, config.n_channels),
            name="means_dev",
        )

    covs_dev_inf_input_layer = MultiLayerPerceptronLayer(
        config.dev_inf_n_layers,
        config.dev_inf_n_units,
        config.dev_inf_normalization,
        config.dev_inf_activation,
        config.dev_inf_dropout,
        name="covs_dev_inf_input",
    )
    covs_dev_inf_mu_layer = layers.Dense(
        config.n_channels * (config.n_channels + 1) // 2, name="covs_dev_inf_mu"
    )
    covs_dev_inf_sigma_layer = layers.Dense(
        config.n_channels * (config.n_channels + 1) // 2,
        activation="softplus",
        name="covs_dev_inf_sigma",
    )
    if config.learn_covariances:
        covs_dev_layer = SampleNormalDistributionLayer(
            config.theta_std_epsilon, name="covs_dev"
        )
    else:
        covs_dev_layer = ZeroLayer(
            shape=(
                config.n_subjects,
                config.n_modes,
                config.n_channels * (config.n_channels + 1) // 2,
            ),
            name="covs_dev",
        )

    subject_means_layer = SubjectMapLayer(
        "means", config.covariances_epsilon, name="subject_means"
    )
    subject_covs_layer = SubjectMapLayer(
        "covariances", config.covariances_epsilon, name="subject_covs"
    )
    mix_subject_means_covs_layer = MixSubjectEmbeddingParametersLayer(
        name="mix_subject_means_covs"
    )
    ll_loss_layer = LogLikelihoodLossLayer(config.covariances_epsilon, name="ll_loss")

    # Data flow
    subjects = subjects_layer(data)  # data not used here
    subject_embeddings = subject_embedding_layer(subjects)

    group_mu = group_means_layer(data)  # data not used
    group_D = group_covs_layer(data)  # data not used

    # spatial map embeddings
    means_mode_embedding = means_mode_embedding_layer(group_mu)
    covs_mode_embedding = covs_mode_embedding_layer(
        InverseCholeskyLayer(config.covariances_epsilon)(group_D)
    )

    # Now get the subject specific spatial maps
    means_concat_embedding = means_concat_embedding_layer(
        [subject_embeddings, means_mode_embedding]
    )
    covs_concat_embedding = covs_concat_embedding_layer(
        [subject_embeddings, covs_mode_embedding]
    )

    means_dev_inf_input = means_dev_inf_input_layer(means_concat_embedding)
    means_dev_inf_mu = means_dev_inf_mu_layer(means_dev_inf_input)
    means_dev_inf_sigma = means_dev_inf_sigma_layer(means_dev_inf_input)
    means_dev = means_dev_layer([means_dev_inf_mu, means_dev_inf_sigma])

    covs_dev_inf_input = covs_dev_inf_input_layer(covs_concat_embedding)
    covs_dev_inf_mu = covs_dev_inf_mu_layer(covs_dev_inf_input)
    covs_dev_inf_sigma = covs_dev_inf_sigma_layer(covs_dev_inf_input)
    covs_dev = covs_dev_layer([covs_dev_inf_mu, covs_dev_inf_sigma])

    mu = subject_means_layer([group_mu, means_dev])
    D = subject_covs_layer([group_D, covs_dev])

    # Mix with the mode time course
    m, C = mix_subject_means_covs_layer([alpha, mu, D, subj_id])
    ll_loss = ll_loss_layer([data, m, C])

    # Prior part of deviations
    means_dev_mod_input_layer = MultiLayerPerceptronLayer(
        config.dev_inf_n_layers,
        config.dev_inf_n_units,
        config.dev_inf_normalization,
        config.dev_inf_activation,
        config.dev_inf_dropout,
        name="means_dev_mod_input",
    )
    means_dev_mod_sigma_layer = layers.Dense(
        config.n_channels, activation="softplus", name="means_dev_mod_sigma"
    )
    covs_dev_mod_input_layer = MultiLayerPerceptronLayer(
        config.dev_inf_n_layers,
        config.dev_inf_n_units,
        config.dev_inf_normalization,
        config.dev_inf_activation,
        config.dev_inf_dropout,
        name="covs_dev_mod_input",
    )
    covs_dev_mod_sigma_layer = layers.Dense(
        config.n_channels * (config.n_channels + 1) // 2,
        activation="softplus",
        name="covs_dev_mod_sigma",
    )

    if config.learn_means:
        means_dev_kl_loss_layer = SubjectMapKLDivergenceLayer(
            config.theta_std_epsilon, name="means_dev_kl_loss"
        )
    else:
        means_dev_kl_loss_layer = ZeroLayer((), name="means_dev_kl_loss")

    if config.learn_covariances:
        covs_dev_kl_loss_layer = SubjectMapKLDivergenceLayer(
            config.theta_std_epsilon, name="covs_dev_kl_loss"
        )
    else:
        covs_dev_kl_loss_layer = ZeroLayer((), name="covs_dev_kl_loss")

    kl_loss_layer = KLLossLayer(do_annealing=config.do_kl_annealing, name="kl_loss")

    # Data flow
    means_dev_mod_input = means_dev_mod_input_layer(means_concat_embedding)
    means_dev_mod_sigma = means_dev_mod_sigma_layer(means_dev_mod_input)

    covs_dev_mod_input = covs_dev_mod_input_layer(covs_concat_embedding)
    covs_dev_mod_sigma = covs_dev_mod_sigma_layer(covs_dev_mod_input)

    means_dev_kl_loss = means_dev_kl_loss_layer(
        [data, means_dev_inf_mu, means_dev_inf_sigma, means_dev_mod_sigma]
    )
    covs_dev_kl_loss = covs_dev_kl_loss_layer(
        [data, covs_dev_inf_mu, covs_dev_inf_sigma, covs_dev_mod_sigma]
    )
    kl_loss = kl_loss_layer([means_dev_kl_loss, covs_dev_kl_loss])

    return tf.keras.Model(
        inputs=[data, alpha, subj_id],
        outputs=[ll_loss, kl_loss],
        name="Se_DyNeMo-Obs",
    )


def get_group_means_covariances(model):
    group_means_layer = model.get_layer("group_means")
    group_covs_layer = model.get_layer("group_covs")

    group_means = group_means_layer.vectors
    group_covs = add_epsilon(
        group_covs_layer.bijector(group_covs_layer.flattened_cholesky_factors),
        group_covs_layer.epsilon,
        diag=True,
    )
    return group_means.numpy(), group_covs.numpy()


def get_subject_embeddings(model):
    subject_embedding_layer = model.get_layer("subject_embeddings")
    n_subjects = subject_embedding_layer.input_dim
    return subject_embedding_layer(np.arange(n_subjects)).numpy()


def get_mode_embeddings(model):
    cholesky_bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])

    group_means, group_covs = get_group_means_covariances(model)
    means_mode_embedding_layer = model.get_layer("means_mode_embedding")
    covs_mode_embedding_layer = model.get_layer("covs_mode_embedding")

    means_mode_embedding = means_mode_embedding_layer(group_means)
    covs_mode_embedding = covs_mode_embedding_layer(
        cholesky_bijector.inverse(group_covs)
    )

    return means_mode_embedding.numpy(), covs_mode_embedding.numpy()


def get_concatenated_embeddings(model):
    subject_embeddings = get_subject_embeddings(model)
    means_mode_embedding, covs_mode_embedding = get_mode_embeddings(model)

    means_concat_embedding_layer = model.get_layer("means_concat_embedding")
    covs_concat_embedding_layer = model.get_layer("covs_concat_embedding")

    means_concat_embedding = means_concat_embedding_layer(
        [subject_embeddings, means_mode_embedding]
    )
    covs_concat_embedding = covs_concat_embedding_layer(
        [subject_embeddings, covs_mode_embedding]
    )
    return means_concat_embedding.numpy(), covs_concat_embedding.numpy()


def get_subject_dev(model):
    means_concat_embedding, covs_concat_embedding = get_concatenated_embeddings(model)
    means_dev_inf_input_layer = model.get_layer("means_dev_inf_input")
    covs_dev_inf_input_layer = model.get_layer("covs_dev_inf_input")
    means_dev_inf_layer = model.get_layer("means_dev_inf_mu")
    covs_dev_inf_layer = model.get_layer("covs_dev_inf_mu")

    means_dev_inf_input = means_dev_inf_input_layer(means_concat_embedding)
    covs_dev_inf_input = covs_dev_inf_input_layer(covs_concat_embedding)
    means_dev = means_dev_inf_layer(means_dev_inf_input)
    covs_dev = covs_dev_inf_layer(covs_dev_inf_input)

    return means_dev.numpy(), covs_dev.numpy()


def get_subject_means_covariances(model):
    group_means, group_covs = get_group_means_covariances(model)
    means_dev, covs_dev = get_subject_dev(model)

    subject_means_layer = model.get_layer("subject_means")
    subject_covs_layer = model.get_layer("subject_covs")

    mu = subject_means_layer([group_means, means_dev])
    D = subject_covs_layer([group_covs, covs_dev])
    return mu.numpy(), D.numpy()


def set_bayesian_kl_scaling(model, n_batches, learn_means, learn_covariances):
    if learn_means:
        means_dev_kl_loss_layer = model.get_layer("means_dev_kl_loss")
        means_dev_kl_loss_layer.n_batches = n_batches

    if learn_covariances:
        covs_dev_kl_loss_layer = model.get_layer("covs_dev_kl_loss")
        covs_dev_kl_loss_layer.n_batches = n_batches
