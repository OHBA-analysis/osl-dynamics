"""Subject Embedded DyNeMo (SE-DyNeMo) observation model."""

from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from ohba_models.models.mod_base import BaseModelConfig, ModelBase

from ohba_models.inference.layers import (
    LogLikelihoodLossLayer,
    MeanVectorsLayer,
    CovarianceMatricesLayer,
    SubjectMeansCovsLayer,
    MixSubjectEmbeddingParametersLayer,
    TFRangeLayer,
)


@dataclass
class Config(BaseModelConfig):
    """Settings for DyNeMo observation model.

    Parameters
    ----------
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

    n_subjects : int
        Number of subjects.
    subject_embedding_dim : int
        Number of dimensions for the subject embedding.
    mode_embedding_dim : int
        Number of dimensions for the mode embedding in the spatial maps encoder.

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

    # Parameters specific to subject embedding model
    n_subjects: int = None
    subject_embedding_dim: int = None
    mode_embedding_dim: int = None

    def __post_init__(self):
        self.validate_observation_model_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()
        self.validate_subject_embedding_parameters()

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")

    def validate_subject_embedding_parameters(self):
        if (
            self.n_subjects is None
            or self.subject_embedding_dim is None
            or self.mode_embedding_dim is None
        ):
            raise ValueError(
                "n_subjects, subject_embedding_dim and mode_embedding_dim must be passed."
            )


class Model(ModelBase):
    """SE-DyNeMo observation model class.
    Parameters

    ----------
    config : ohba_models.models.dynemo_obs.Config
    """

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_group_means_covariances(self):
        """Get the group means and covariances of each mode

        Returns
        -------
        means : np.ndarray
            Mode means for the group.
        covariances : np.ndarray
            Mode covariances for the group.
        """
        return get_group_means_covariances(self.model)

    def get_subject_embeddings(self):
        """Get the subject embedding vectors

        Returns
        -------
        subject_embeddings : np.ndarray
            Embedding vectors for subjects. Shape is (n_subjects, subject_embedding_dim)
        """
        return get_subject_embeddings(self.model)

    def get_subject_means_covariances(self):
        """Get the means and covariances for each subject

        Returns
        -------
        subject_means : np.ndarray
            Mode means for each subject
        subject_covs : np.ndarray
            Mode covariances for each subject
        """
        return get_subject_means_covariances(self.model)

    def set_group_means(self, means, update_initializer=True):
        """Set the group means of each mode.

        Parameters
        ----------
        means : np.ndarray
            Mode means.
        update_initializer : bool
            Do we want to use the passed means when we re-initialize
            the model?
        """
        set_group_means(self.model, means, update_initializer)

    def set_group_covariances(self, covariances, update_initializer=True):
        """Set the group covariances of each mode.

        Parameters
        ----------
        covariances : np.ndarray
            Mode covariances.
        update_initializer : bool
            Do we want to use the passed covariances when we re-initialize
            the model?
        """
        set_group_covariances(self.model, covariances, update_initializer)


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

    # Subject embedding layer
    subjects_layer = TFRangeLayer(config.n_subjects, name="subjects")
    subject_embedding_layer = layers.Embedding(
        config.n_subjects, config.subject_embedding_dim, name="subject_embeddings"
    )

    # Data flow
    subjects = subjects_layer(data)  # data not used here
    subject_embeddings = subject_embedding_layer(subjects)

    # Definition of layers
    group_means_layer = MeanVectorsLayer(
        config.n_modes,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        name="group_means",
    )
    group_covs_layer = CovarianceMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_covariances,
        config.initial_covariances,
        name="group_covs",
    )
    delta_mu_layer = layers.Dense(config.n_channels, name="delta_mu")
    flattened_delta_D_cholesky_factors_layer = layers.Dense(
        config.n_channels * (config.n_channels + 1) // 2,
        name="flattened_delta_D_cholesky_factors",
    )
    subject_means_covs_layer = SubjectMeansCovsLayer(
        config.n_modes,
        config.n_channels,
        config.n_subjects,
        config.mode_embedding_dim,
        name="subject_means_covs",
    )
    mix_subject_means_covs_layer = MixSubjectEmbeddingParametersLayer(
        name="mix_subject_means_covs"
    )
    ll_loss_layer = LogLikelihoodLossLayer(name="ll_loss")

    # Data flow
    group_mu = group_means_layer(data)  # data not used
    group_D = group_covs_layer(data)  # data not used
    delta_mu = delta_mu_layer(subject_embeddings)
    flattened_delta_D_cholesky_factors = flattened_delta_D_cholesky_factors_layer(
        subject_embeddings
    )
    mu, D = subject_means_covs_layer(
        [group_mu, group_D, delta_mu, flattened_delta_D_cholesky_factors]
    )
    m, C = mix_subject_means_covs_layer([alpha, mu, D, subj_id])
    ll_loss = ll_loss_layer([data, m, C])

    return tf.keras.Model(
        inputs=[data, alpha, subj_id], outputs=[ll_loss], name="SE-DyNeMo-Obs"
    )


def get_group_means_covariances(model):
    group_means_layer = model.get_layer("group_means")
    group_covs_layer = model.get_layer("group_covs")
    return group_means_layer(1).numpy(), group_covs_layer(1).numpy()


def get_subject_embeddings(model):
    subject_embedding_layer = model.get_layer("subject_embeddings")
    n_subjects = subject_embedding_layer.input_dim
    return subject_embedding_layer(np.arange(n_subjects))


def get_subject_means_covariances(model):
    group_means, group_covs = get_group_means_covariances(model)
    subject_embeddings = get_subject_embeddings(model)
    subject_means_covs_layer = model.get_layer("subject_means_covs")
    mu, D = subject_means_covs_layer([group_means, group_covs, subject_embeddings])
    return mu.numpy(), D.numpy()


def set_group_means(model, means, update_initializer=True):
    means = means.astype(np.float32)
    group_means_layer = model.get_layer("group_means")
    layer_weights = group_means_layer.means
    layer_weights.assign(means)

    if update_initializer:
        group_means_layer.initial_value = means


def set_group_covariances(model, covariances, update_initializer=True):
    covariances = covariances.astype(np.float32)
    group_covs_layer = model.get_layer("group_covs")
    layer_weights = group_covs_layer.flattened_cholesky_factors
    flattened_cholesky_factors = group_covs_layer.bijector.inverse(covariances)
    layer_weights.assign(flattened_cholesky_factors)

    if update_initializer:
        group_covs_layer.initial_value = covariances
        group_covs_layer.initial_flattened_cholesky_factors = flattened_cholesky_factors
        grooup_covs_layer.flattened_cholesky_factors_initializer.initial_value = (
            flattened_cholesky_factors
        )
