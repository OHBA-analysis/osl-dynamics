"""Subject Embedding Dynamic Network Modes (Se-DyNeMo) observation model.
"""
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow.keras import layers
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase

from osl_dynamics.inference import regularizers
from osl_dynamics.inference.layers import (
    DummyLayer,
    LogLikelihoodLossLayer,
    MeanVectorsLayer,
    CovarianceMatricesLayer,
    SubjectDevEmbeddingLayer,
    SubjectMapLayer,
    MixSubjectEmbeddingParametersLayer,
    TFRangeLayer,
    ZeroLayer,
    InverseCholeskyLayer,
    SampleNormalDistributionLayer,
    ScalarLayer,
    SubjectMapKLDivergenceLayer,
    KLLossLayer,
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
    mode_embedding_activation : str
        Activation of the mode encoders.
    dev_reg : str
        Type of regularisation applied to the deviations.
    dev_reg_strength : float
        Strength of regularisation to deviations.
    dev_bayesian : bool
        Do we want to be fully Bayesian on deviations?.
    learn_dev_mod_sigma : bool
        Do we want to learn the prior std of the deviation.
    initial_dev_mod_sigma : float
        Initial value for prior std of the deviation.
    do_kl_annealing : bool
        Should we do annealing on the KL loss during training?

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
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

    # Parameters specific to subject embedding model
    n_subjects: int = None
    subject_embedding_dim: int = None
    mode_embedding_dim: int = None
    mode_embedding_activation: str = None
    dev_reg: str = None
    dev_reg_strength: float = 0.0
    dev_bayesian: bool = False
    learn_dev_mod_sigma: bool = True
    initial_dev_mod_sigma: float = 1
    do_kl_annealing: bool = False

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
    config : osl_dynamics.models.dynemo_obs.Config
    """

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
        return get_subject_dev(self.model, self.config.dev_bayesian)

    def get_subject_means_covariances(self):
        """Get the means and covariances for each subject

        Returns
        -------
        subject_means : np.ndarray
            Mode means for each subject. Shape is (n_subjects, n_modes, n_channels).
        subject_covs : np.ndarray
            Mode covariances for each subject. Shape is (n_subjects, n_modes, n_channels, n_channels).
        """
        return get_subject_means_covariances(self.model, self.config.dev_bayesian)

    def set_means_regularizer(self, n_batches, training_data=None, mu=None, sigma=None):
        """Set the group means vector regularizer.

        The regularization is equivalent to applying a multivariate normal prior.

        Parameters
        ----------
        n_batches : int
            Number of batches.
        training_data : osl_dynamics.data.Data
            Estimate mu and sigma using the training data instead of specifying it
            explicitly. If training_data is passed, diag(sigma)=((max - min) / 2)**2.
        mu : np.ndarray
            Mean vector of the prior. Shape must be (n_channels,).
        sigma : np.ndarray
            Covariance matrix of the prior. Shape must be (n_channels,n_channels).
        """
        set_means_regularizer(self.model, n_batches, training_data, mu, sigma)

    def set_covariances_regularizer(
        self, n_batches, training_data=None, nu=None, psi=None
    ):
        """Set the group covariance matrices regularizer.

        Parameters
        ----------
        n_batches : int
            Number of batches.
        training_data : osl_dynamics.data.Data
            Estimate nu and psi using the training data instead of specifying it
            explicitly. If training_data is passed, nu=n_channels - 1 + 0.1
            and psi=diag(1 / (max - min)).
        nu : float
            Degrees of freedom of the prior.
        psi : np.ndarray
            Scale matrix of the prior. Shape must be (n_channels, n_channels).
        """
        set_covariances_regularizer(self.model, n_batches, training_data, nu, psi)

    def set_regularizers(self, n_batches, training_data):
        """Set the means and covariances regularizer based on the training data.

        A multivariate normal prior is applied to the mean vectors with mu = 0,
        sigma=diag((range / 2)**2) and an inverse Wishart prior is applied to the
        covariances matrices with nu=n_channels - 1 + 0.1 and psi=diag(1 / range).

        Parameters
        ----------
        n_batches : int
            Number of batches.
        training_data : osl_dynamics.data.Data
            Training dataset.
        """
        if self.config.learn_means:
            self.set_means_regularizer(n_batches, training_data)

        if self.config.learn_covariances:
            self.set_covariances_regularizer(n_batches, training_data)

    def set_bayesian_deviation_parameters(self, n_batches):
        """Set the correct scaling for KL loss between deviation posterior and prior."""
        set_bayesian_kl_scaling(self.model, self.config, n_batches)


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

    group_means_layer = MeanVectorsLayer(
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
        config.covariances_regularizer,
        name="group_covs",
    )
    means_mode_embedding_layer = layers.Dense(
        config.mode_embedding_dim,
        config.mode_embedding_activation,
        name="means_mode_embedding",
    )
    covs_mode_embedding_layer = layers.Dense(
        config.mode_embedding_dim,
        config.mode_embedding_activation,
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
    # ----------------------------------------- #
    # Layers specific to the non Bayesian model #
    # ----------------------------------------- #
    if not config.dev_bayesian:
        if config.learn_means:
            means_dev_layer = layers.Dense(config.n_channels, name="means_dev")
        else:
            means_dev_layer = ZeroLayer(
                shape=(config.n_subjects, config.n_modes, config.n_channels),
                name="means_dev",
            )

        if config.learn_covariances:
            covs_dev_layer = layers.Dense(
                config.n_channels * (config.n_channels + 1) // 2,
                name="covs_dev",
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

        means_dev_reg_layer = DummyLayer(
            config.dev_reg,
            config.dev_reg_strength,
            name="means_dev_reg",
        )
        covs_dev_reg_layer = DummyLayer(
            config.dev_reg,
            config.dev_reg_strength,
            name="covs_dev_reg",
        )
    # ------------------------------------- #
    # Layers specific to the Bayesian model #
    # ------------------------------------- #
    else:
        means_dev_inf_mu_layer = layers.Dense(
            config.n_channels, name="means_dev_inf_mu"
        )
        means_dev_inf_sigma_layer = layers.Dense(
            config.n_channels, activation="softplus", name="means_dev_inf_sigma"
        )
        if config.learn_means:
            means_dev_layer = SampleNormalDistributionLayer(name="means_dev")
        else:
            means_dev_layer = ZeroLayer(
                shape=(config.n_subjects, config.n_modes, config.n_channels),
                name="means_dev",
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
            covs_dev_layer = SampleNormalDistributionLayer(name="covs_dev")
        else:
            covs_dev_layer = ZeroLayer(
                shape=(
                    config.n_subjects,
                    config.n_modes,
                    config.n_channels * (config.n_channels + 1) // 2,
                ),
                name="covs_dev",
            )

    subject_means_layer = SubjectMapLayer("means", name="subject_means")
    subject_covs_layer = SubjectMapLayer("covariances", name="subject_covs")
    mix_subject_means_covs_layer = MixSubjectEmbeddingParametersLayer(
        name="mix_subject_means_covs"
    )
    ll_loss_layer = LogLikelihoodLossLayer(name="ll_loss")

    # Data flow
    subjects = subjects_layer(data)  # data not used here
    subject_embeddings = subject_embedding_layer(subjects)

    group_mu = group_means_layer(data)  # data not used
    group_D = group_covs_layer(data)  # data not used

    # spatial map embeddings
    means_mode_embedding = means_mode_embedding_layer(group_mu)
    covs_mode_embedding = covs_mode_embedding_layer(InverseCholeskyLayer()(group_D))

    # Now get the subject specific spatial maps
    means_concat_embedding = means_concat_embedding_layer(
        [subject_embeddings, means_mode_embedding]
    )
    covs_concat_embedding = covs_concat_embedding_layer(
        [subject_embeddings, covs_mode_embedding]
    )

    if not config.dev_bayesian:
        means_dev = means_dev_layer(means_concat_embedding)
        means_dev = means_dev_reg_layer(means_dev)

        covs_dev = covs_dev_layer(covs_concat_embedding)
        covs_dev = covs_dev_reg_layer(covs_dev)
    else:
        means_dev_inf_mu = means_dev_inf_mu_layer(means_concat_embedding)
        means_dev_inf_sigma = means_dev_inf_sigma_layer(means_concat_embedding)
        means_dev = means_dev_layer([means_dev_inf_mu, means_dev_inf_sigma])

        covs_dev_inf_mu = covs_dev_inf_mu_layer(covs_concat_embedding)
        covs_dev_inf_sigma = covs_dev_inf_sigma_layer(covs_concat_embedding)
        covs_dev = covs_dev_layer([covs_dev_inf_mu, covs_dev_inf_sigma])

    mu = subject_means_layer([group_mu, means_dev])
    D = subject_covs_layer([group_D, covs_dev])

    # Mix with the mode time course
    m, C = mix_subject_means_covs_layer([alpha, mu, D, subj_id])
    ll_loss = ll_loss_layer([data, m, C])

    if not config.dev_bayesian:
        return tf.keras.Model(
            inputs=[data, alpha, subj_id], outputs=[ll_loss], name="Se-DyNeMo-Obs"
        )
    else:
        # Layers for dev prior
        dev_mod_sigma_layer = ScalarLayer(
            config.learn_dev_mod_sigma,
            config.initial_dev_mod_sigma,
            name="dev_mod_sigma",
        )
        if config.learn_means:
            means_dev_kl_loss_layer = SubjectMapKLDivergenceLayer(
                name="means_dev_kl_loss"
            )
        else:
            means_dev_kl_loss_layer = ZeroLayer((), name="means_dev_kl_loss")

        if config.learn_covariances:
            covs_dev_kl_loss_layer = SubjectMapKLDivergenceLayer(
                name="covs_dev_kl_loss"
            )
        else:
            covs_dev_kl_loss_layer = ZeroLayer((), name="covs_dev_kl_loss")

        kl_loss_layer = KLLossLayer(do_annealing=config.do_kl_annealing, name="kl_loss")

        # Data flow
        dev_mod_sigma = dev_mod_sigma_layer(data)  # Data not used
        means_dev_kl_loss = means_dev_kl_loss_layer(
            [data, means_dev_inf_mu, means_dev_inf_sigma, dev_mod_sigma]
        )
        covs_dev_kl_loss = covs_dev_kl_loss_layer(
            [data, covs_dev_inf_mu, covs_dev_inf_sigma, dev_mod_sigma]
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
    return group_means_layer(1).numpy(), group_covs_layer(1).numpy()


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


def get_subject_dev(model, dev_bayesian):
    means_concat_embedding, covs_concat_embedding = get_concatenated_embeddings(model)
    means_dev_layer = (
        model.get_layer("means_dev_inf_mu")
        if dev_bayesian
        else model.get_layer("means_dev")
    )
    covs_dev_layer = (
        model.get_layer("covs_dev_inf_mu")
        if dev_bayesian
        else model.get_layer("covs_dev")
    )

    means_dev = means_dev_layer(means_concat_embedding)
    covs_dev = covs_dev_layer(covs_concat_embedding)

    return means_dev.numpy(), covs_dev.numpy()


def get_subject_means_covariances(model, dev_bayesian):
    group_means, group_covs = get_group_means_covariances(model)
    means_dev, covs_dev = get_subject_dev(model, dev_bayesian)

    subject_means_layer = model.get_layer("subject_means")
    subject_covs_layer = model.get_layer("subject_covs")

    mu = subject_means_layer([group_means, means_dev])
    D = subject_covs_layer([group_covs, covs_dev])
    return mu.numpy(), D.numpy()


def set_means_regularizer(model, n_batches, training_data=None, mu=None, sigma=None):
    if training_data is None:
        if mu is None or sigma is None:
            raise ValueError(
                "Either prior parameters (mu, sigma) or training_data must be passed."
            )
    else:
        ts = training_data.time_series(concatenate=True)
        n_channels = ts.shape[1]
        range_ = np.amax(ts, axis=0) - np.amin(ts, axis=0)
        sigma = np.diag((range_ / 2) ** 2)
        mu = np.zeros(n_channels, dtype=np.float32)

    means_layer = model.get_layer("group_means")
    means_layer.regularizer = regularizers.MultivariateNormal(mu, sigma, n_batches)


def set_covariances_regularizer(
    model, n_batches, training_data=None, nu=None, psi=None
):
    if training_data is None:
        if nu is None or psi is None:
            raise ValueError("Both nu and psi must be passed.")

    else:
        ts = training_data.time_series(concatenate=True)
        n_channels = ts.shape[1]
        nu = n_channels - 1 + 0.1
        range_ = np.amax(ts, axis=0) - np.amin(ts, axis=0)
        psi = np.diag(1 / range_)

    covs_layer = model.get_layer("group_covs")
    covs_layer.regularizer = regularizers.InverseWishart(nu, psi, n_batches)


def set_bayesian_kl_scaling(model, config, n_batches):
    if config.learn_means:
        means_dev_kl_loss_layer = model.get_layer("means_dev_kl_loss")
        means_dev_kl_loss_layer.n_batches = n_batches

    if config.learn_covariances:
        covs_dev_kl_loss_layer = model.get_layer("covs_dev_kl_loss")
        covs_dev_kl_loss_layer.n_batches = n_batches
