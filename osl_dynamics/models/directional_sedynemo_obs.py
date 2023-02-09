"""Subject Embedding Dynamic Network Modes (Se-DyNeMo) observation model.

"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow.keras import layers, initializers

import osl_dynamics.data.tf as dtf
from osl_dynamics.models import dynemo_obs
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.inference.layers import (
    add_epsilon,
    LogLikelihoodLossLayer,
    LearnableTensorLayer,
    VectorsLayer,
    CovarianceMatricesLayer,
    ConcatEmbeddingsLayer,
    SubjectMapLayer,
    MixSubjectSpecificParametersLayer,
    TFRangeLayer,
    ZeroLayer,
    InverseCholeskyLayer,
    SampleNormalDistributionLayer,
    StaticKLDivergenceLayer,
    KLLossLayer,
    MultiLayerPerceptronLayer,
    StandardizationLayer,
)
import osl_dynamics.inference.initializers as osld_initializers


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
    subject_embeddings_dim : int
        Number of dimensions for the subject embeddings.
    mode_embeddings_dim : int
        Number of dimensions for the mode embeddings in the spatial maps encoder.

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
    subject_embeddings_dim: int = None
    mode_embeddings_dim: int = None

    dev_n_layers: int = 0
    dev_n_units: int = None
    dev_normalization: str = None
    dev_activation: str = None
    dev_dropout: float = 0.0

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
    """Observation model for directional SE-DyNeMo

    Parameters
    ----------
    config : osl_dynamics.models.directional_sedynemo_obs.Config
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
            Embedding vectors for subjects. Shape is (n_subjects, subject_embeddings_dim).
        """
        return get_subject_embeddings(self.model)

    def get_subject_means_covariances(self):
        """Get the means and covariances for each subject

        Returns
        -------
        subject_means : np.ndarray
            Mode means for each subject. Shape is (n_subjects, n_modes, n_channels).
        subject_covs : np.ndarray
            Mode covariances for each subject. Shape is (n_subjects, n_modes, n_channels, n_channels).
        """
        return get_subject_means_covariances(
            self.model, self.config.learn_means, self.config.learn_covariances
        )

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
                self.model,
                self.config.covariances_epsilon,
                training_dataset,
                layer_name="group_covs",
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

    # Subject embedding layers
    subjects_layer = TFRangeLayer(config.n_subjects, name="subjects")
    subject_embeddings_layer = layers.Embedding(
        config.n_subjects, config.subject_embeddings_dim, name="subject_embeddings"
    )

    # Group level observation model parameters
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
            config.n_modes,
            config.n_channels,
            config.n_subjects,
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
        norm_means_dev_map_layer = StandardizationLayer(
            axis=-1, name="norm_means_dev_map"
        )

        means_dev_mag_mod_sigma_layer = layers.Dense(
            1, activation="softplus", name="means_dev_mag_mod_sigma"
        )
        means_dev_mag_inf_mu_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_modes, 1),
            learn=config.learn_means,
            initializer=initializers.TruncatedNormal(mean=0, stddev=0.02),
            name="means_dev_mag_inf_mu",
        )
        means_dev_mag_inf_sigma_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_modes, 1),
            learn=config.learn_means,
            initializer=osld_initializers.SoftplusNormalInitializer(mean=0, std=0.02),
            name="means_dev_mag_inf_sigma_input",
        )
        means_dev_mag_inf_sigma_layer = layers.Activation(
            "softplus", name="means_dev_mag_inf_sigma"
        )
        means_dev_mag_input_layer = SampleNormalDistributionLayer(
            config.covariances_epsilon, name="means_dev_mag_input"
        )
        means_dev_mag_layer = layers.Activation("softplus", name="means_dev_mag")

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
        means_dev_mag_mod_sigma = means_dev_mag_mod_sigma_layer(means_concat_embeddings)
        means_dev_mag_inf_mu = means_dev_mag_inf_mu_layer(data)
        means_dev_mag_inf_sigma_input = means_dev_mag_inf_sigma_input_layer(data)
        means_dev_mag_inf_sigma = means_dev_mag_inf_sigma_layer(
            means_dev_mag_inf_sigma_input
        )
        means_dev_mag_input = means_dev_mag_input_layer(
            [means_dev_mag_inf_mu, means_dev_mag_inf_sigma]
        )
        means_dev_mag = means_dev_mag_layer(means_dev_mag_input)
        means_dev = means_dev_layer([means_dev_mag, norm_means_dev_map])
    else:
        means_dev_layer = ZeroLayer(
            shape=(config.n_subjects, config.n_modes, config.n_channels),
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
            config.n_modes,
            config.n_channels,
            config.n_subjects,
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
        norm_covs_dev_map_layer = StandardizationLayer(
            axis=-1, name="norm_covs_dev_map"
        )

        covs_dev_mag_mod_sigma_layer = layers.Dense(
            1, activation="softplus", name="covs_dev_mag_mod_sigma"
        )
        covs_dev_mag_inf_mu_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_modes, 1),
            learn=config.learn_covariances,
            initializer=initializers.TruncatedNormal(mean=0, stddev=0.02),
            name="covs_dev_mag_inf_mu_layer",
        )
        covs_dev_mag_inf_sigma_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_modes, 1),
            learn=config.learn_covariances,
            initializer=osld_initializers.SoftplusNormalInitializer(mean=0, std=0.02),
            name="covs_dev_mag_inf_sigma_input",
        )
        covs_dev_mag_inf_sigma_layer = layers.Activation(
            "softplus", name="covs_dev_mag_inf_sigma"
        )
        covs_dev_mag_input_layer = SampleNormalDistributionLayer(
            config.covariances_epsilon, name="covs_dev_mag_input"
        )
        covs_dev_mag_layer = layers.Activation("softplus", name="covs_dev_mag")
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
        covs_dev_mag_mod_sigma = covs_dev_mag_mod_sigma_layer(covs_concat_embeddings)
        covs_dev_mag_inf_mu = covs_dev_mag_inf_mu_layer(data)
        covs_dev_mag_inf_sigma_input = covs_dev_mag_inf_sigma_input_layer(data)
        covs_dev_mag_inf_sigma = covs_dev_mag_inf_sigma_layer(
            covs_dev_mag_inf_sigma_input
        )
        covs_dev_mag_input = covs_dev_mag_input_layer(
            [covs_dev_mag_inf_mu, covs_dev_mag_inf_sigma]
        )
        covs_dev_mag = covs_dev_mag_layer(covs_dev_mag_input)
        covs_dev = covs_dev_layer([covs_dev_mag, norm_covs_dev_map])
    else:
        covs_dev_layer = ZeroLayer(
            shape=(
                config.n_subjects,
                config.n_modes,
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
    mix_subject_means_covs_layer = MixSubjectSpecificParametersLayer(
        name="mix_subject_means_covs"
    )
    ll_loss_layer = LogLikelihoodLossLayer(config.covariances_epsilon, name="ll_loss")

    # Data flow
    m, C = mix_subject_means_covs_layer([alpha, mu, D, subj_id])
    ll_loss = ll_loss_layer([data, m, C])

    # ---------
    # KL losses

    if config.learn_means:
        # Layer definitions
        means_dev_mag_mod_sigma_layer = layers.Dense(
            1, activation="softplus", name="means_dev_mag_mod_sigma"
        )
        means_dev_mag_kl_loss_layer = StaticKLDivergenceLayer(
            config.covariances_epsilon, name="means_dev_mag_kl_loss"
        )

        # Data flow
        means_dev_mag_mod_sigma = means_dev_mag_mod_sigma_layer(means_concat_embeddings)
        means_dev_mag_kl_loss = means_dev_mag_kl_loss_layer(
            [
                data,
                means_dev_mag_inf_mu,
                means_dev_mag_inf_sigma,
                means_dev_mag_mod_sigma,
            ]
        )
    else:
        means_dev_mag_kl_loss_layer = ZeroLayer((), name="means_dev_mag_kl_loss")
        means_dev_mag_kl_loss = means_dev_mag_kl_loss_layer(data)

    if config.learn_covariances:
        # Layer definitions
        covs_dev_mag_mod_sigma_layer = layers.Dense(
            1, activation="softplus", name="covs_dev_mag_mod_sigma"
        )
        covs_dev_mag_kl_loss_layer = StaticKLDivergenceLayer(
            config.covariances_epsilon, name="covs_dev_mag_kl_loss"
        )

        # Data flow
        covs_dev_mag_mod_sigma = covs_dev_mag_mod_sigma_layer(covs_concat_embeddings)
        covs_dev_mag_kl_loss = covs_dev_mag_kl_loss_layer(
            [
                data,
                covs_dev_mag_inf_mu,
                covs_dev_mag_inf_sigma,
                covs_dev_mag_mod_sigma,
            ]
        )
    else:
        covs_dev_mag_kl_loss_layer = ZeroLayer((), name="covs_dev_mag_kl_loss")
        covs_dev_mag_kl_loss = means_dev_mag_kl_loss_layer(data)

    # Total KL loss
    # Layer definitions
    kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")

    # Data flow
    kl_loss = kl_loss_layer([means_dev_mag_kl_loss, covs_dev_mag_kl_loss])

    return tf.keras.Model(
        inputs=[data, alpha, subj_id],
        outputs=[ll_loss, kl_loss],
        name="Se_DyNeMo-Obs",
    )


def get_group_means_covariances(model):
    group_means_layer = model.get_layer("group_means")
    group_covs_layer = model.get_layer("group_covs")

    group_means = group_means_layer(1)
    group_covs = group_covs_layer(1)
    return group_means.numpy(), group_covs.numpy()


def get_subject_embeddings(model):
    subject_embeddings_layer = model.get_layer("subject_embeddings")
    n_subjects = subject_embeddings_layer.input_dim
    return subject_embeddings_layer(np.arange(n_subjects)).numpy()


def get_means_mode_embeddings(model):
    group_means, _ = get_group_means_covariances(model)
    means_mode_embeddings_layer = model.get_layer("means_mode_embeddings")
    means_mode_embeddings = means_mode_embeddings_layer(group_means)
    return means_mode_embeddings.numpy()


def get_covariances_mode_embeddings(model):
    cholesky_bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])
    _, group_covs = get_group_means_covariances(model)
    covs_mode_embeddings_layer = model.get_layer("covs_mode_embeddings")
    covs_mode_embeddings = covs_mode_embeddings_layer(
        cholesky_bijector.inverse(group_covs)
    )
    return covs_mode_embeddings.numpy()


def get_means_concatenated_embeddings(model):
    subject_embeddings = get_subject_embeddings(model)
    means_mode_embeddings = get_means_mode_embeddings(model)
    means_concat_embeddings_layer = model.get_layer("means_concat_embeddings")
    means_concat_embeddings = means_concat_embeddings_layer(
        [subject_embeddings, means_mode_embeddings]
    )
    return means_concat_embeddings.numpy()


def get_covariances_concatenated_embeddings(model):
    subject_embeddings = get_subject_embeddings(model)
    covs_mode_embeddings = get_covariances_mode_embeddings(model)
    covs_concat_embeddings_layer = model.get_layer("covs_concat_embeddings")
    covs_concat_embeddings = covs_concat_embeddings_layer(
        [subject_embeddings, covs_mode_embeddings]
    )
    return covs_concat_embeddings.numpy()


def get_means_dev_mag(model):
    means_dev_mag_input_layer = model.get_layer("means_dev_mag_inf_mu")
    means_dev_mag_layer = model.get_layer("means_dev_mag")

    means_dev_mag_input = means_dev_mag_input_layer(1)
    means_dev_mag = means_dev_mag_layer(means_dev_mag_input)
    return means_dev_mag.numpy()


def get_covariances_dev_mag(model):
    covs_dev_mag_input_layer = model.get_layer("covs_dev_mag_inf_mu")
    covs_dev_mag_layer = model.get_layer("covs_dev_mag")

    covs_dev_mag_input = covs_dev_mag_input_layer(1)
    covs_dev_mag = covs_dev_mag_layer(covs_dev_mag_input)
    return covs_dev_mag.numpy()


def get_means_dev_map(model):
    means_concat_embeddings = get_means_concatenated_embeddings(model)

    means_dev_map_input_layer = model.get_layer("means_dev_map_input")
    means_dev_map_layer = model.get_layer("means_dev_map")
    norm_means_dev_map_layer = model.get_layer("norm_means_dev_map")

    means_dev_map_input = means_dev_map_input_layer(means_concat_embeddings)
    means_dev_map = means_dev_map_layer(means_dev_map_input)
    norm_means_dev_map = norm_means_dev_map_layer(means_dev_map)
    return norm_means_dev_map.numpy()


def get_covariances_dev_map(model):
    covs_concat_embeddings = get_covariances_concatenated_embeddings(model)

    covs_dev_map_input_layer = model.get_layer("covs_dev_map_input")
    covs_dev_map_layer = model.get_layer("covs_dev_map")
    norm_covs_dev_map_layer = model.get_layer("norm_covs_dev_map")

    covs_dev_map_input = covs_dev_map_input_layer(covs_concat_embeddings)
    covs_dev_map = covs_dev_map_layer(covs_dev_map_input)
    norm_covs_dev_map = norm_covs_dev_map_layer(covs_dev_map)
    return norm_covs_dev_map.numpy()


def get_subject_dev(model, learn_means, learn_covariances):
    means_dev_layer = model.get_layer("means_dev")
    covs_dev_layer = model.get_layer("covs_dev")
    if learn_means:
        means_dev_mag = get_means_dev_mag(model)
        means_dev_map = get_means_dev_map(model)
        means_dev = means_dev_layer([means_dev_mag, means_dev_map])
    else:
        means_dev = means_dev_layer(1)

    if learn_covariances:
        covs_dev_mag = get_covariances_dev_mag(model)
        covs_dev_map = get_covariances_dev_map(model)
        covs_dev = covs_dev_layer([covs_dev_mag, covs_dev_map])
    else:
        covs_dev = covs_dev_layer(1)

    return means_dev.numpy(), covs_dev.numpy()


def get_subject_means_covariances(model, learn_means, learn_covariances):
    group_means, group_covs = get_group_means_covariances(model)
    means_dev, covs_dev = get_subject_dev(model, learn_means, learn_covariances)

    subject_means_layer = model.get_layer("subject_means")
    subject_covs_layer = model.get_layer("subject_covs")

    mu = subject_means_layer([group_means, means_dev])
    D = subject_covs_layer([group_covs, covs_dev])
    return mu.numpy(), D.numpy()


def set_bayesian_kl_scaling(model, n_batches, learn_means, learn_covariances):
    if learn_means:
        means_dev_mag_kl_loss_layer = model.get_layer("means_dev_mag_kl_loss")
        means_dev_mag_kl_loss_layer.n_batches = n_batches

    if learn_covariances:
        covs_dev_mag_kl_loss_layer = model.get_layer("covs_dev_mag_kl_loss")
        covs_dev_mag_kl_loss_layer.n_batches = n_batches
