"""Subject-Embedding DyNeMo (SE-DyNeMo).

"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers

import osl_dynamics.data.tf as dtf
from osl_dynamics.models import obs_mod
from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.models.inf_mod_base import (
    VariationalInferenceModelConfig,
    VariationalInferenceModelBase,
)
from osl_dynamics.inference.layers import (
    InferenceRNNLayer,
    LogLikelihoodLossLayer,
    VectorsLayer,
    CovarianceMatricesLayer,
    ModelRNNLayer,
    NormalizationLayer,
    KLDivergenceLayer,
    KLLossLayer,
    SampleNormalDistributionLayer,
    SampleGammaDistributionLayer,
    SoftmaxLayer,
    ConcatEmbeddingsLayer,
    SubjectMapLayer,
    MixSubjectSpecificParametersLayer,
    TFRangeLayer,
    ZeroLayer,
    InverseCholeskyLayer,
    StaticKLDivergenceLayer,
    MultiLayerPerceptronLayer,
    LearnableTensorLayer,
    StaticLossScalingFactorLayer,
)


@dataclass
class Config(BaseModelConfig, VariationalInferenceModelConfig):
    """Settings for SE-DyNeMo.

    Parameters
    ----------
    model_name : str
        Model name.

    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.

    inference_rnn : str
        RNN to use, either :code:`'gru'` or :code:`'lstm'`.
    inference_n_layers : int
        Number of layers.
    inference_n_untis : int
        Number of units.
    inference_normalization : str
        Type of normalization to use. Either :code:`None`, :code:`'batch'`
        or :code:`'layer'`.
    inference_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. :code:`'relu'`, :code:`'elu'`, etc.
    inference_dropout : float
        Dropout rate.
    inference_regularizer : str
        Regularizer.

    model_rnn : str
        RNN to use, either :code:`'gru'` or :code:`'lstm'`.
    model_n_layers : int
        Number of layers.
    model_n_units : int
        Number of units.
    model_normalization : str
        Type of normalization to use. Either :code:`None`, :code:`'batch'`
        or :code:`'layer'`.
    model_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. :code:`'relu'`, :code:`'elu'`, etc.
    model_dropout : float
        Dropout rate.
    model_regularizer : str
        Regularizer.

    theta_normalization : str
        Type of normalization to apply to the posterior samples, :code:`theta`.
        Either :code:`'layer'`, :code:`'batch'` or :code:`None`.
    learn_alpha_temperature : bool
        Should we learn :code:`alpha_temperature`?
    initial_alpha_temperature : float
        Initial value for :code:`alpha_temperature`.

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

    do_kl_annealing : bool
        Should we use KL annealing during training?
    kl_annealing_curve : str
        Type of KL annealing curve. Either :code:`'linear'` or :code:`'tanh'`.
    kl_annealing_sharpness : float
        Parameter to control the shape of the annealing curve if
        :code:`kl_annealing_curve='tanh'`.
    n_kl_annealing_epochs : int
        Number of epochs to perform KL annealing.

    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
    lr_decay : float
        Decay for learning rate. Default is 0.1. We use
        :code:`lr = learning_rate * exp(-lr_decay * epoch)`.
    gradient_clip : float
        Value to clip gradients by. This is the :code:`clipnorm` argument
        passed to the Keras optimizer. Cannot be used if :code:`multi_gpu=True`.
    n_epochs : int
        Number of training epochs.
    optimizer : str or tf.keras.optimizers.Optimizer
        Optimizer to use. :code:`'adam'` is recommended.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.

    n_subjects : int
        Number of subjects.
    subject_embeddings_dim : int
        Number of dimensions for the subject embedding.
    mode_embeddings_dim : int
        Number of dimensions for the mode embedding in the spatial maps encoder.

    dev_n_layers : int
        Number of layers for the MLP for deviations.
    dev_n_units : int
        Number of units for the MLP for deviations.
    dev_normalization : str
        Type of normalization for the MLP for deviations.
    dev_activation : str
        Type of activation to use for the MLP for deviations.
    dev_dropout : float
        Dropout rate for the MLP for deviations.
    dev_regularizer : str
        Regularizer for the MLP for deviations.
    dev_regularizer_factor : float
        Regularizer factor for the MLP for deviations.
        This will be scaled by the amount of data.
    """

    model_name: str = "SE-DyNeMo"

    # Inference network parameters
    inference_rnn: str = "lstm"
    inference_n_layers: int = 1
    inference_n_units: int = None
    inference_normalization: str = None
    inference_activation: str = None
    inference_dropout: float = 0.0
    inference_regularizer: str = None

    # Model network parameters
    model_rnn: str = "lstm"
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: str = None
    model_activation: str = None
    model_dropout: float = 0.0
    model_regularizer: str = None

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
    dev_regularizer: str = None
    dev_regularizer_factor: float = 0.0

    def __post_init__(self):
        self.validate_rnn_parameters()
        self.validate_observation_model_parameters()
        self.validate_alpha_parameters()
        self.validate_kl_annealing_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()
        self.validate_subject_embedding_parameters()

    def validate_rnn_parameters(self):
        if self.inference_n_units is None:
            raise ValueError("Please pass inference_n_units.")

        if self.model_n_units is None:
            raise ValueError("Please pass model_n_units.")

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
                "n_subjects, subject_embedding_dim and mode_embedding_dim must "
                "be passed."
            )

        if self.dev_n_layers != 0 and self.dev_n_units is None:
            raise ValueError("Please pass dev_n_units.")


class Model(VariationalInferenceModelBase):
    """Directional Subject embedding DyNeMo.

    Parameters
    ----------
    config : osl_dynamics_models.sedynemo.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_group_means(self):
        """Get the group level mode means.

        Returns
        -------
        means : np.ndarray
            Group means. Shape is (n_modes, n_channels).
        """
        return obs_mod.get_observation_model_parameter(
            self.model,
            "group_means",
        )

    def get_group_covariances(self):
        """Get the group level mode covariances.

        Returns
        -------
        covariances : np.ndarray
            Group covariances. Shape is (n_modes, n_channels, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "group_covs")

    def get_group_means_covariances(self):
        """Get the group level mode means and covariances.

        This is a wrapper for :code:`get_group_means` and
        :code:`get_group_covariances`.

        Returns
        -------
        means : np.ndarray
            Group means. Shape is (n_modes, n_channels).
        covariances : np.ndarray
            Group covariances. Shape is (n_modes, n_channels, n_channels).
        """
        return self.get_group_means(), self.get_group_covariances()

    def get_group_observation_model_parameters(self):
        """Wrapper for :code:`get_group_means_covariances`."""
        return self.get_group_means_covariances()

    def get_subject_embeddings(self):
        """Get the subject embedding vectors

        Returns
        -------
        subject_embeddings : np.ndarray
            Embedding vectors for subjects.
            Shape is (n_subjects, subject_embedding_dim).
        """
        return obs_mod.get_subject_embeddings(self.model)

    def get_subject_means_covariances(
        self,
        subject_embeddings=None,
        n_neighbours=2,
    ):
        """Get the means and covariances for each subject.

        Parameters
        ----------
        subject_embeddings : np.ndarray, optional
            Input embedding vectors for subjects.
            Shape is (n_subjects, subject_embeddings_dim).
        n_neighbours : int, optional
            Number of nearest neighbours.
            Ignored if :code:`subject_embedding=None`.

        Returns
        -------
        subject_means : np.ndarray
            Mode means for each subject.
            Shape is (n_subjects, n_modes, n_channels).
        subject_covs : np.ndarray
            Mode covariances for each subject.
            Shape is (n_subjects, n_modes, n_channels, n_channels).
        """
        return obs_mod.get_subject_means_covariances(
            self.model,
            self.config.learn_means,
            self.config.learn_covariances,
            subject_embeddings,
            n_neighbours,
        )

    def set_regularizers(self, training_dataset):
        """Set the means and covariances regularizer based on the training data.

        A multivariate normal prior is applied to the mean vectors with
        :code:`mu=0`, :code:`sigma=diag((range/2)**2)` and an inverse Wishart
        prior is applied to the covariances matrices with
        :code:`nu=n_channels-1+0.1` and :code:`psi=diag(range)`.

        Parameters
        ----------
        training_dataset : tf.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        """
        training_dataset = self.make_dataset(
            training_dataset,
            concatenate=True,
        )

        if self.config.learn_means:
            obs_mod.set_means_regularizer(
                self.model, training_dataset, layer_name="group_means"
            )

        if self.config.learn_covariances:
            obs_mod.set_covariances_regularizer(
                self.model,
                training_dataset,
                self.config.covariances_epsilon,
                layer_name="group_covs",
            )

    def set_group_means(self, group_means, update_initializer=True):
        """Set the group means of each mode.

        Parameters
        ----------
        group_means : np.ndarray
            Group level mode means. Shape is (n_modes, n_channels).
        update_initializer : bool, optional
            Do we want to use the passed group means when we re-initialize
            the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            group_means,
            layer_name="group_means",
            update_initializer=update_initializer,
        )

    def set_group_covariances(self, group_covariances, update_initializer=True):
        """Set the group covariances of each mode.

        Parameters
        ----------
        group_covariances : np.ndarray
            Group level mode covariances.
            Shape is (n_modes, n_channels, n_channels).
        update_initializer : bool, optional
            Do we want to use the passed group covariances when we re-initialize
            the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            group_covariances,
            layer_name="group_covs",
            update_initializer=update_initializer,
        )

    def set_group_means_covariances(
        self, group_means, group_covariances, update_initializer=True
    ):
        """This is a wrapper for :code:`set_group_means` and
        :code:`set_group_covariances`."""
        self.set_group_means(
            group_means,
            update_initializer=update_initializer,
        )
        self.set_group_covariances(
            group_covariances,
            update_initializer=update_initializer,
        )

    def set_group_observation_model_parameters(
        self, group_observation_model_parameters, update_initializer=True
    ):
        """Wrapper for :code:`set_group_means_covariances`."""
        self.set_group_means_covariances(
            group_observation_model_parameters[0],
            group_observation_model_parameters[1],
            update_initializer=update_initializer,
        )

    def set_bayesian_kl_scaling(self, training_dataset):
        """Set the correct scaling for KL loss between deviation posterior
        and prior.

        Parameters
        ----------
        training_dataset : tf.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        """
        training_dataset = self.make_dataset(training_dataset, concatenate=True)
        n_batches = dtf.get_n_batches(training_dataset)
        learn_means = self.config.learn_means
        learn_covariances = self.config.learn_covariances
        obs_mod.set_bayesian_kl_scaling(
            self.model, n_batches, learn_means, learn_covariances
        )

    def set_dev_mlp_reg_scaling(self, training_dataset):
        """Set the correct scaling for the deviation MLP regularization.

        Parameters
        ----------
        training_dataset : tf.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        """
        training_dataset = self.make_dataset(training_dataset, concatenate=True)
        n_batches = dtf.get_n_batches(training_dataset)
        learn_means = self.config.learn_means
        learn_covariances = self.config.learn_covariances
        obs_mod.set_dev_mlp_reg_scaling(
            self.model, n_batches, learn_means, learn_covariances
        )

    def random_subject_initialization(self, **kwargs):
        """random subject initialisation not compatible with SE-DyNeMo."""
        raise AttributeError(
            " 'Model' object has no attribute 'random_subject_initialization'."
        )

    def get_n_params_generative_model(self):
        """Get the number of trainable parameters in the generative model.

        This includes the model RNN weights and biases, mixing coefficients,
        mode means and covariances (group and subject deviation) and subject
        embeddings.

        Returns
        -------
        n_params : int
            Number of parameters in the generative model.
        """
        n_params = 0

        for var in self.trainable_weights:
            var_name = var.name
            if (
                "mod_" in var_name
                or "alpha" in var_name
                or "group_means" in var_name
                or "group_covs" in var_name
                or "_embeddings" in var_name
                or "dev_map" in var_name
            ):
                n_params += np.prod(var.shape)

        return int(n_params)


def _model_structure(config):
    # Layers for inputs
    data = layers.Input(
        shape=(config.sequence_length, config.n_channels),
        name="data",
    )
    array_id = layers.Input(shape=(config.sequence_length,), name="array_id")

    # Static loss scaling factor
    static_loss_scaling_factor_layer = StaticLossScalingFactorLayer(
        name="static_loss_scaling_factor"
    )
    static_loss_scaling_factor = static_loss_scaling_factor_layer(data)

    # Inference RNN:
    # Layer definitions
    inference_input_dropout_layer = layers.Dropout(
        config.inference_dropout, name="data_drop"
    )
    inference_output_layer = InferenceRNNLayer(
        config.inference_rnn,
        config.inference_normalization,
        config.inference_activation,
        config.inference_n_layers,
        config.inference_n_units,
        config.inference_dropout,
        config.inference_regularizer,
        name="inf_rnn",
    )
    inf_mu_layer = layers.Dense(config.n_modes, name="inf_mu")
    inf_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="inf_sigma"
    )

    # Layers to sample theta from q(theta) and to convert to mode mixing
    # factors alpha
    theta_layer = SampleNormalDistributionLayer(
        config.theta_std_epsilon,
        name="theta",
    )
    theta_norm_layer = NormalizationLayer(
        config.theta_normalization,
        name="theta_norm",
    )
    alpha_layer = SoftmaxLayer(
        config.initial_alpha_temperature,
        config.learn_alpha_temperature,
        name="alpha",
    )

    # Data flow
    inference_input_dropout = inference_input_dropout_layer(data)
    inference_output = inference_output_layer(inference_input_dropout)
    inf_mu = inf_mu_layer(inference_output)
    inf_sigma = inf_sigma_layer(inference_output)
    theta = theta_layer([inf_mu, inf_sigma])
    theta_norm = theta_norm_layer(theta)
    alpha = alpha_layer(theta_norm)

    # -----------------
    # Observation model

    # Subject embedding layers
    subjects_layer = TFRangeLayer(config.n_subjects, name="subjects")
    subject_embeddings_layer = layers.Embedding(
        config.n_subjects,
        config.subject_embeddings_dim,
        name="subject_embeddings",
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

    group_mu = group_means_layer(
        data, static_loss_scaling_factor=static_loss_scaling_factor
    )
    group_D = group_covs_layer(
        data, static_loss_scaling_factor=static_loss_scaling_factor
    )

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
            config.dev_regularizer,
            config.dev_regularizer_factor,
            name="means_dev_map_input",
        )
        means_dev_map_layer = layers.Dense(
            config.n_channels,
            name="means_dev_map",
        )

        norm_means_dev_map_layer = layers.LayerNormalization(
            axis=-1, name="norm_means_dev_map"
        )

        means_dev_mag_inf_alpha_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_modes, 1),
            learn=config.learn_means,
            initializer=initializers.TruncatedNormal(mean=0, stddev=0.02),
            name="means_dev_mag_inf_alpha_input",
        )
        means_dev_mag_inf_alpha_layer = layers.Activation(
            "softplus", name="means_dev_mag_inf_alpha"
        )
        means_dev_mag_inf_beta_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_modes, 1),
            learn=config.learn_means,
            initializer=initializers.TruncatedNormal(mean=10, stddev=0.02),
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
        means_dev_map_input = means_dev_map_input_layer(
            means_concat_embeddings,
            static_loss_scaling_factor=static_loss_scaling_factor,
        )
        means_dev_map = means_dev_map_layer(means_dev_map_input)
        norm_means_dev_map = norm_means_dev_map_layer(means_dev_map)

        # Get the deviation magnitudes (scale deviation maps globally)

        means_dev_mag_inf_alpha_input = means_dev_mag_inf_alpha_input_layer(
            data,
        )
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
            name="covs_concat_embeddings",
        )

        covs_dev_map_input_layer = MultiLayerPerceptronLayer(
            config.dev_n_layers,
            config.dev_n_units,
            config.dev_normalization,
            config.dev_activation,
            config.dev_dropout,
            config.dev_regularizer,
            config.dev_regularizer_factor,
            name="covs_dev_map_input",
        )
        covs_dev_map_layer = layers.Dense(
            config.n_channels * (config.n_channels + 1) // 2, name="covs_dev_map"
        )
        norm_covs_dev_map_layer = layers.LayerNormalization(
            axis=-1, name="norm_covs_dev_map"
        )

        covs_dev_mag_inf_alpha_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_modes, 1),
            learn=config.learn_covariances,
            initializer=initializers.TruncatedNormal(mean=20, stddev=10),
            name="covs_dev_mag_inf_alpha_input",
        )
        covs_dev_mag_inf_alpha_layer = layers.Activation(
            "softplus", name="covs_dev_mag_inf_alpha"
        )
        covs_dev_mag_inf_beta_input_layer = LearnableTensorLayer(
            shape=(config.n_subjects, config.n_modes, 1),
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
        covs_dev_map_input = covs_dev_map_input_layer(
            covs_concat_embeddings,
            static_loss_scaling_factor=static_loss_scaling_factor,
        )
        covs_dev_map = covs_dev_map_layer(covs_dev_map_input)
        norm_covs_dev_map = norm_covs_dev_map_layer(covs_dev_map)

        # Get the deviation magnitudes (scale deviation maps globally)
        covs_dev_mag_inf_alpha_input = covs_dev_mag_inf_alpha_input_layer(data)
        covs_dev_mag_inf_alpha = covs_dev_mag_inf_alpha_layer(
            covs_dev_mag_inf_alpha_input
        )
        covs_dev_mag_inf_beta_input = covs_dev_mag_inf_beta_input_layer(data)
        covs_dev_mag_inf_beta = covs_dev_mag_inf_beta_layer(
            covs_dev_mag_inf_beta_input,
        )
        covs_dev_mag = covs_dev_mag_layer(
            [covs_dev_mag_inf_alpha, covs_dev_mag_inf_beta]
        )
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
    ll_loss_layer = LogLikelihoodLossLayer(
        config.covariances_epsilon,
        name="ll_loss",
    )

    # Data flow
    m, C = mix_subject_means_covs_layer([alpha, mu, D, array_id])
    ll_loss = ll_loss_layer([data, m, C])

    # ---------
    # KL losses

    # For the time courses (dynamic KL loss)
    # Layer definitions
    model_input_dropout_layer = layers.Dropout(
        config.model_dropout, name="theta_norm_drop"
    )
    model_output_layer = ModelRNNLayer(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout,
        config.model_regularizer,
        name="mod_rnn",
    )
    mod_mu_layer = layers.Dense(config.n_modes, name="mod_mu")
    mod_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="mod_sigma"
    )
    kl_div_layer = KLDivergenceLayer(config.theta_std_epsilon, name="kl_div")

    # Data flow
    model_input_dropout = model_input_dropout_layer(theta_norm)
    model_output = model_output_layer(model_input_dropout)
    mod_mu = mod_mu_layer(model_output)
    mod_sigma = mod_sigma_layer(model_output)
    kl_div = kl_div_layer([inf_mu, inf_sigma, mod_mu, mod_sigma])

    # For the observation model (static KL loss)
    if config.learn_means:
        # Layer definitions
        means_dev_mag_mod_beta_input_layer = MultiLayerPerceptronLayer(
            config.dev_n_layers,
            config.dev_n_units,
            config.dev_normalization,
            config.dev_activation,
            config.dev_dropout,
            config.dev_regularizer,
            config.dev_regularizer_factor,
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
            means_concat_embeddings,
            static_loss_scaling_factor=static_loss_scaling_factor,
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
            ],
            static_loss_scaling_factor=static_loss_scaling_factor,
        )
    else:
        means_dev_mag_kl_loss_layer = ZeroLayer(
            (),
            name="means_dev_mag_kl_loss",
        )
        means_dev_mag_kl_loss = means_dev_mag_kl_loss_layer(data)

    if config.learn_covariances:
        # Layer definitions
        covs_dev_mag_mod_beta_input_layer = MultiLayerPerceptronLayer(
            config.dev_n_layers,
            config.dev_n_units,
            config.dev_normalization,
            config.dev_activation,
            config.dev_dropout,
            config.dev_regularizer,
            config.dev_regularizer_factor,
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
            covs_concat_embeddings,
            static_loss_scaling_factor=static_loss_scaling_factor,
        )
        covs_dev_mag_mod_beta = covs_dev_mag_mod_beta_layer(
            covs_dev_mag_mod_beta_input,
        )
        covs_dev_mag_kl_loss = covs_dev_mag_kl_loss_layer(
            [
                data,
                covs_dev_mag_inf_alpha,
                covs_dev_mag_inf_beta,
                covs_dev_mag_mod_beta,
            ],
            static_loss_scaling_factor=static_loss_scaling_factor,
        )
    else:
        covs_dev_mag_kl_loss_layer = ZeroLayer((), name="covs_dev_mag_kl_loss")
        covs_dev_mag_kl_loss = covs_dev_mag_kl_loss_layer(data)

    # Total KL loss
    # Layer definitions
    kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")

    # Data flow
    kl_loss = kl_loss_layer(
        [kl_div, means_dev_mag_kl_loss, covs_dev_mag_kl_loss],
    )

    return tf.keras.Model(
        inputs=[data, array_id],
        outputs=[ll_loss, kl_loss, theta_norm],
        name="Se-DyNeMo",
    )
