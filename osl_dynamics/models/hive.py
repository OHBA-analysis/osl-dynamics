"""HIVE (HMM with Integrated Variability Estimation).

"""

import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

from osl_dynamics.inference.layers import (
    LearnableTensorLayer,
    VectorsLayer,
    CovarianceMatricesLayer,
    ConcatEmbeddingsLayer,
    SessionMapLayer,
    TFRangeLayer,
    ZeroLayer,
    InverseCholeskyLayer,
    SampleGammaDistributionLayer,
    StaticKLDivergenceLayer,
    KLLossLayer,
    MultiLayerPerceptronLayer,
    StaticLossScalingFactorLayer,
    HiddenMarkovStateInferenceLayer,
    SeparateLogLikelihoodLayer,
    SumLogLikelihoodLossLayer,
    BatchSizeLayer,
)
from osl_dynamics.models import obs_mod
from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.inference import callbacks
import osl_dynamics.inference.initializers as osld_initializers
from osl_dynamics.models.inf_mod_base import (
    MarkovStateInferenceModelConfig,
    MarkovStateInferenceModelBase,
)
from osl_dynamics.utils.misc import replace_argument

_logger = logging.getLogger("osl-dynamics")


@dataclass
class Config(BaseModelConfig, MarkovStateInferenceModelConfig):
    """Settings for HIVE.

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
        Error added to state covariances for numerical stability.
    means_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for group mean vectors.
    covariances_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for group covariance matrices.

    n_sessions : int
        Number of arrays whose observation model parameters can vary.
    embeddings_dim : int
        Number of dimensions for embeddings dimension.
    spatial_embeddings_dim : int
        Number of dimensions for spatial embeddings.

    dev_n_layers : int
        Number of layers for the MLP for deviations.
    dev_n_units : int
        Number of units for the MLP for deviations.
    dev_normalization : str
        Type of normalization for the MLP for deviations.
        Either :code:`None`, :code:`'batch'` or :code:`'layer'`.
    dev_activation : str
        Type of activation to use for the MLP for deviations.
        E.g. :code:`'relu'`, :code:`'sigmoid'`, :code:`'tanh'`, etc.
    dev_dropout : float
        Dropout rate for the MLP for deviations.
    dev_regularizer : str
        Regularizer for the MLP for deviations.
    dev_regularizer_factor : float
        Regularizer factor for the MLP for deviations.
        This will be scaled by the amount of data.
    initial_dev : dict
        Initialisation for dev posterior parameters.

    initial_trans_prob : np.ndarray
        Initialisation for transition probability matrix.
    learn_trans_prob : bool
        Should we make the transition probability matrix trainable?
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

    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
    lr_decay : float
        Decay for learning rate. Default is 0.1. We use
        :code:`lr = learning_rate * exp(-lr_decay * epoch)`.
    n_epochs : int
        Number of training epochs.
    optimizer : str or tf.keras.optimizers.Optimizer
        Optimizer to use.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.

    do_kl_annealing : bool
        Should we use KL annealing during training?
    kl_annealing_curve : str
        Type of KL annealing curve. Either :code:`'linear'` or :code:`'tanh'`.
    kl_annealing_sharpness : float
        Parameter to control the shape of the annealing curve if
        :code:`kl_annealing_curve='tanh'`.
    n_kl_annealing_epochs : int
        Number of epochs to perform KL annealing.
    """

    model_name: str = "HIVE"

    # Parameters specific to embedding model
    n_sessions: int = None
    embeddings_dim: int = None
    spatial_embeddings_dim: int = None

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None
    covariances_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

    dev_n_layers: int = 0
    dev_n_units: int = None
    dev_normalization: str = None
    dev_activation: str = None
    dev_dropout: float = 0.0
    dev_regularizer: str = None
    dev_regularizer_factor: float = 0.0
    initial_dev: dict = None

    # KL annealing parameters
    do_kl_annealing: bool = False
    kl_annealing_curve: str = None
    kl_annealing_sharpness: float = None
    n_kl_annealing_epochs: int = None

    def __post_init__(self):
        self.validate_observation_model_parameters()
        self.validate_trans_prob_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()
        self.validate_embedding_parameters()
        self.validate_kl_annealing_parameters()

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")

        if self.covariances_epsilon is None:
            if self.learn_covariances:
                self.covariances_epsilon = 1e-6
            else:
                self.covariances_epsilon = 0.0

        if self.initial_dev is None:
            self.initial_dev = dict()

    def validate_embedding_parameters(self):
        if (
            self.n_sessions is None
            or self.embeddings_dim is None
            or self.spatial_embeddings_dim is None
        ):
            raise ValueError(
                "n_sessions, embeddings_dim and spatial_embeddings_dim must be passed."
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
                        "kl_annealing_curve='tanh'."
                    )

                if self.kl_annealing_sharpness < 0:
                    raise ValueError("KL annealing sharpness must be positive.")

            if self.n_kl_annealing_epochs is None:
                raise ValueError(
                    "If we are performing KL annealing, "
                    "n_kl_annealing_epochs must be passed."
                )

            if self.n_kl_annealing_epochs < 1:
                raise ValueError(
                    "Number of KL annealing epochs must be greater than zero."
                )


class Model(MarkovStateInferenceModelBase):
    """HIVE model class.

    Parameters
    ----------
    config : osl_dynamics.models.hive.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def fit(self, *args, kl_annealing_callback=None, **kwargs):
        """Wrapper for the standard keras fit method.

        Parameters
        ----------
        *args : arguments
            Arguments for :code:`MarkovStateInferenceModelBase.fit()`.
        kl_annealing_callback : bool, optional
            Should we update the KL annealing factor during training?
        **kwargs : keyword arguments, optional
            Keyword arguments for :code:`MarkovStateInferenceModelBase.fit()`.

        Returns
        -------
        history : history
            The training history.
        """
        # Callback for KL annealing
        if kl_annealing_callback is None:
            kl_annealing_callback = self.config.do_kl_annealing

        # KL annealing
        if kl_annealing_callback:
            kl_annealing_callback = callbacks.KLAnnealingCallback(
                curve=self.config.kl_annealing_curve,
                annealing_sharpness=self.config.kl_annealing_sharpness,
                n_annealing_epochs=self.config.n_kl_annealing_epochs,
            )

            # Update arguments to pass to the fit method
            args, kwargs = replace_argument(
                self.model.fit,
                "callbacks",
                [kl_annealing_callback],
                args,
                kwargs,
                append=True,
            )

        return super().fit(*args, **kwargs)

    def reset_weights(self, keep=None):
        """Reset the model weights.

        Parameters
        ----------
        keep : list of str, optional
            Layer names to NOT reset.
        """
        super().reset_weights(keep=keep)
        self.reset_kl_annealing_factor()

    def reset_kl_annealing_factor(self):
        """Reset the KL annealing factor."""
        if self.config.do_kl_annealing:
            kl_loss_layer = self.model.get_layer("kl_loss")
            kl_loss_layer.annealing_factor.assign(0.0)

    def get_group_means(self):
        """Get the group level state means.

        Returns
        -------
        means : np.ndarray
            Group means. Shape is (n_states, n_channels).
        """
        return obs_mod.get_observation_model_parameter(
            self.model,
            "group_means",
        )

    def get_means(self):
        """Wrapper for :code:`get_group_means`."""
        return self.get_group_means()

    def get_group_covariances(self):
        """Get the group level state covariances.

        Returns
        -------
        covariances : np.ndarray
            Group covariances. Shape is (n_states, n_channels, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "group_covs")

    def get_covariances(self):
        """Wrapper for :code:`get_group_covariances`."""
        return self.get_group_covariances()

    def get_group_means_covariances(self):
        """Get the group level state means and covariances.

        This is a wrapper for :code:`get_group_means` and
        :code:`get_group_covariances`.

        Returns
        -------
        means : np.ndarray
            Group means. Shape is (n_states, n_channels).
        covariances : np.ndarray
            Group covariances. Shape is (n_states, n_channels, n_channels).
        """
        return self.get_group_means(), self.get_group_covariances()

    def get_means_covariances(self):
        """Wrapper for :code:`get_group_means_covariances`."""
        return self.get_group_means_covariances()

    def get_group_observation_model_parameters(self):
        """Wrapper for get_group_means_covariances."""
        return self.get_group_means_covariances()

    def get_observation_model_parameters(self):
        """Wrapper for :code:`get_group_observation_model_parameters`."""
        return self.get_group_observation_model_parameters()

    def get_session_means_covariances(
        self,
        embeddings=None,
        n_neighbours=2,
    ):
        """Get the array means and covariances.

        Parameters
        ----------
        embeddings : np.ndarray, optional
            Input embedding vectors.
            Shape is (n_sessions, embeddings_dim).
        n_neighbours : int, optional
            Number of nearest neighbours. Ignored if
            :code:`embeddings=None`.

        Returns
        -------
        means : np.ndarray
            Session means. Shape is (n_sessions, n_states, n_channels).
        covs : np.ndarray
            Session covariances.
            Shape is (n_sessions, n_states, n_channels, n_channels).
        """
        return obs_mod.get_session_means_covariances(
            self.model,
            self.config.learn_means,
            self.config.learn_covariances,
            embeddings,
            n_neighbours,
        )

    def get_embeddings(self):
        """Get the embedding vectors.

        Returns
        -------
        embeddings : np.ndarray
            Embedding vectors.
            Shape is (n_sessions, embeddings_dim).
        """
        return obs_mod.get_embeddings(self.model)

    def set_group_means(self, group_means, update_initializer=True):
        """Set the group means of each state.

        Parameters
        ----------
        group_means : np.ndarray
            Group level state means. Shape is (n_states, n_channels).
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
        """Set the group covariances of each state.

        Parameters
        ----------
        group_covariances : np.ndarray
            Group level state covariances.
            Shape is (n_states, n_channels, n_channels).
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
        """Wrapper for :code:`set_group_means` and
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

    def set_means(self, means, update_initializer=True):
        """Wrapper for :code:`set_group_means`."""
        self.set_group_means(means, update_initializer)

    def set_covariances(self, covariances, update_initializer=True):
        """Wrapper for :code:`set_group_covariances`."""
        self.set_group_covariances(covariances, update_initializer)

    def set_means_covariances(
        self,
        means,
        covariances,
        update_initializer=True,
    ):
        """Wrapper for :code:`set_group_means_covariances`."""
        self.set_group_means_covariances(means, covariances, update_initializer)

    def set_observation_model_parameters(
        self, observation_model_parameters, update_initializer=True
    ):
        """Wrapper for :code:`set_group_observation_model_parameters`."""
        self.set_group_observation_model_parameters(
            observation_model_parameters, update_initializer
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

    def set_dev_parameters_initializer(self, training_dataset):
        """Set the deviance parameters initializer based on training data.

        Parameters
        ----------
        training_dataset : osl_dynamics.data.Data
            The training dataset.
        """
        obs_mod.set_dev_parameters_initializer(
            self.model,
            training_dataset,
            self.config.learn_means,
            self.config.learn_covariances,
        )
        self.reset()

    def set_embeddings_initializer(self, embeddings):
        """Set the embeddings initializer.

        Parameters
        ----------
        embeddings : np.ndarray
            The embeddings. Shape is (n_sessions, embeddings_dim).
        """
        obs_mod.set_embeddings_initializer(
            self.model,
            embeddings,
        )
        self.reset()


def _model_structure(config):
    # Inputs
    data = layers.Input(
        shape=(config.sequence_length, config.n_channels),
        dtype=tf.float32,
        name="data",
    )
    session_id = layers.Input(
        shape=(config.sequence_length,),
        dtype=tf.int32,
        name="session_id",
    )

    batch_size_layer = BatchSizeLayer(name="batch_size")
    batch_size = batch_size_layer(data)

    # Static loss scaling factor
    static_loss_scaling_factor_layer = StaticLossScalingFactorLayer(
        name="static_loss_scaling_factor"
    )
    static_loss_scaling_factor = static_loss_scaling_factor_layer(data)

    # Embedding layers
    arrays_layer = TFRangeLayer(config.n_sessions, name="arrays")
    embeddings_layer = layers.Embedding(
        config.n_sessions,
        config.embeddings_dim,
        name="embeddings",
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

    arrays = arrays_layer(data)
    embeddings = embeddings_layer(arrays)

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
        means_spatial_embeddings_layer = layers.Dense(
            config.spatial_embeddings_dim,
            name="means_spatial_embeddings",
        )
        means_concat_embeddings_layer = ConcatEmbeddingsLayer(
            name="means_concat_embeddings",
        )
        means_dev_decoder_layer = MultiLayerPerceptronLayer(
            config.dev_n_layers,
            config.dev_n_units,
            config.dev_normalization,
            config.dev_activation,
            config.dev_dropout,
            config.dev_regularizer,
            config.dev_regularizer_factor,
            name="means_dev_decoder",
        )
        means_dev_map_layer = layers.Dense(
            config.n_channels,
            name="means_dev_map",
        )
        norm_means_dev_map_layer = layers.LayerNormalization(
            axis=-1, scale=False, name="norm_means_dev_map"
        )

        means_dev_mag_inf_alpha_input_layer = LearnableTensorLayer(
            shape=(config.n_sessions, config.n_states, 1),
            learn=config.learn_means,
            initializer=osld_initializers.RandomWeightInitializer(
                tfp.math.softplus_inverse(config.initial_dev.get("means_alpha", 0.0)),
                0.1,
            ),
            name="means_dev_mag_inf_alpha_input",
        )
        means_dev_mag_inf_alpha_layer = layers.Activation(
            "softplus", name="means_dev_mag_inf_alpha"
        )
        means_dev_mag_inf_beta_input_layer = LearnableTensorLayer(
            shape=(config.n_sessions, config.n_states, 1),
            learn=config.learn_means,
            initializer=osld_initializers.RandomWeightInitializer(
                tfp.math.softplus_inverse(config.initial_dev.get("means_beta", 5.0)),
                0.1,
            ),
            name="means_dev_mag_inf_beta_input",
        )
        means_dev_mag_inf_beta_layer = layers.Activation(
            "softplus", name="means_dev_mag_inf_beta"
        )
        means_dev_mag_layer = SampleGammaDistributionLayer(
            config.covariances_epsilon, config.do_kl_annealing, name="means_dev_mag"
        )

        means_dev_layer = layers.Multiply(name="means_dev")

        # Data flow to get mean deviations

        # Get the concatenated embeddings
        means_spatial_embeddings = means_spatial_embeddings_layer(group_mu)
        means_concat_embeddings = means_concat_embeddings_layer(
            [embeddings, means_spatial_embeddings]
        )

        # Get the mean deviation maps (no global magnitude information)
        means_dev_decoder = means_dev_decoder_layer(
            means_concat_embeddings,
            static_loss_scaling_factor=static_loss_scaling_factor,
        )
        means_dev_map = means_dev_map_layer(means_dev_decoder)
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
            [means_dev_mag_inf_alpha, means_dev_mag_inf_beta, session_id]
        )
        norm_means_dev_map = tf.gather(norm_means_dev_map, session_id[:, 0], axis=0)
        means_dev = means_dev_layer([means_dev_mag, norm_means_dev_map])
    else:
        means_dev_layer = ZeroLayer(
            shape=(config.n_states, config.n_channels),
            name="means_dev",
        )
        means_dev = tf.broadcast_to(
            means_dev_layer(data),
            (batch_size, config.n_states, config.n_channels),
        )

    # ----------------------
    # Covariances deviations

    # Layer definitions
    if config.learn_covariances:
        covs_spatial_embeddings_layer = layers.Dense(
            config.spatial_embeddings_dim,
            name="covs_spatial_embeddings",
        )
        covs_concat_embeddings_layer = ConcatEmbeddingsLayer(
            name="covs_concat_embeddings",
        )

        covs_dev_decoder_layer = MultiLayerPerceptronLayer(
            config.dev_n_layers,
            config.dev_n_units,
            config.dev_normalization,
            config.dev_activation,
            config.dev_dropout,
            config.dev_regularizer,
            config.dev_regularizer_factor,
            name="covs_dev_decoder",
        )
        covs_dev_map_layer = layers.Dense(
            config.n_channels * (config.n_channels + 1) // 2,
            name="covs_dev_map",
        )
        norm_covs_dev_map_layer = layers.LayerNormalization(
            axis=-1, scale=False, name="norm_covs_dev_map"
        )

        covs_dev_mag_inf_alpha_input_layer = LearnableTensorLayer(
            shape=(config.n_sessions, config.n_states, 1),
            learn=config.learn_covariances,
            initializer=osld_initializers.RandomWeightInitializer(
                tfp.math.softplus_inverse(config.initial_dev.get("covs_alpha", 0.0)),
                0.1,
            ),
            name="covs_dev_mag_inf_alpha_input",
        )
        covs_dev_mag_inf_alpha_layer = layers.Activation(
            "softplus", name="covs_dev_mag_inf_alpha"
        )
        covs_dev_mag_inf_beta_input_layer = LearnableTensorLayer(
            shape=(config.n_sessions, config.n_states, 1),
            learn=config.learn_covariances,
            initializer=osld_initializers.RandomWeightInitializer(
                tfp.math.softplus_inverse(config.initial_dev.get("covs_beta", 5.0)),
                0.1,
            ),
            name="covs_dev_mag_inf_beta_input",
        )
        covs_dev_mag_inf_beta_layer = layers.Activation(
            "softplus", name="covs_dev_mag_inf_beta"
        )
        covs_dev_mag_layer = SampleGammaDistributionLayer(
            config.covariances_epsilon, config.do_kl_annealing, name="covs_dev_mag"
        )
        covs_dev_layer = layers.Multiply(name="covs_dev")

        # Data flow to get covariances deviations

        # Get the concatenated embeddings
        covs_spatial_embeddings = covs_spatial_embeddings_layer(
            InverseCholeskyLayer(config.covariances_epsilon)(group_D)
        )
        covs_concat_embeddings = covs_concat_embeddings_layer(
            [embeddings, covs_spatial_embeddings]
        )

        # Get the covariance deviation maps (no global magnitude information)
        covs_dev_decoder = covs_dev_decoder_layer(
            covs_concat_embeddings,
            static_loss_scaling_factor=static_loss_scaling_factor,
        )
        covs_dev_map = covs_dev_map_layer(covs_dev_decoder)
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
            [covs_dev_mag_inf_alpha, covs_dev_mag_inf_beta, session_id]
        )
        # covs_dev_mag.shape = (None, n_states, 1)
        norm_covs_dev_map = tf.gather(norm_covs_dev_map, session_id[:, 0], axis=0)
        # norm_covs_dev_map.shape = (None, n_states, n_channels * (n_channels + 1) // 2)
        covs_dev = covs_dev_layer([covs_dev_mag, norm_covs_dev_map])
        # covs_dev.shape = (None, n_states, n_channels * (n_channels + 1) // 2)
    else:
        covs_dev_layer = ZeroLayer(
            shape=(
                config.n_states,
                config.n_channels * (config.n_channels + 1) // 2,
            ),
            name="covs_dev",
        )
        covs_dev = tf.broadcast_to(
            covs_dev_layer(data),
            (
                batch_size,
                config.n_states,
                config.n_channels * (config.n_channels + 1) // 2,
            ),
        )

    # ----------------------------------------
    # Add deviations to group level parameters

    # Layer definitions
    session_means_layer = SessionMapLayer(
        "means", config.covariances_epsilon, name="session_means"
    )
    array_covs_layer = SessionMapLayer(
        "covariances", config.covariances_epsilon, name="array_covs"
    )

    # Data flow
    mu = session_means_layer(
        [group_mu, means_dev]
    )  # shape = (None, n_states, n_channels)
    D = array_covs_layer(
        [group_D, covs_dev]
    )  # shape = (None, n_states, n_channels, n_channels)

    # -----------------------------------
    # Get the log likelihood

    # Layer definitions
    ll_layer = SeparateLogLikelihoodLayer(
        config.n_states, config.covariances_epsilon, name="ll"
    )

    # Data flow
    ll = ll_layer([data, mu, D])

    # Hidden state inference
    hidden_state_inference_layer = HiddenMarkovStateInferenceLayer(
        config.n_states,
        config.initial_trans_prob,
        config.learn_trans_prob,
        dtype="float64",
        name="hid_state_inf",
    )
    gamma, xi = hidden_state_inference_layer(ll)

    # Loss
    ll_loss_layer = SumLogLikelihoodLossLayer(name="ll_loss")
    ll_loss = ll_loss_layer([ll, gamma])

    # ---------
    # KL losses

    # For the observation model (static KL loss)
    if config.learn_means:
        # Layer definitions
        means_dev_mag_mod_beta_layer = layers.Dense(
            1,
            activation="softplus",
            name="means_dev_mag_mod_beta",
        )

        means_dev_mag_kl_loss_layer = StaticKLDivergenceLayer(
            config.covariances_epsilon, name="means_dev_mag_kl_loss"
        )

        # Data flow
        means_dev_mag_mod_beta = means_dev_mag_mod_beta_layer(means_dev_decoder)
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
        covs_dev_mag_mod_beta_layer = layers.Dense(
            1,
            activation="softplus",
            name="covs_dev_mag_mod_beta",
        )

        covs_dev_mag_kl_loss_layer = StaticKLDivergenceLayer(
            config.covariances_epsilon, name="covs_dev_mag_kl_loss"
        )

        # Data flow
        covs_dev_mag_mod_beta = covs_dev_mag_mod_beta_layer(
            covs_dev_decoder,
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
    kl_loss_layer = KLLossLayer(do_annealing=config.do_kl_annealing, name="kl_loss")

    # Data flow
    kl_loss = kl_loss_layer([means_dev_mag_kl_loss, covs_dev_mag_kl_loss])

    return tf.keras.Model(
        inputs=[data, session_id],
        outputs=[ll_loss, kl_loss, gamma, xi],
        name="HIVE",
    )
