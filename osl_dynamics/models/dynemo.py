"""Dynamic Network Modes (DyNeMo).

See the `documentation <https://osl-dynamics.readthedocs.io/en/latest/models\
/dynemo.html>`_ for a description of this model.
"""

import os
import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm.auto import trange

from osl_dynamics.inference.layers import (
    CovarianceMatricesLayer,
    DiagonalMatricesLayer,
    InferenceRNNLayer,
    KLDivergenceLayer,
    KLLossLayer,
    LogLikelihoodLossLayer,
    MixMatricesLayer,
    MixVectorsLayer,
    ModelRNNLayer,
    NormalizationLayer,
    SampleNormalDistributionLayer,
    SoftmaxLayer,
    VectorsLayer,
    StaticLossScalingFactorLayer,
)
from osl_dynamics.models import obs_mod
from osl_dynamics.models.inf_mod_base import (
    VariationalInferenceModelBase,
    VariationalInferenceModelConfig,
)
from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.utils.misc import set_logging_level

_logger = logging.getLogger("osl-dynamics")


@dataclass
class Config(BaseModelConfig, VariationalInferenceModelConfig):
    """Settings for DyNeMo.

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
    inference_n_units : int
        Number of units.
    inference_normalization : str
        Type of normalization to use. Either :code:`None`, :code:`'batch'` or
        :code:`'layer'`.
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
        Type of normalization to use. Either None, :code:`'batch'` or
        :code:`'layer'`.
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
        Initialisation for state covariances.
        If :code:`diagonal_covariances=True` and full matrices are passed,
        the diagonal is extracted.
    covariances_epsilon : float
        Error added to mode covariances for numerical stability.
    diagonal_covariances : bool
        Should we learn diagonal mode covariances?
    means_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for mean vectors.
    covariances_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for covariance matrices.

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
    loss_calc : str
        How should we collapse the time dimension in the loss?
        Either :code:`'mean'` or :code:`'sum'`.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    """

    model_name: str = "DyNeMo"

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
    diagonal_covariances: bool = False
    covariances_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

    def __post_init__(self):
        self.validate_rnn_parameters()
        self.validate_observation_model_parameters()
        self.validate_alpha_parameters()
        self.validate_kl_annealing_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

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


class Model(VariationalInferenceModelBase):
    """DyNeMo model class.

    Parameters
    ----------
    config : osl_dynamics.models.dynemo.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""
        self.model = self._model_structure()

    def get_means(self):
        """Get the mode means.

        Returns
        -------
        means : np.ndarary
            Mode means.
        """
        return obs_mod.get_observation_model_parameter(self.model, "means")

    def get_covariances(self):
        """Get the mode covariances.

        Returns
        -------
        covariances : np.ndarary
            Mode covariances.
        """
        return obs_mod.get_observation_model_parameter(self.model, "covs")

    def get_means_covariances(self):
        """Get the mode means and covariances.

        This is a wrapper for :code:`get_means` and :code:`get_covariances`.

        Returns
        -------
        means : np.ndarary
            Mode means.
        covariances : np.ndarray
            Mode covariances.
        """
        return self.get_means(), self.get_covariances()

    def get_observation_model_parameters(self):
        """Wrapper for :code:`get_means_covariances`."""
        return self.get_means_covariances()

    def set_means(self, means, update_initializer=True):
        """Set the mode means.

        Parameters
        ----------
        means : np.ndarray
            Mode means. Shape is (n_modes, n_channels).
        update_initializer : bool
            Do we want to use the passed means when we re-initialize
            the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            means,
            layer_name="means",
            update_initializer=update_initializer,
        )

    def set_covariances(self, covariances, update_initializer=True):
        """Set the mode covariances.

        Parameters
        ----------
        covariances : np.ndarray
            Mode covariances. Shape is (n_modes, n_channels, n_channels).
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
        self.set_means(
            means,
            update_initializer=update_initializer,
        )
        self.set_covariances(
            covariances,
            update_initializer=update_initializer,
        )

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
        :code:`config.diagonal_covariances=True`, a log normal prior is
        applied to the diagonal of the covariances matrices with :code:`mu=0`,
        :code:`sigma=sqrt(log(2*range))`, otherwise an inverse Wishart prior
        is applied to the covariances matrices with :code:`nu=n_channels-1+0.1`
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

    def sample_alpha(self, n_samples, theta_norm=None):
        """Uses the model RNN to sample mode mixing factors, :code:`alpha`.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.
        theta_norm : np.ndarray, optional
            Normalized logits to initialise the sampling with.
            Shape must be (sequence_length, n_modes).

        Returns
        -------
        alpha : np.ndarray
            Sampled alpha.
        """
        # Get layers
        model_rnn_layer = self.model.get_layer("mod_rnn")
        mod_mu_layer = self.model.get_layer("mod_mu")
        mod_sigma_layer = self.model.get_layer("mod_sigma")
        theta_norm_layer = self.model.get_layer("theta_norm")
        alpha_layer = self.model.get_layer("alpha")

        # Normally distributed random numbers used to sample the logits theta
        epsilon = np.random.normal(0, 1, [n_samples + 1, self.config.n_modes]).astype(
            np.float32
        )

        if theta_norm is None:
            # Sequence of the underlying logits theta
            theta_norm = np.zeros(
                [self.config.sequence_length, self.config.n_modes],
                dtype=np.float32,
            )

            # Randomly sample the first time step
            theta_norm[-1] = np.random.normal(size=self.config.n_modes)

        # Sample the mode fixing factors
        alpha = np.empty([n_samples, self.config.n_modes], dtype=np.float32)
        for i in trange(n_samples, desc="Sampling mode time course"):
            # If there are leading zeros we trim theta so that we don't pass
            # the zeros
            trimmed_theta = theta_norm[~np.all(theta_norm == 0, axis=1)][
                np.newaxis, :, :
            ]

            # Predict the probability distribution function for theta one time
            # step in the future, p(theta|theta_<t) ~ N(mod_mu, sigma_theta_jt)
            model_rnn = model_rnn_layer(trimmed_theta)
            mod_mu = mod_mu_layer(model_rnn)[0, -1]
            mod_sigma = mod_sigma_layer(model_rnn)[0, -1]

            # Shift theta one time step to the left
            theta_norm = np.roll(theta_norm, -1, axis=0)

            # Sample from the probability distribution function
            theta = mod_mu + mod_sigma * epsilon[i]
            theta_norm[-1] = theta_norm_layer(theta[np.newaxis, np.newaxis, :])[0]

            # Calculate the mode mixing factors
            alpha[i] = alpha_layer(mod_mu[np.newaxis, np.newaxis, :])[0, 0]

        return alpha

    def get_n_params_generative_model(self):
        """Get the number of trainable parameters in the generative model.

        This includes the model RNN weights and biases, mixing coefficients,
        mode means and covariances.

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
                or "means" in var_name
                or "covs" in var_name
            ):
                n_params += np.prod(var.shape)

        return int(n_params)

    def fine_tuning(
        self, training_data, n_epochs=None, learning_rate=None, store_dir="tmp"
    ):
        """Fine tuning the model for each session.

        Here, we train the inference RNN and observation model with the model
        RNN fixed held fixed at the group-level.

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
            Session-specific mixing coefficients.
            Each element has shape (n_samples, n_modes).
        means : np.ndarray
            Session-specific means. Shape is (n_sessions, n_modes, n_channels).
        covariances : np.ndarray
            Session-specific covariances.
            Shape is (n_sessions, n_modes, n_channels, n_channels).
        """
        # Save the group level model
        os.makedirs(store_dir, exist_ok=True)
        self.save_weights(f"{store_dir}/weights.h5")

        # Save original training hyperparameters
        original_n_epochs = self.config.n_epochs
        original_learning_rate = self.config.learning_rate
        original_do_kl_annealing = self.config.do_kl_annealing
        self.config.n_epochs = n_epochs or self.config.n_epochs
        self.config.learning_rate = learning_rate or self.config.learning_rate
        self.config.do_kl_annealing = False

        # Layers to fix (i.e. make non-trainable)
        fixed_layers = ["mod_rnn", "mod_mu", "mod_sigma"]

        # Fine tune on sessions
        alpha = []
        means = []
        covariances = []
        with self.set_trainable(fixed_layers, False), set_logging_level(
            _logger, logging.WARNING
        ):
            for i in trange(training_data.n_sessions, desc="Fine tuning"):
                # Train on this session
                with training_data.set_keep(i):
                    self.fit(training_data, verbose=0)
                    a = self.get_alpha(
                        training_data,
                        concatenate=True,
                        verbose=0,
                    )

                # Get inferred parameters
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
        self.config.do_kl_annealing = original_do_kl_annealing

        return alpha, np.array(means), np.array(covariances)

    def dual_estimation(
        self,
        training_data,
        n_epochs=None,
        learning_rate=None,
        store_dir="tmp",
    ):
        """Dual estimation to get the session-specific observation model
        parameters.

        Here, we train the observation model parameters (mode means and
        covariances) with the inference RNN and model RNN held fixed at
        the group-level.

        Parameters
        ----------
        training_data : osl_dynamics.data.Data or list of tf.data.Dataset
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
        means : np.ndarray
            Session-specific means. Shape is (n_sessions, n_modes, n_channels).
        covariances : np.ndarray
            Session-specific covariances.
            Shape is (n_sessions, n_modes, n_channels, n_channels).
        """
        # Save the group level model
        os.makedirs(store_dir, exist_ok=True)
        self.save_weights(f"{store_dir}/weights.h5")

        # Save original training hyperparameters
        original_n_epochs = self.config.n_epochs
        original_learning_rate = self.config.learning_rate
        self.config.n_epochs = n_epochs or self.config.n_epochs
        self.config.learning_rate = learning_rate or self.config.learning_rate

        # Layers to fix (i.e. make non-trainable)
        fixed_layers = [
            "mod_rnn",
            "mod_mu",
            "mod_sigma",
            "inf_rnn",
            "inf_mu",
            "inf_sigma",
            "theta_norm",
            "alpha",
        ]

        # Dual estimation on sessions
        means = []
        covariances = []
        with self.set_trainable(fixed_layers, False):
            if isinstance(training_data, list):
                n_sessions = len(training_data)
            else:
                n_sessions = training_data.n_sessions

            for i in trange(n_sessions, desc="Dual estimation"):
                # Train on this session
                if isinstance(training_data, list):
                    self.fit(training_data[i], verbose=0)
                else:
                    with training_data.set_keep(i):
                        self.fit(training_data, verbose=0)

                # Get inferred parameters
                m, c = self.get_means_covariances()
                means.append(m)
                covariances.append(c)

                # Reset back to group-level model parameters
                self.load_weights(f"{store_dir}/weights.h5")
                self.compile()

        # Reset hyperparameters
        self.config.n_epochs = original_n_epochs
        self.config.learning_rate = original_learning_rate

        return np.array(means), np.array(covariances)

    def _select_covariance_layer(self):
        """Select the covariance layer based on the config."""
        config = self.config
        if config.diagonal_covariances:
            CovsLayer = DiagonalMatricesLayer
        else:
            CovsLayer = CovarianceMatricesLayer
        return CovsLayer(
            config.n_modes or config.n_states,
            config.n_channels,
            config.learn_covariances,
            config.initial_covariances,
            config.covariances_epsilon,
            config.covariances_regularizer,
            name="covs",
        )

    def _model_structure(self):
        """Build the model structure."""

        config = self.config

        # Layer for input
        inputs = layers.Input(
            shape=(config.sequence_length, config.n_channels), name="data"
        )

        # Static loss scaling factor
        static_loss_scaling_factor_layer = StaticLossScalingFactorLayer(
            config.sequence_length,
            config.loss_calc,
            name="static_loss_scaling_factor",
        )
        static_loss_scaling_factor = static_loss_scaling_factor_layer(inputs)

        # Inference RNN:
        # - Learns q(theta) ~ N(theta | inf_mu, inf_sigma), where
        #     - inf_mu    ~ affine(RNN(inputs_<=t))
        #     - inf_sigma ~ softplus(RNN(inputs_<=t))

        # Definition of layers
        data_drop_layer = layers.Dropout(config.inference_dropout, name="data_drop")
        inf_rnn_layer = InferenceRNNLayer(
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
        data_drop = data_drop_layer(inputs)
        inf_rnn = inf_rnn_layer(data_drop)
        inf_mu = inf_mu_layer(inf_rnn)
        inf_sigma = inf_sigma_layer(inf_rnn)
        theta = theta_layer([inf_mu, inf_sigma])
        theta_norm = theta_norm_layer(theta)
        alpha = alpha_layer(theta_norm)

        # Observation model:
        # - We use a multivariate normal with a mean vector and covariance matrix
        #   for each mode as the observation model.
        # - We calculate the likelihood of generating the training data with alpha
        #   and the observation model.

        # Definition of layers
        means_layer = VectorsLayer(
            config.n_modes,
            config.n_channels,
            config.learn_means,
            config.initial_means,
            config.means_regularizer,
            name="means",
        )
        covs_layer = self._select_covariance_layer()
        mix_means_layer = MixVectorsLayer(name="mix_means")
        mix_covs_layer = MixMatricesLayer(name="mix_covs")
        ll_loss_layer = LogLikelihoodLossLayer(
            config.covariances_epsilon,
            config.loss_calc,
            name="ll_loss",
        )

        # Data flow
        mu = means_layer(
            inputs, static_loss_scaling_factor=static_loss_scaling_factor
        )  # inputs not used
        D = covs_layer(
            inputs, static_loss_scaling_factor=static_loss_scaling_factor
        )  # inputs not used
        m = mix_means_layer([alpha, mu])
        C = mix_covs_layer([alpha, D])
        ll_loss = ll_loss_layer([inputs, m, C])

        # Model RNN:
        # - Learns p(theta_t |theta_<t) ~ N(theta_t | mod_mu, mod_sigma), where
        #     - mod_mu    ~ affine(RNN(theta_<t))
        #     - mod_sigma ~ softplus(RNN(theta_<t))

        # Definition of layers
        theta_norm_drop_layer = layers.Dropout(
            config.model_dropout,
            name="theta_norm_drop",
        )
        mod_rnn_layer = ModelRNNLayer(
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
        kl_div_layer = KLDivergenceLayer(
            config.theta_std_epsilon, config.loss_calc, name="kl_div"
        )
        kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")

        # Data flow
        theta_norm_drop = theta_norm_drop_layer(theta_norm)
        mod_rnn = mod_rnn_layer(theta_norm_drop)
        mod_mu = mod_mu_layer(mod_rnn)
        mod_sigma = mod_sigma_layer(mod_rnn)
        kl_div = kl_div_layer([inf_mu, inf_sigma, mod_mu, mod_sigma])
        kl_loss = kl_loss_layer(kl_div)

        return tf.keras.Model(
            inputs=inputs,
            outputs=[ll_loss, kl_loss, theta_norm],
            name=config.model_name,
        )
