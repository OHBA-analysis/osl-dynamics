"""Multi-Dynamic Network Modes (M-DyNeMo).

See Also
--------
`Example script <https://github.com/OHBA-analysis/osl-dynamics/blob/main\
/examples/simulation/mdynemo_hmm-mvn.py>`_ for training M-DyNeMo on simulated
data (with multiple dynamics).
"""

import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm.auto import trange

from osl_dynamics.inference.layers import (
    ConcatenateLayer,
    CorrelationMatricesLayer,
    DiagonalMatricesLayer,
    InferenceRNNLayer,
    KLDivergenceLayer,
    KLLossLayer,
    LogLikelihoodLossLayer,
    MatMulLayer,
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

_logger = logging.getLogger("osl-dynamics")


@dataclass
class Config(BaseModelConfig, VariationalInferenceModelConfig):
    """Settings for M-DyNeMo.

    Parameters
    ----------
    model_name : str
        Model name.
    n_modes : int
        Number of modes for both power.
    n_fc_modes : int
        Number of modes for functional connectivity.
        If :code:`None`, then set to :code:`n_modes`.
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
        The same parameter is used for the :code:`gamma` time course.
    learn_alpha_temperature : bool
        Should we learn the :code:`alpha_temperature`?
        The same parameter is used for the :code:`gamma` time course.
    initial_alpha_temperature : float
        Initial value for :code:`alpha_temperature`.
        The same parameter is used for the :code:`gamma` time course.

    learn_means : bool
        Should we make the mean for each mode trainable?
    learn_stds : bool
        Should we make the standard deviation for each mode trainable?
    learn_fcs : bool
        Should we make the functional connectivity for each mode trainable?
    initial_means : np.ndarray
        Initialisation for the mode means.
    initial_stds : np.ndarray
        Initialisation for mode standard deviations.
    initial_fcs : np.ndarray
        Initialisation for mode functional connectivity matrices.
    stds_epsilon : float
        Error added to mode stds for numerical stability.
    fcs_epsilon : float
        Error added to mode fcs for numerical stability.
    means_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for the mean vectors.
    stds_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for the standard deviation vectors.
    fcs_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for the correlation matrices.

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
    """

    model_name: str = "M-DyNeMo"

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
    n_fc_modes: int = None
    learn_means: bool = None
    learn_stds: bool = None
    learn_fcs: bool = None
    initial_means: np.ndarray = None
    initial_stds: np.ndarray = None
    initial_fcs: np.ndarray = None
    stds_epsilon: float = None
    fcs_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    stds_regularizer: tf.keras.regularizers.Regularizer = None
    fcs_regularizer: tf.keras.regularizers.Regularizer = None
    multiple_dynamics: bool = True

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
        if (
            self.learn_means is None
            or self.learn_stds is None
            or self.learn_fcs is None
        ):
            raise ValueError("learn_means, learn_stds and learn_fcs must be passed.")

        if self.stds_epsilon is None:
            if self.learn_stds:
                self.stds_epsilon = 1e-6
            else:
                self.stds_epsilon = 0.0

        if self.fcs_epsilon is None:
            if self.learn_fcs:
                self.fcs_epsilon = 1e-6
            else:
                self.fcs_epsilon = 0.0

    def validate_dimension_parameters(self):
        super().validate_dimension_parameters()
        if self.n_fc_modes is None:
            self.n_fc_modes = self.n_modes
            _logger.warning("n_fc_modes is None, set to n_modes.")


class Model(VariationalInferenceModelBase):
    """M-DyNeMo model class.

    Parameters
    ----------
    config : osl_dynamics.models.mdynemo.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_means(self):
        """Get the mode means.

        Returns
        -------
        means : np.ndarray
            Mode means. Shape (n_modes, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "means")

    def get_stds(self):
        """Get the mode standard deviations.

        Returns
        -------
        stds : np.ndarray
            Mode standard deviations. Shape (n_modes, n_channels, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "stds")

    def get_fcs(self):
        """Get the mode functional connectivities.

        Returns
        -------
        fcs : np.ndarray
            Mode functional connectivities.
            Shape (n_modes, n_channels, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "fcs")

    def get_means_stds_fcs(self):
        """Get the mode means, standard deviations, functional connectivities.

        This is a wrapper for :code:`get_means`, :code:`get_stds`,
        :code:`get_fcs`.

        Returns
        -------
        means : np.ndarray
            Mode means. Shape is (n_modes, n_channels).
        stds : np.ndarray
            Mode standard deviations.
            Shape is (n_modes, n_channels, n_channels).
        fcs : np.ndarray
            Mode functional connectivities.
            Shape is (n_modes, n_channels, n_channels).
        """
        return self.get_means(), self.get_stds(), self.get_fcs()

    def get_observation_model_parameters(self):
        """Wrapper for :code:`get_means_stds_fcs`."""
        return self.get_means_stds_fcs()

    def set_means(self, means, update_initializer=True):
        """Set the mode means.

        Parameters
        ----------
        means : np.ndarray
            Mode means. Shape is (n_modes, n_channels).
        update_initializer : bool
            Do we want to use the passed parameters when we re-initialize
            the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            means,
            layer_name="means",
            update_initializer=update_initializer,
        )

    def set_stds(self, stds, update_initializer=True):
        """Set the mode standard deviations.

        Parameters
        ----------
        stds : np.ndarray
            Mode standard deviations.
            Shape is (n_modes, n_channels, n_channels) or (n_modes, n_channels).
        update_initializer : bool
            Do we want to use the passed parameters when we re-initialize
            the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            stds,
            layer_name="stds",
            update_initializer=update_initializer,
        )

    def set_fcs(self, fcs, update_initializer=True):
        """Set the mode functional connectivities.

        Parameters
        ----------
        fcs : np.ndarray
            Mode functional connectivities.
            Shape is (n_modes, n_channels, n_channels).
        update_initializer : bool
            Do we want to use the passed parameters when we re-initialize
            the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            fcs,
            layer_name="fcs",
            update_initializer=update_initializer,
        )

    def set_means_stds_fcs(self, means, stds, fcs, update_initializer=True):
        """This is a wrapper for set_means, set_stds, set_fcs."""
        self.set_means(means, update_initializer=update_initializer)
        self.set_stds(stds, update_initializer=update_initializer)
        self.set_fcs(fcs, update_initializer=update_initializer)

    def set_observation_model_parameters(
        self, observation_model_parameters, update_initializer=True
    ):
        """Wrapper for set_means_stds_fcs."""
        self.set_means_stds_fcs(
            observation_model_parameters[0],
            observation_model_parameters[1],
            observation_model_parameters[2],
            update_initializer=update_initializer,
        )

    def set_regularizers(self, training_dataset):
        """Set the regularizers of means, stds and fcs based on the training
        data.

        A multivariate normal prior is applied to the mean vectors with
        :code:`mu=0`, :code:`sigma=diag((range/2)**2)`, a log normal prior is
        applied to the standard deviations with :code:`mu=0`,
        :code:`sigma=sqrt(log(2*range))` and a marginal inverse Wishart prior
        is applied to the functional connectivity matrices with
        :code:`nu=n_channels-1+0.1`.

        Parameters
        ----------
        training_dataset : tf.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        """
        training_dataset = self.make_dataset(training_dataset, concatenate=True)

        if self.config.learn_means:
            obs_mod.set_means_regularizer(self.model, training_dataset)

        if self.config.learn_stds:
            obs_mod.set_stds_regularizer(
                self.model, training_dataset, self.config.stds_epsilon
            )

        if self.config.learn_fcs:
            obs_mod.set_fcs_regularizer(
                self.model, training_dataset, self.config.fcs_epsilon
            )

    def sample_time_courses(self, n_samples: int):
        """Uses the model RNN to sample mode mixing factors,
        :code:`alpha` and :code:`gamma`.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.

        Returns
        -------
        alpha : np.ndarray
            Sampled :code:`alpha`.
        gamma : np.ndarray
            Sampled :code:`gamma`.
        """
        # Get layers
        model_rnn_layer = self.model.get_layer("mod_rnn")
        mean_mod_mu_layer = self.model.get_layer("mean_mod_mu")
        mean_mod_sigma_layer = self.model.get_layer("mean_mod_sigma")
        mean_theta_norm_layer = self.model.get_layer("mean_theta_norm")
        alpha_layer = self.model.get_layer("alpha")
        fc_mod_mu_layer = self.model.get_layer("fc_mod_mu")
        fc_mod_sigma_layer = self.model.get_layer("fc_mod_sigma")
        fc_theta_norm_layer = self.model.get_layer("fc_theta_norm")
        gamma_layer = self.model.get_layer("gamma")
        concatenate_layer = self.model.get_layer("theta_norm")

        # Normally distributed random numbers used to sample the logits theta
        mean_epsilon = np.random.normal(
            0, 1, [n_samples + 1, self.config.n_modes]
        ).astype(np.float32)
        fc_epsilon = np.random.normal(
            0, 1, [n_samples + 1, self.config.n_fc_modes]
        ).astype(np.float32)

        # Initialise sequence of underlying logits theta
        mean_theta_norm = np.zeros(
            [self.config.sequence_length, self.config.n_modes],
            dtype=np.float32,
        )
        mean_theta_norm[-1] = np.random.normal(size=self.config.n_modes)
        fc_theta_norm = np.zeros(
            [self.config.sequence_length, self.config.n_fc_modes],
            dtype=np.float32,
        )
        fc_theta_norm[-1] = np.random.normal(size=self.config.n_fc_modes)

        # Sample the mode time courses
        alpha = np.empty([n_samples, self.config.n_modes])
        gamma = np.empty([n_samples, self.config.n_fc_modes])
        for i in trange(n_samples, desc="Sampling mode time courses"):
            # If there are leading zeros we trim theta so that we don't pass
            # the zeros
            trimmed_mean_theta = mean_theta_norm[~np.all(mean_theta_norm == 0, axis=1)][
                np.newaxis, :, :
            ]
            trimmed_fc_theta = fc_theta_norm[~np.all(fc_theta_norm == 0, axis=1)][
                np.newaxis, :, :
            ]
            trimmed_theta = concatenate_layer([trimmed_mean_theta, trimmed_fc_theta])
            # p(theta|theta_<t) ~ N(mod_mu, sigma_theta_jt)
            model_rnn = model_rnn_layer(trimmed_theta)
            mean_mod_mu = mean_mod_mu_layer(model_rnn)[0, -1]
            mean_mod_sigma = mean_mod_sigma_layer(model_rnn)[0, -1]
            fc_mod_mu = fc_mod_mu_layer(model_rnn)[0, -1]
            fc_mod_sigma = fc_mod_sigma_layer(model_rnn)[0, -1]

            # Shift theta one time step to the left
            mean_theta_norm = np.roll(mean_theta_norm, -1, axis=0)
            fc_theta_norm = np.roll(fc_theta_norm, -1, axis=0)

            # Sample from the probability distribution function
            mean_theta = mean_mod_mu + mean_mod_sigma * mean_epsilon[i]
            mean_theta_norm[-1] = mean_theta_norm_layer(
                mean_theta[np.newaxis, np.newaxis, :]
            )
            fc_theta = fc_mod_mu + fc_mod_sigma * fc_epsilon[i]
            fc_theta_norm[-1] = fc_theta_norm_layer(fc_theta[np.newaxis, np.newaxis, :])

            alpha[i] = alpha_layer(mean_theta_norm[-1][np.newaxis, np.newaxis, :])[0, 0]
            gamma[i] = gamma_layer(fc_theta_norm[-1][np.newaxis, np.newaxis, :])[0, 0]

        return alpha, gamma

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
                or "gamma" in var_name
                or "means" in var_name
                or "stds" in var_name
                or "fcs" in var_name
            ):
                n_params += np.prod(var.shape)

            return int(n_params)


def _model_structure(config):
    # Layer for input
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )

    # Static loss scaling factor
    static_loss_scaling_factor_layer = StaticLossScalingFactorLayer(
        name="static_loss_scaling_factor"
    )
    static_loss_scaling_factor = static_loss_scaling_factor_layer(inputs)

    #
    # Inference RNN
    #

    # Layers
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

    # Data flow
    data_drop = data_drop_layer(inputs)
    inf_rnn = inf_rnn_layer(data_drop)

    #
    # Mode time course for the mean and standard deviation
    #

    # Layers
    mean_inf_mu_layer = layers.Dense(config.n_modes, name="mean_inf_mu")
    mean_inf_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="mean_inf_sigma"
    )
    mean_theta_layer = SampleNormalDistributionLayer(
        config.theta_std_epsilon, name="mean_theta"
    )
    mean_theta_norm_layer = NormalizationLayer(
        config.theta_normalization, name="mean_theta_norm"
    )
    alpha_layer = SoftmaxLayer(
        config.initial_alpha_temperature,
        config.learn_alpha_temperature,
        name="alpha",
    )

    # Data flow
    mean_inf_mu = mean_inf_mu_layer(inf_rnn)
    mean_inf_sigma = mean_inf_sigma_layer(inf_rnn)
    mean_theta = mean_theta_layer([mean_inf_mu, mean_inf_sigma])
    mean_theta_norm = mean_theta_norm_layer(mean_theta)
    alpha = alpha_layer(mean_theta_norm)

    #
    # Mode time course for the FCs
    #

    # Layers
    fc_inf_mu_layer = layers.Dense(config.n_fc_modes, name="fc_inf_mu")
    fc_inf_sigma_layer = layers.Dense(
        config.n_fc_modes, activation="softplus", name="fc_inf_sigma"
    )
    fc_theta_layer = SampleNormalDistributionLayer(
        config.theta_std_epsilon, name="fc_theta"
    )
    fc_theta_norm_layer = NormalizationLayer(
        config.theta_normalization, name="fc_theta_norm"
    )
    gamma_layer = SoftmaxLayer(
        config.initial_alpha_temperature,
        config.learn_alpha_temperature,
        name="gamma",
    )

    # Data flow
    fc_inf_mu = fc_inf_mu_layer(inf_rnn)
    fc_inf_sigma = fc_inf_sigma_layer(inf_rnn)
    fc_theta = fc_theta_layer([fc_inf_mu, fc_inf_sigma])
    fc_theta_norm = fc_theta_norm_layer(fc_theta)
    gamma = gamma_layer(fc_theta_norm)

    #
    # Observation model
    #

    # Layers
    means_layer = VectorsLayer(
        config.n_modes,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        config.means_regularizer,
        name="means",
    )
    stds_layer = DiagonalMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_stds,
        config.initial_stds,
        config.stds_epsilon,
        config.stds_regularizer,
        name="stds",
    )
    fcs_layer = CorrelationMatricesLayer(
        config.n_fc_modes,
        config.n_channels,
        config.learn_fcs,
        config.initial_fcs,
        config.fcs_epsilon,
        config.fcs_regularizer,
        name="fcs",
    )
    mix_means_layer = MixVectorsLayer(name="mix_means")
    mix_stds_layer = MixMatricesLayer(name="mix_stds")
    mix_fcs_layer = MixMatricesLayer(name="mix_fcs")
    matmul_layer = MatMulLayer(name="cov")
    ll_loss_layer = LogLikelihoodLossLayer(
        np.maximum(config.stds_epsilon, config.fcs_epsilon), name="ll_loss"
    )

    # Data flow
    mu = means_layer(
        inputs, static_loss_scaling_factor=static_loss_scaling_factor
    )  # inputs not used
    E = stds_layer(
        inputs, static_loss_scaling_factor=static_loss_scaling_factor
    )  # inputs not used
    D = fcs_layer(
        inputs, static_loss_scaling_factor=static_loss_scaling_factor
    )  # inputs not used

    m = mix_means_layer([alpha, mu])
    G = mix_stds_layer([alpha, E])
    F = mix_fcs_layer([gamma, D])
    C = matmul_layer([G, F, G])

    ll_loss = ll_loss_layer([inputs, m, C])

    #
    # Model RNN
    #

    # Layers
    concatenate_layer = ConcatenateLayer(axis=2, name="theta_norm")
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

    # Data flow
    theta_norm = concatenate_layer([mean_theta_norm, fc_theta_norm])
    theta_norm_drop = theta_norm_drop_layer(theta_norm)
    mod_rnn = mod_rnn_layer(theta_norm_drop)

    #
    # Mode time course for the mean
    #

    # Layers
    mean_mod_mu_layer = layers.Dense(config.n_modes, name="mean_mod_mu")
    mean_mod_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="mean_mod_sigma"
    )
    kl_div_layer_mean = KLDivergenceLayer(
        config.theta_std_epsilon,
        name="mean_kl_div",
    )

    # Data flow
    mean_mod_mu = mean_mod_mu_layer(mod_rnn)
    mean_mod_sigma = mean_mod_sigma_layer(mod_rnn)
    mean_kl_div = kl_div_layer_mean(
        [mean_inf_mu, mean_inf_sigma, mean_mod_mu, mean_mod_sigma]
    )

    #
    # Mode time course for the functional connectivity
    #

    # Layers
    fc_mod_mu_layer = layers.Dense(config.n_fc_modes, name="fc_mod_mu")
    fc_mod_sigma_layer = layers.Dense(
        config.n_fc_modes, activation="softplus", name="fc_mod_sigma"
    )
    fc_kl_div_layer = KLDivergenceLayer(
        config.theta_std_epsilon,
        name="fc_kl_div",
    )

    # Data flow
    fc_mod_mu = fc_mod_mu_layer(mod_rnn)
    fc_mod_sigma = fc_mod_sigma_layer(mod_rnn)
    fc_kl_div = fc_kl_div_layer(
        [fc_inf_mu, fc_inf_sigma, fc_mod_mu, fc_mod_sigma],
    )

    #
    # Total KL loss
    #
    kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")
    kl_loss = kl_loss_layer([mean_kl_div, fc_kl_div])

    return tf.keras.Model(
        inputs=inputs,
        outputs=[ll_loss, kl_loss, mean_theta_norm, fc_theta_norm],
        name="M-DyNeMo",
    )
