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
from tqdm.auto import trange

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference.layers import (
    TFConcatLayer,
    CorrelationMatricesLayer,
    DiagonalMatricesLayer,
    InferenceRNNLayer,
    KLDivergenceLayer,
    KLLossLayer,
    LogLikelihoodLossLayer,
    TFMatMulLayer,
    MixMatricesLayer,
    MixVectorsLayer,
    ModelRNNLayer,
    NormalizationLayer,
    SampleNormalDistributionLayer,
    SoftmaxLayer,
    VectorsLayer,
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
        Number of modes.
    n_corr_modes : int
        Number of modes for correlation.
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

    learn_means : bool
        Should we make the mean for each mode trainable?
    learn_stds : bool
        Should we make the standard deviation for each mode trainable?
    learn_corrs : bool
        Should we make the correlation for each mode trainable?
    initial_means : np.ndarray
        Initialisation for the mode means.
    initial_stds : np.ndarray
        Initialisation for mode standard deviations.
    initial_corrs : np.ndarray
        Initialisation for mode correlation matrices.
    stds_epsilon : float
        Error added to mode stds for numerical stability.
    corrs_epsilon : float
        Error added to mode corrs for numerical stability.
    means_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for the mean vectors.
    stds_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for the standard deviation vectors.
    corrs_regularizer : tf.keras.regularizers.Regularizer
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

    init_method : str
        Initialization method to use. Defaults to 'random_subset'.
    n_init : int
        Number of initializations. Defaults to 5.
    n_init_epochs : int
        Number of epochs for each initialization. Defaults to 2.
    init_take : float
        Fraction of dataset to use in the initialization.
        Defaults to 1.0.

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
    best_of : int
        Number of full training runs to perform. A single run includes
        its own initialization and fitting from scratch.
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
    n_corr_modes: int = None
    learn_means: bool = None
    learn_stds: bool = None
    learn_corrs: bool = None
    initial_means: np.ndarray = None
    initial_stds: np.ndarray = None
    initial_corrs: np.ndarray = None
    stds_epsilon: float = None
    corrs_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    stds_regularizer: tf.keras.regularizers.Regularizer = None
    corrs_regularizer: tf.keras.regularizers.Regularizer = None
    multiple_dynamics: bool = True

    pca_components: np.ndarray = None

    # Initialization
    init_method: str = "random_subset"
    n_init: int = 5
    n_init_epochs: int = 2
    init_take: float = 1.0

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
            or self.learn_corrs is None
        ):
            raise ValueError("learn_means, learn_stds and learn_corrs must be passed.")

        if self.stds_epsilon is None:
            if self.learn_stds:
                self.stds_epsilon = 1e-6
            else:
                self.stds_epsilon = 0.0

        if self.corrs_epsilon is None:
            if self.learn_corrs:
                self.corrs_epsilon = 1e-6
            else:
                self.corrs_epsilon = 0.0

        if self.pca_components is None:
            self.pca_components = np.eye(self.n_channels)
        self.pca_components = self.pca_components.astype(np.float32)

    def validate_dimension_parameters(self):
        super().validate_dimension_parameters()
        if self.n_corr_modes is None:
            self.n_corr_modes = self.n_modes
            _logger.warning("n_corr_modes is None, set to n_modes.")


class Model(VariationalInferenceModelBase):
    """M-DyNeMo model class.

    Parameters
    ----------
    config : osl_dynamics.models.mdynemo.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""

        config = self.config

        # ---------- Define layers ---------- #
        data_drop_layer = tf.keras.layers.Dropout(
            config.inference_dropout, name="data_drop"
        )
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
        power_inf_mu_layer = tf.keras.layers.Dense(config.n_modes, name="power_inf_mu")
        power_inf_sigma_layer = tf.keras.layers.Dense(
            config.n_modes, activation="softplus", name="power_inf_sigma"
        )
        power_theta_layer = SampleNormalDistributionLayer(
            config.theta_std_epsilon, name="power_theta"
        )
        fc_inf_mu_layer = tf.keras.layers.Dense(config.n_corr_modes, name="fc_inf_mu")
        fc_inf_sigma_layer = tf.keras.layers.Dense(
            config.n_corr_modes, activation="softplus", name="fc_inf_sigma"
        )
        fc_theta_layer = SampleNormalDistributionLayer(
            config.theta_std_epsilon, name="fc_theta"
        )
        alpha_layer = SoftmaxLayer(
            initial_temperature=1.0,
            learn_temperature=False,
            name="alpha",
        )
        beta_layer = SoftmaxLayer(
            initial_temperature=1.0,
            learn_temperature=False,
            name="beta",
        )
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
        corrs_layer = CorrelationMatricesLayer(
            config.n_corr_modes,
            config.n_channels,
            config.learn_corrs,
            config.initial_corrs,
            config.corrs_epsilon,
            config.corrs_regularizer,
            name="corrs",
        )

        class PCATransformLayer(tf.keras.layers.Layer):
            def __init__(self, pca_components, **kwargs):
                super(PCATransformLayer, self).__init__(**kwargs)
                self.pca_components = tf.Variable(
                    initial_value=tf.convert_to_tensor(
                        pca_components, dtype=tf.float32
                    ),
                    trainable=False,
                )

            def call(self, inputs):
                mu, E, R = inputs
                pca_mu = tf.squeeze(
                    tf.matmul(
                        tf.expand_dims(tf.transpose(self.pca_components), 0),
                        tf.expand_dims(mu, -1),
                    )
                )
                pca_E = tf.matmul(
                    tf.matmul(
                        tf.expand_dims(tf.transpose(self.pca_components), 0),
                        E,
                    ),
                    tf.expand_dims(self.pca_components, 0),
                )
                pca_R = tf.matmul(
                    tf.matmul(
                        tf.expand_dims(tf.transpose(self.pca_components), 0),
                        R,
                    ),
                    tf.expand_dims(self.pca_components, 0),
                )
                return pca_mu, pca_E, pca_R

        pca_layer = PCATransformLayer(config.pca_components, name="pca")
        mix_means_layer = MixVectorsLayer(name="mix_means")
        mix_stds_layer = MixMatricesLayer(name="mix_stds")
        mix_corrs_layer = MixMatricesLayer(name="mix_corrs")
        matmul_layer = TFMatMulLayer(name="cov")
        ll_loss_layer = LogLikelihoodLossLayer(config.loss_calc, name="ll_loss")
        theta_layer = TFConcatLayer(axis=2, name="theta")
        theta_drop_layer = tf.keras.layers.Dropout(
            config.model_dropout,
            name="theta_drop",
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
        power_mod_mu_layer = tf.keras.layers.Dense(config.n_modes, name="power_mod_mu")
        power_mod_sigma_layer = tf.keras.layers.Dense(
            config.n_modes, activation="softplus", name="power_mod_sigma"
        )
        fc_mod_mu_layer = tf.keras.layers.Dense(config.n_corr_modes, name="fc_mod_mu")
        fc_mod_sigma_layer = tf.keras.layers.Dense(
            config.n_corr_modes, activation="softplus", name="fc_mod_sigma"
        )
        fc_kl_div_layer = KLDivergenceLayer(
            config.theta_std_epsilon,
            config.loss_calc,
            name="fc_kl_div",
        )
        kl_div_layer_power = KLDivergenceLayer(
            config.theta_std_epsilon,
            config.loss_calc,
            name="power_kl_div",
        )
        kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")

        # ---------- Forward pass ---------- #

        # Encoder
        data = tf.keras.layers.Input(
            shape=(config.sequence_length, config.n_channels), name="data"
        )
        data_drop = data_drop_layer(data)
        inf_rnn = inf_rnn_layer(data_drop)
        power_inf_mu = power_inf_mu_layer(inf_rnn)
        power_inf_sigma = power_inf_sigma_layer(inf_rnn)
        power_theta = power_theta_layer([power_inf_mu, power_inf_sigma])
        fc_inf_mu = fc_inf_mu_layer(inf_rnn)
        fc_inf_sigma = fc_inf_sigma_layer(inf_rnn)
        fc_theta = fc_theta_layer([fc_inf_mu, fc_inf_sigma])
        alpha = alpha_layer(power_theta)
        beta = beta_layer(fc_theta)

        # Observation model
        mu = means_layer(data)
        E = stds_layer(data)
        R = corrs_layer(data)
        pca_mu, pca_E, pca_R = pca_layer([mu, E, R])
        m = mix_means_layer([alpha, pca_mu])
        G = mix_stds_layer([alpha, pca_E])
        F = mix_corrs_layer([beta, pca_R])
        C = matmul_layer([G, F, G])
        ll_loss = ll_loss_layer([data, m, C])

        # Decoder
        theta = theta_layer([power_theta, fc_theta])
        theta_drop = theta_drop_layer(theta)
        mod_rnn = mod_rnn_layer(theta_drop)
        power_mod_mu = power_mod_mu_layer(mod_rnn)
        power_mod_sigma = power_mod_sigma_layer(mod_rnn)
        fc_mod_mu = fc_mod_mu_layer(mod_rnn)
        fc_mod_sigma = fc_mod_sigma_layer(mod_rnn)
        power_kl_div = kl_div_layer_power(
            [power_inf_mu, power_inf_sigma, power_mod_mu, power_mod_sigma]
        )
        fc_kl_div = fc_kl_div_layer(
            [fc_inf_mu, fc_inf_sigma, fc_mod_mu, fc_mod_sigma],
        )
        kl_loss = kl_loss_layer([power_kl_div, fc_kl_div])

        # ---------- Create model ---------- #
        inputs = {"data": data}
        outputs = {
            "ll_loss": ll_loss,
            "kl_loss": kl_loss,
            "power_theta": power_theta,
            "fc_theta": fc_theta,
        }
        name = config.model_name
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

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

    def get_corrs(self):
        """Get the mode correlations.

        Returns
        -------
        corrs : np.ndarray
            Mode correlations.
            Shape (n_modes, n_channels, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "corrs")

    def get_means_stds_corrs(self):
        """Get the mode means, standard deviations, correlations.

        This is a wrapper for :code:`get_means`, :code:`get_stds`,
        :code:`get_corrs`.

        Returns
        -------
        means : np.ndarray
            Mode means. Shape is (n_modes, n_channels).
        stds : np.ndarray
            Mode standard deviations.
            Shape is (n_modes, n_channels, n_channels).
        corrs : np.ndarray
            Mode correlations.
            Shape is (n_modes, n_channels, n_channels).
        """
        return self.get_means(), self.get_stds(), self.get_corrs()

    def get_observation_model_parameters(self):
        """Wrapper for :code:`get_means_stds_corrs`."""
        return self.get_means_stds_corrs()

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

    def set_corrs(self, corrs, update_initializer=True):
        """Set the mode correlations.

        Parameters
        ----------
        corrs : np.ndarray
            Mode correlations.
            Shape is (n_modes, n_channels, n_channels).
        update_initializer : bool
            Do we want to use the passed parameters when we re-initialize
            the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            corrs,
            layer_name="corrs",
            update_initializer=update_initializer,
        )

    def set_means_stds_corrs(self, means, stds, corrs, update_initializer=True):
        """This is a wrapper for set_means, set_stds, set_corrs."""
        self.set_means(means, update_initializer=update_initializer)
        self.set_stds(stds, update_initializer=update_initializer)
        self.set_corrs(corrs, update_initializer=update_initializer)

    def set_observation_model_parameters(
        self, observation_model_parameters, update_initializer=True
    ):
        """Wrapper for set_means_stds_corrs."""
        self.set_means_stds_corrs(
            observation_model_parameters[0],
            observation_model_parameters[1],
            observation_model_parameters[2],
            update_initializer=update_initializer,
        )

    def set_regularizers(self, training_dataset):
        """Set the regularizers of means, stds and corrs based on the training
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
        _logger.info("Setting regularizers")

        training_dataset = self.make_dataset(
            training_dataset, shuffle=False, concatenate=True
        )
        n_sequences, range_ = dtf.get_n_sequences_and_range(training_dataset)
        scale_factor = self.get_static_loss_scaling_factor(n_sequences)

        if self.config.learn_means:
            obs_mod.set_means_regularizer(self.model, range_, scale_factor)

        if self.config.learn_stds:
            obs_mod.set_stds_regularizer(
                self.model,
                range_,
                self.config.stds_epsilon,
                scale_factor,
            )

        if self.config.learn_corrs:
            obs_mod.set_corrs_regularizer(
                self.model,
                self.config.n_channels,
                self.config.corrs_epsilon,
                scale_factor,
            )

    def sample_time_courses(self, n_samples):
        """Uses the model RNN to sample mode mixing factors,
        :code:`alpha` and :code:`beta`.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.

        Returns
        -------
        alpha : np.ndarray
            Sampled :code:`alpha`.
        beta : np.ndarray
            Sampled :code:`beta`.
        """
        # Get layers
        model_rnn_layer = self.model.get_layer("mod_rnn")
        power_mod_mu_layer = self.model.get_layer("power_mod_mu")
        power_mod_sigma_layer = self.model.get_layer("power_mod_sigma")
        alpha_layer = self.model.get_layer("alpha")
        fc_mod_mu_layer = self.model.get_layer("fc_mod_mu")
        fc_mod_sigma_layer = self.model.get_layer("fc_mod_sigma")
        beta_layer = self.model.get_layer("beta")
        theta_layer = self.model.get_layer("theta")

        # Normally distributed random numbers used to sample the logits theta
        power_epsilon = np.random.normal(
            0, 1, [n_samples + 1, self.config.n_modes]
        ).astype(np.float32)
        fc_epsilon = np.random.normal(
            0, 1, [n_samples + 1, self.config.n_corr_modes]
        ).astype(np.float32)

        # Initialise sequence of underlying logits theta
        power_theta = np.zeros(
            [self.config.sequence_length, self.config.n_modes],
            dtype=np.float32,
        )
        power_theta[-1] = np.random.normal(size=self.config.n_modes)
        fc_theta = np.zeros(
            [self.config.sequence_length, self.config.n_corr_modes],
            dtype=np.float32,
        )
        fc_theta[-1] = np.random.normal(size=self.config.n_corr_modes)

        # Sample the mode time courses
        alpha = np.empty([n_samples, self.config.n_modes])
        beta = np.empty([n_samples, self.config.n_corr_modes])
        for i in trange(n_samples, desc="Sampling mode time courses"):
            # If there are leading zeros we trim theta so that we don't pass
            # the zeros
            trimmed_power_theta = power_theta[~np.all(power_theta == 0, axis=1)][
                np.newaxis, :, :
            ]
            trimmed_fc_theta = fc_theta[~np.all(fc_theta == 0, axis=1)][
                np.newaxis, :, :
            ]
            trimmed_theta = theta_layer([trimmed_power_theta, trimmed_fc_theta])
            # p(theta|theta_<t) ~ N(mod_mu, sigma_theta_jt)
            model_rnn = model_rnn_layer(trimmed_theta)
            power_mod_mu = power_mod_mu_layer(model_rnn)[0, -1]
            power_mod_sigma = power_mod_sigma_layer(model_rnn)[0, -1]
            fc_mod_mu = fc_mod_mu_layer(model_rnn)[0, -1]
            fc_mod_sigma = fc_mod_sigma_layer(model_rnn)[0, -1]

            # Shift theta one time step to the left
            power_theta = np.roll(power_theta, -1, axis=0)
            fc_theta = np.roll(fc_theta, -1, axis=0)

            # Sample from the probability distribution function
            power_theta[-1] = power_mod_mu + power_mod_sigma * power_epsilon[i]
            fc_theta[-1] = fc_mod_mu + fc_mod_sigma * fc_epsilon[i]

            alpha[i] = alpha_layer(power_theta[-1][np.newaxis, np.newaxis, :])[0, 0]
            beta[i] = beta_layer(fc_theta[-1][np.newaxis, np.newaxis, :])[0, 0]

        return alpha, beta
