"""Dynamic Network States (DyNeStE)."""

import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tqdm.auto import trange

from osl_dynamics.inference.layers import (
    CategoricalKLDivergenceLayer,
    CategoricalLogLikelihoodLossLayer,
    CovarianceMatricesLayer,
    DiagonalMatricesLayer,
    InferenceRNNLayer,
    KLLossLayer,
    ModelRNNLayer,
    SampleGumbelSoftmaxDistributionLayer,
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
    """Settings for DyNeStE.

    Parameters
    ----------
    model_name : str
        Model name.
    n_states : int
        Number of states.
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
        Type of normalization to use. Either :code:`None`, :code:`'batch'`or
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
        Should we make the mean vectors for each state trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each state trainable?
    initial_means : np.ndarray
        Initialisation for mean vectors.
    initial_covariances : np.ndarray
        Initialisation for state covariances.
        If :code:`diagonal_covariances=True` and full matrices are passed,
        the diagonal is extracted.
    covariances_epsilon : float
        Error added to standard deviations for numerical stability.
    diagonal_covariances : bool
        Should we learn diagonal state covariances?
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

    model_name: str = "DyNeStE"

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
    """DyNeStE model class.

    Parameters
    ----------
    config : osl_dynamics.models.dyneste.Config
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

    def sample_alpha(self, n_samples, states=None):
        """Uses the model RNN to sample a state probability time course, :code:`alpha`.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.
        states : np.ndarray, optional
            One-hot state vectors to initialize the sampling with.
            Shape must be (sequence_length, n_states).

        Returns
        -------
        alpha : np.ndarray
            Sampled alpha.
        """
        # Get layers
        mod_rnn_layer = self.model.get_layer("mod_rnn")
        mod_theta_layer = self.model.get_layer("mod_theta")
        alpha_layer = self.model.get_layer("alpha")
        states_layer = self.model.get_layer("states")

        # Get the final temperature of Gumbel-Softmax distribution
        final_temperature = tf.cast(states_layer.temperature, tf.float32)

        # Preallocate Gumbel noise
        gumbel_noise = tfp.distributions.Gumbel(loc=0, scale=1).sample(
            [n_samples, self.config.n_states]
        )

        if states is None:
            # Sequence of the underlying state time course
            states = np.zeros(
                [self.config.sequence_length, self.config.n_states],
                dtype=np.float32,
            )

            # Randomly sample the first time step
            init_gs = tfp.distributions.RelaxedOneHotCategorical(
                temperature=final_temperature,
                logits=tf.zeros([self.config.n_states], dtype=tf.float32),
            )
            states[-1] = init_gs.sample()

        # Sample the state probability time course
        alpha = np.empty([n_samples, self.config.n_states], dtype=np.float32)
        for i in trange(n_samples, desc="Sampling state probability time course"):
            # If there are leading zeros we trim the state time course so that
            # we don't pass the zeros
            trimmed_states = states[~np.all(states == 0, axis=1)][np.newaxis, :, :]

            # Predict the probability distribution function for theta one time
            # step in the future, p(state_t|state_<t) ~ Cat(mod_theta)
            mod_rnn = mod_rnn_layer(trimmed_states)
            mod_theta = mod_theta_layer(mod_rnn)[0, -1]

            # Shift the state time course one time step to the left
            states = np.roll(states, -1, axis=0)

            # Sample from the probability distribution function
            states[-1] = tf.nn.softmax((mod_theta + gumbel_noise[i]) / final_temperature, axis=-1)

            # Calculate the state probability time courses
            alpha[i] = alpha_layer(mod_theta[np.newaxis, np.newaxis, :])[0, 0]

        return alpha
    
    def _select_covariance_layer(self):
        """Select the covariance layer based on the config."""
        config = self.config
        if config.diagonal_covariances:
            CovsLayer = DiagonalMatricesLayer
        else:
            CovsLayer = CovarianceMatricesLayer
        return CovsLayer(
            config.n_states,
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

        # Inputs
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
        # - Learns q(state_t) = softmax(inf_theta_t), where 
        #     - inf_theta_t ~ affine(RNN(inputs_<=t)) is a set of logits

        # Definition of layers
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
        inf_theta_layer = layers.Dense(config.n_states, name="inf_theta")
        alpha_layer = SoftmaxLayer(
            initial_temperature=1.0,
            learn_temperature=False,
            name="alpha",
        )
        states_layer = SampleGumbelSoftmaxDistributionLayer(
            temperature=1.0, name="states"
        ) # NOTE: NEED EDIT; temperature as a configurable parameter

        # Data flow
        inf_rnn = inf_rnn_layer(inputs)
        inf_theta = inf_theta_layer(inf_rnn)
        alpha = alpha_layer(inf_theta)
        states = states_layer(inf_theta)

        # Observation model:
        # - We use a multivariate normal with a mean vector and covariance matrix
        #   for each state as the observation model.
        # - We calculate the likelihood of generating the training data with alpha
        #   and the observation model.
        # - p(x_t | theta_tk) = N(mu_k, D_k), where mu_k and D_k are state(k)-dependent
        #   means/covariances.

        # Definition of layers
        means_layer = VectorsLayer(
            config.n_states,
            config.n_channels,
            config.learn_means,
            config.initial_means,
            config.means_regularizer,
            name="means",
        )
        covs_layer = self._select_covariance_layer()
        ll_loss_layer = CategoricalLogLikelihoodLossLayer(
            config.n_states,
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
        ll_loss = ll_loss_layer([inputs, mu, D, alpha, None])

        # Model RNN:
        # - Learns p(state_t | state_<t) ~ Cat(mod_theta_t), where
        #     - mod_theta_t ~ affine(RNN(states_<t)) is a set of logits
        # - Here, the model RNN predicts logits for the next state based
        #   on a history of states.

        # Definition of layers
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
        mod_theta_layer = layers.Dense(config.n_states, name="mod_theta")
        kl_div_layer = CategoricalKLDivergenceLayer(config.loss_calc, name="kl_div")
        kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")

        # Data flow
        mod_rnn = mod_rnn_layer(states)
        mod_theta = mod_theta_layer(mod_rnn)
        kl_div = kl_div_layer([inf_theta, mod_theta])
        kl_loss = kl_loss_layer(kl_div)

        return tf.keras.Model(
            inputs=inputs,
            outputs=[ll_loss, kl_loss, inf_theta],
            name=config.model_name,
        )
