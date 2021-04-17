"""Data class for model settings.

"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tensorflow.python.distribute.distribution_strategy_context import get_strategy
from vrad.utils.decorators import pass_if_all_none


@dataclass
class Config:
    """Settings for a model in VRAD.

    Alpha Parameters
    ----------------
    alpha_pdf : str
        Probability distribution used to generate alpha. Either 'normal' or
        'dirichlet'.
    theta_normalization : str
        Type of normalization to apply to the posterior samples, theta.
        Either 'layer', 'batch' or None.
    alpha_xform : str
        Functional form of alpha. Either 'gumbel-softmax', 'softmax' or 'softplus'.
    initial_alpha_temperature : float
        Initial value for the alpha temperature if it is being learnt or if
        we are performing alpha temperature annealing
    learn_alpha_temperature : bool
        Should we learn the alpha temperature when alpha_xform = 'softmax' or
        'gumbel-softmax'?
    do_alpha_temperature_annealing : bool
        Should we perform alpha temperature annealing. Can be used when
        alpha_xform = 'softmax' or 'gumbel-softmax'.
    final_alpha_temperature : bool
        Final value for the alpha temperature if we are annealing.
    n_epochs_alpha_temperature_annealing : int
        Number of alpha temperature annealing epochs.

    Dimension Parameters
    --------------------
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.

    Inference Network Parameters
    ----------------------------
    inference_rnn : str
        RNN to use, either 'gru' or 'lstm'.
    inference_n_layers : int
        Number of layers.
    inference_n_units : int
        Number of units.
    inference_normalization : str
        Type of normalization to use. Either None, 'batch' or 'layer'.
    inference_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. 'relu', 'elu', etc.
    inference_dropout_rate : float
        Dropout rate.

    Model Network Parameters
    ------------------------
    model_rnn : str
        RNN to use, either 'gru' or 'lstm'.
    model_n_layers : int
        Number of layers.
    model_n_units : int
        Number of units.
    model_normalization : str
        Type of normalization to use. Either None, 'batch' or 'layer'.
    model_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. 'relu', 'elu', etc.
    model_dropout_rate : float
        Dropout rate.

    Observation Model Parameters
    ----------------------------
    model : str
        Type of observation model.
        Either 'multivariate_normal' or 'multivariate_autoregressive'.
    learn_covariances : bool
        Should we make the covariance matrix for each state trainable?
        Pass if model='multivariate_normal'.
    learn_alpha_scaling : bool
        Should we learn a scaling for alpha? Pass if model='multivariate_normal'.
    normalize_covariances : bool
        Should we trace normalize the state covariances? Pass if
        model='multivariate_normal'.
    initial_covariances : np.ndarray
        Initialisation for state covariances. Pass if model='multivariate_normal'.
    n_lags : int
        Number of autoregressive lags. Pass if model='multivariate_autoregressive'.
    learn_coeffs : bool
        Should we learn the autoregressive coefficients? Pass if
        model='multivariate_autoregressive'.
    learn_cov : bool
        Should we learn the covariances? Pass if model='multivariate_autoregressive'.
    initial_coeffs : np.ndarray
        Initialisation for autoregressive coefficients. Pass if
        model='multivariate_autoregressive'.
    initial_cov : np.ndarray
        Initialisation for covariances. Pass if model='multivariate_autoregressive'.

    KL Annealing Parameters
    -----------------------
    do_kl_annealing : bool
        Should we use KL annealing during training?
    kl_annealing_curve : str
        Type of KL annealing curve. Either 'linear' or 'tanh'.
    kl_annealing_sharpness : float
        Parameter to control the shape of the annealing curve if
        kl_annealing_curve='tanh'.
    n_epochs_kl_annealing : int
        Number of epochs to perform KL annealing.
    n_cycles_kl_annealing : int
        Number of times to perform KL annealing within n_epochs_kl_annealing.

    Training Parameters
    -------------------
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
    n_epochs : int
        Number of training epochs.
    optimizer : tensorflow.keras.optimizers.Optimizer
        Optimizer to use. Must be 'adam'.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    """

    # Parameters related to the model choice
    alpha_pdf: str = "normal"
    observation_model: str = "multivariate_normal"

    # Dimension parameters
    n_states: int = None
    n_channels: int = None
    sequence_length: int = None

    # Inference network parameters
    inference_rnn: Literal["gru", "lstm"] = None
    inference_n_layers: int = None
    inference_n_units: int = None
    inference_normalization: Literal[None, "batch", "layer"] = None
    inference_activation: str = None
    inference_dropout_rate: float = None

    # Model network parameters
    model_rnn: Literal["gru", "lstm"] = None
    model_n_layers: int = None
    model_n_units: int = None
    model_normalization: Literal[None, "batch", "layer"] = None
    model_activation: str = None
    model_dropout_rate: float = None

    # Alpha parameters
    theta_normalization: Literal[None, "batch", "layer"] = None
    alpha_xform: Literal["gumbel-softmax", "softmax", "softplus"] = None

    learn_alpha_temperature: bool = None
    initial_alpha_temperature: float = None

    do_alpha_temperature_annealing: bool = None
    final_alpha_temperature: float = None
    n_epochs_alpha_temperature_annealing: int = None

    # Observation model parameters
    learn_covariances: bool = None
    learn_alpha_scaling: bool = None
    normalize_covariances: bool = None
    initial_covariances: np.ndarray = None

    n_lags: int = None
    learn_coeffs: bool = None
    learn_cov: bool = None
    initial_coeffs: np.ndarray = None
    initial_cov: np.ndarray = None

    # KL annealing parameters
    do_kl_annealing: bool = None
    kl_annealing_curve: Literal["linear", "tanh"] = None
    kl_annealing_sharpness: float = None
    n_epochs_kl_annealing: int = None
    n_cycles_kl_annealing: int = None

    # Training parameters
    batch_size: int = None
    learning_rate: float = None
    n_epochs: int = None
    optimizer: tensorflow.keras.optimizers.Optimizer = None
    multi_gpu: bool = None
    strategy: str = None

    def __post_init__(self):
        validate_model_choice_parameters(
            self.alpha_pdf,
            self.observation_model,
        )
        validate_dimension_parameters(
            self.n_states, self.n_channels, self.sequence_length
        )
        validate_rnn_parameters(
            self.inference_rnn,
            self.inference_n_layers,
            self.inference_n_units,
            self.inference_dropout_rate,
            self.inference_normalization,
        )
        validate_rnn_parameters(
            self.model_rnn,
            self.model_n_layers,
            self.model_n_units,
            self.model_dropout_rate,
            self.model_normalization,
        )
        validate_alpha_parameters(
            self.theta_normalization,
            self.alpha_xform,
            self.learn_alpha_temperature,
            self.initial_alpha_temperature,
            self.do_alpha_temperature_annealing,
            self.final_alpha_temperature,
            self.n_epochs_alpha_temperature_annealing,
        )
        validate_observation_model_parameters(
            self.observation_model,
            self.learn_covariances,
            self.learn_alpha_scaling,
            self.normalize_covariances,
            self.n_lags,
            self.learn_coeffs,
            self.learn_cov,
        )
        validate_kl_annealing_parameters(
            self.do_kl_annealing,
            self.kl_annealing_curve,
            self.kl_annealing_sharpness,
            self.n_epochs_kl_annealing,
            self.n_cycles_kl_annealing,
        )
        validate_training_parameters(
            self.batch_size, self.n_epochs, self.learning_rate, self.optimizer
        )

        # Optimizer
        if self.optimizer is None:
            self.optimizer = "adam"

        if self.optimizer not in ["adam", "Adam"]:
            raise NotImplementedError("Please use optimizer='adam'.")

        self.optimizer = tensorflow.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        # Multi-GPU training
        if self.multi_gpu is None:
            self.multi_gpu = True

        # Strategy for distributed learning
        if self.multi_gpu:
            self.strategy = MirroredStrategy()
        elif self.strategy is None:
            self.strategy = get_strategy()


def validate_model_choice_parameters(alpha_pdf, observation_model):

    if alpha_pdf not in ["normal", "dirichlet"]:
        raise ValueError("alpha_pdf must be 'normal' or 'dirichlet'.")

    if observation_model not in ["multivariate_normal", "multivariate_autoregressive"]:
        raise ValueError(
            "observation_model must be 'multivariate_normal' or "
            + "'multivariate_autoregressive'."
        )


@pass_if_all_none
def validate_alpha_parameters(
    theta_normalization,
    alpha_xform,
    learn_alpha_temperature,
    initial_alpha_temperature,
    do_alpha_temperature_annealing,
    final_alpha_temperature,
    n_epochs_alpha_temperature_annealing,
):
    if theta_normalization not in [None, "batch", "layer"]:
        raise ValueError("normalization must be None, 'batch' or 'layer'.")

    if alpha_xform not in ["gumbel-softmax", "softmax", "softplus"]:
        raise ValueError(
            "alpha_xform must be 'gumbel-softmax', 'softmax' or 'softplus'."
        )

    if "softmax" in alpha_xform:
        if learn_alpha_temperature is None and do_alpha_temperature_annealing is None:
            raise ValueError(
                "Either learn_alpha_temperature or do_alpha_temperature_annealing "
                + "must be passed."
            )

        if initial_alpha_temperature is None:
            raise ValueError("initial_alpha_temperature must be passed.")

        if initial_alpha_temperature <= 0:
            raise ValueError("initial_alpha_temperature must be greater than zero.")

        if do_alpha_temperature_annealing:
            if final_alpha_temperature is None:
                raise ValueError(
                    "If we are performing alpha temperature annealing, "
                    + "final_alpha_temperature must be passed."
                )

            if final_alpha_temperature <= 0:
                raise ValueError("final_alpha_temperature must be greater than zero.")

            if n_epochs_alpha_temperature_annealing is None:
                raise ValueError(
                    "If we are performing alpha temperature annealing, "
                    + "n_epochs_alpha_temperature_annealing must be passed."
                )

            if n_epochs_alpha_temperature_annealing < 1:
                raise ValueError(
                    "n_epochs_alpha_temperature_annealing must be one or above."
                )

    elif alpha_xform == "softplus":
        initial_alpha_temperature = 1.0  # not used in the model
        learn_alpha_temperature = False


def validate_dimension_parameters(n_states, n_channels, sequence_length):

    if sequence_length is None:
        raise ValueError("sequence_length must be passed.")

    if n_states is not None:
        if n_states < 1:
            raise ValueError("n_states must be one or greater.")

    if n_channels is not None:
        if n_channels < 1:
            raise ValueError("n_channels must be one or greater.")

    if sequence_length < 1:
        raise ValueError("sequence_length must be one or greater.")


@pass_if_all_none
def validate_kl_annealing_parameters(
    do_kl_annealing,
    kl_annealing_curve,
    kl_annealing_sharpness,
    n_epochs_kl_annealing,
    n_cycles_kl_annealing,
):
    if do_kl_annealing is None:
        raise ValueError("do_kl_annealing must be passed.")

    if do_kl_annealing:
        if kl_annealing_curve is None:
            raise ValueError(
                "If we are performing KL annealing, kl_annealing_curve must be passed."
            )

        if kl_annealing_curve not in ["linear", "tanh"]:
            raise ValueError("KL annealing curve must be 'linear' or 'tanh'.")

        if kl_annealing_curve == "tanh":
            if kl_annealing_sharpness is None:
                raise ValueError(
                    "kl_annealing_sharpness must be passed if "
                    + "kl_annealing_curve='tanh'."
                )

            if kl_annealing_sharpness < 0:
                raise ValueError("KL annealing sharpness must be positive.")

        if n_epochs_kl_annealing is None:
            raise ValueError(
                "If we are performing KL annealing, n_epochs_kl_annealing must be "
                + "passed."
            )

        if n_epochs_kl_annealing < 1:
            raise ValueError("Number of KL annealing epochs must be greater than zero.")

        if n_cycles_kl_annealing is None:
            raise ValueError(
                "If we are perform KL annealing, n_cycles_kl_annealing must be passed."
            )

        if n_cycles_kl_annealing < 1:
            raise ValueError("n_cycles_kl_annealing must be one or greater.")


def validate_observation_model_parameters(
    observation_model,
    learn_covariances,
    learn_alpha_scaling,
    normalize_covariances,
    n_lags,
    learn_coeffs,
    learn_cov,
):
    if observation_model == "multivariate_normal":
        if learn_covariances is None:
            learn_covariances = True

        if learn_alpha_scaling is None:
            learn_alpha_scaling = False

        if normalize_covariances is None:
            normalize_covariances = False

    elif observation_model == "multivariate_autoregressive":
        if n_lags is None:
            raise ValueError(
                "If model='multivariate_autoregressive', n_lags must be passed."
            )

        if learn_coeffs is None:
            learn_coeffs = True

        if learn_cov is None:
            learn_cov = True


@pass_if_all_none
def validate_rnn_parameters(rnn, n_layers, n_units, dropout_rate, normalization):

    if rnn not in ["gru", "lstm"]:
        raise ValueError("rnn must be 'gru' or 'lstm'.")

    if n_layers is None:
        raise ValueError("n_layers must be passed.")

    if n_layers < 1:
        raise ValueError("n_layers must be one or greater.")

    if n_units is None:
        raise ValueError("n_units must be passed.")

    if n_units < 1:
        raise ValueError("n_units must be one or greater.")

    if dropout_rate is None:
        raise ValueError("dropout_rate must be passed.")

    if dropout_rate < 0:
        raise ValueError("dropout_rate must be greater than zero.")

    if normalization not in [None, "batch", "layer"]:
        raise ValueError("normalization must be None, 'batch' or 'layer'.")


def validate_training_parameters(batch_size, n_epochs, learning_rate, optimizer):

    if batch_size is None:
        raise ValueError("batch_size must be passed.")

    if batch_size < 1:
        raise ValueError("batch_size must be one or greater.")

    if n_epochs is None:
        raise ValueError("n_epochs must be passed.")

    if n_epochs < 1:
        raise ValueError("n_epochs must be one or greater.")

    if learning_rate is None:
        raise ValueError("learning_rate must be passed.")

    if learning_rate < 0:
        raise ValueError("learning_rate must be greater than zero.")
