"""Data classes containing settings for models.

Each class is a container for model settings. The classes also validate the input.

"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tensorflow.python.distribute.distribution_strategy_context import get_strategy


@dataclass
class Alpha:
    """Parameters related to alpha.

    Parameters
    ----------
    theta_normalization : str
        Type of normalization to apply to the posterior samples, theta.
        Either 'layer', 'batch' or None.
    xform : str
        Functional form of alpha. Either 'gumbel-softmax', 'softmax' or 'softplus'.
    initial_temperature : float
        Initial value for the alpha temperature if it is being learnt or if
        we are performing alpha temperature annealing
    learn_temperature : bool
        Should we learn the alpha temperature when xform = 'softmax' or
        'gumbel-softmax'?
    do_annealing : bool
        Should we perform alpha temperature annealing. Can be used when
        xform = 'softmax' or 'gumbel-softmax'.
    final_temperature : bool
        Final value for the alpha temperature if we are annealing.
    n_epochs_annealing : int
        Number of alpha temperature annealing epochs.
    """

    theta_normalization: Literal[None, "batch", "layer"] = None
    xform: Literal["gumbel-softmax", "softmax", "softplus"] = None

    learn_temperature: bool = None
    initial_temperature: float = None

    do_annealing: bool = None
    final_temperature: float = None
    n_epochs_annealing: int = None

    def __post_init__(self):

        if self.theta_normalization not in [None, "batch", "layer"]:
            raise ValueError("normalization must be None, 'batch' or 'layer'.")

        if self.xform not in ["gumbel-softmax", "softmax", "softplus"]:
            raise ValueError("xform must be 'gumbel-softmax', 'softmax' or 'softplus'.")

        if "softmax" in self.xform:
            if self.learn_temperature is None and self.do_annealing is None:
                raise ValueError(
                    "Either learn_temperature or do_annealing must be passed."
                )

            if self.initial_temperature is None:
                raise ValueError("initial_temperature must be passed.")

            if self.initial_temperature <= 0:
                raise ValueError("initial_temperature must be greater than zero.")

            if self.do_annealing:
                if self.final_temperature is None:
                    raise ValueError(
                        "If we are performing alpha temperature annealing, "
                        + "final_temperature must be passed."
                    )

                if self.final_temperature <= 0:
                    raise ValueError("final_temperature must be greater than zero.")

                if self.n_epochs_annealing is None:
                    raise ValueError(
                        "If we are performing alpha temperature annealing, "
                        + "n_epochs_annealing must be passed."
                    )

                if self.n_epochs_annealing < 1:
                    raise ValueError("n_epochs_annealing must be one or above.")

        elif self.xform == "softplus":
            self.initial_temperature = 1.0
            self.learn_temperature = False


@dataclass
class Dimensions:
    """Data dimensions.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    """

    n_states: int = None
    n_channels: int = None
    sequence_length: int = None

    def __post_init__(self):

        if self.n_states is None:
            raise ValueError("n_states must be passed.")

        if self.n_channels is None:
            raise ValueError("n_channels must be passed.")

        if self.sequence_length is None:
            raise ValueError("sequence_length must be passed.")

        if self.n_states < 1:
            raise ValueError("n_states must be one or greater.")

        if self.n_channels < 1:
            raise ValueError("n_channels must be one or greater.")

        if self.sequence_length < 1:
            raise ValueError("sequence_length must be one or greater.")


@dataclass
class KLAnnealing:
    """Parameters related to KL annealing.

    Parameters
    ----------
    do : bool
        Should we use KL annealing during training?
    curve : str
        Type of KL annealing curve. Either 'linear' or 'tanh'.
    sharpness : float
        Parameter to control the shape of the annealing curve if curve='tanh'.
    n_epochs : int
        Number of epochs to perform KL annealing.
    """

    do: bool = None
    curve: Literal["linear", "tanh"] = None
    sharpness: float = None
    n_epochs: int = None

    def __post_init__(self):

        if self.do is None:
            raise ValueError("do must be passed.")

        if self.do:
            if self.curve is None:
                raise ValueError(
                    "If we are performing KL annealing, curve must be passed."
                )

            if self.curve not in ["linear", "tanh"]:
                raise ValueError("Annealing curve must be 'linear' or 'tanh'.")

            if self.n_epochs is None:
                raise ValueError(
                    "If we are performing KL annealing, n_epochs must be passed."
                )

            if self.n_epochs < 1:
                raise ValueError(
                    "Number of annealing epochs must be greater than zero."
                )

            if self.curve == "tanh":
                if self.sharpness is None:
                    raise ValueError("sharpness must be passed if curve='tanh'.")

                if self.sharpness < 0:
                    raise ValueError("Annealing sharpness must be positive.")


@dataclass
class ObservationModel:
    """Observation model parameters.

    Parameters
    ----------
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
    """

    model: str = None

    learn_covariances: bool = None
    learn_alpha_scaling: bool = None
    normalize_covariances: bool = None
    initial_covariances: np.ndarray = None

    n_lags: int = None
    learn_coeffs: bool = None
    learn_cov: bool = None
    initial_coeffs: np.ndarray = None
    initial_cov: np.ndarray = None

    def __post_init__(self):

        if self.model not in [
            "multivariate_normal",
            "multivariate_autoregressive",
        ]:
            raise ValueError(
                "observation_model must be 'multivariable_normal' or "
                + "'multivariate_autoregressive'."
            )

        if self.model == "multivariate_normal":
            if self.learn_covariances is None:
                self.learn_covariances = True

            if self.learn_alpha_scaling is None:
                self.learn_alpha_scaling = False

            if self.normalize_covariances is None:
                self.normalize_covariances = False

        elif self.model == "multivariate_autoregressive":
            if self.n_lags is None:
                raise ValueError(
                    "If model='multivariate_autoregressive', n_lags must be passed."
                )

            if self.learn_coeffs is None:
                self.learn_coeffs = True

            if self.learn_cov is None:
                self.learn_cov = True


@dataclass
class RNN:
    """Recurrent Neural Network (RNN) parameters.

    Parameters
    ----------
    rnn : str
        RNN to use, either 'gru' or 'lstm'.
    n_layers : int
        Number of layers.
    n_units : int
        Number of units.
    normalization : str
        Type of normalization to use. Either None, 'batch' or 'layer'.
    dropout_rate : float
        Dropout rate.
    """

    rnn: Literal["gru", "lstm"] = None
    n_layers: int = None
    n_units: int = None
    normalization: Literal[None, "batch", "layer"] = None
    dropout_rate: float = None

    def __post_init__(self):

        if self.rnn not in ["gru", "lstm"]:
            raise ValueError("rnn must be 'gru' or 'lstm'.")

        if self.n_layers is None:
            raise ValueError("n_layers must be passed.")

        if self.n_layers < 1:
            raise ValueError("n_layers must be one or greater.")

        if self.n_units is None:
            raise ValueError("n_units must be passed.")

        if self.n_units < 1:
            raise ValueError("n_units must be one or greater.")

        if self.normalization not in [None, "batch", "layer"]:
            raise ValueError("normalization must be None, 'batch' or 'layer'.")

        if self.dropout_rate is None:
            raise ValueError("dropout_rate must be passed.")

        if self.dropout_rate < 0:
            raise ValueError("dropout_rate must be greater than zero.")


@dataclass
class Training:
    """Training parameters.

    Parameters
    ----------
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

    batch_size: int = None
    learning_rate: float = None
    n_epochs: int = None
    optimizer: tensorflow.keras.optimizers.Optimizer = None
    multi_gpu: bool = None
    strategy: str = None

    def __post_init__(self):

        if self.batch_size is None:
            raise ValueError("batch_size must be passed.")

        if self.batch_size < 1:
            raise ValueError("batch_size must be one or greater.")

        if self.n_epochs is None:
            raise ValueError("n_epochs must be passed.")

        if self.n_epochs < 1:
            raise ValueError("n_epochs must be one or greater.")

        if self.learning_rate is None:
            raise ValueError("learning_rate must be passed.")

        if self.learning_rate < 0:
            raise ValueError("learning_rate must be greater than zero.")

        # Optimizer
        if self.optimizer is None:
            self.optimizer = "adam"

        if self.optimizer not in ["adam", "Adam"]:
            raise NotImplementedError("Please use optimizer='adam'.")

        self.optimizer = tensorflow.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        if self.multi_gpu is None:
            self.multi_gpu = True

        # Strategy for distributed learning
        if self.multi_gpu:
            self.strategy = MirroredStrategy()
        elif self.strategy is None:
            self.strategy = get_strategy()
