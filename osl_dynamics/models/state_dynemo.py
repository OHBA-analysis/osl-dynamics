"""State-Dynamic Network Modelling (State-DyNeMo).

"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.models.inf_mod_base import VariationalInferenceModelConfig
from osl_dynamics.models.dynemo import Model as DyNeMo
from osl_dynamics.inference.layers import (
    InferenceRNNLayer,
    ModelRNNLayer,
    SoftmaxLayer,
    SampleGumbelSoftmaxDistributionLayer,
    MeanVectorsLayer,
    CovarianceMatricesLayer,
    CategoricalLogLikelihoodLossLayer,
    CategoricalKLDivergenceLayer,
    KLLossLayer,
)


@dataclass
class Config(BaseModelConfig, VariationalInferenceModelConfig):
    """Settings for State-DyNeMo.

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
    inference_dropout : float
        Dropout rate.
    inference_regularizer : str
        Regularizer.

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
    model_dropout : float
        Dropout rate.
    model_regularizer : str
        Regularizer.

    learn_means : bool
        Should we make the mean vectors for each mode trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each mode trainable?
    initial_means : np.ndarray
        Initialisation for mean vectors.
    initial_covariances : np.ndarray
        Initialisation for mode covariances.

    do_kl_annealing : bool
        Should we use KL annealing during training?
    kl_annealing_curve : str
        Type of KL annealing curve. Either 'linear' or 'tanh'.
    kl_annealing_sharpness : float
        Parameter to control the shape of the annealing curve if
        kl_annealing_curve='tanh'.
    n_kl_annealing_epochs : int
        Number of epochs to perform KL annealing.

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

    model_name: str = "State-DyNeMo"

    # Inference network parameters
    inference_rnn: Literal["gru", "lstm"] = "lstm"
    inference_n_layers: int = 1
    inference_n_units: int = None
    inference_normalization: Literal[None, "batch", "layer"] = None
    inference_activation: str = None
    inference_dropout: float = 0.0
    inference_regularizer: str = None

    # Model network parameters
    model_rnn: Literal["gru", "lstm"] = "lstm"
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: Literal[None, "batch", "layer"] = None
    model_activation: str = None
    model_dropout: float = 0.0
    model_regularizer: str = None

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None

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


class Model(DyNeMo):
    """State-DyNeMo model class.

    Parameters
    ----------
    config : osl_dynamics.models.state_dynemo.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def sample_alpha(self, n_samples):
        """Uses the model RNN to sample a state time course, alpha."""
        raise NotImplementedError("This method hasn't been coded yet.")


def _model_structure(config):

    # Layer for input
    data = layers.Input(shape=(config.sequence_length, config.n_channels), name="data")

    # Inference RNN:
    # - q(state_t) = softmax(theta_t), where theta_t is a set of logits
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
        initial_temperature=1.0, learn_temperature=False, name="alpha"
    )
    states_layer = SampleGumbelSoftmaxDistributionLayer(name="states")

    inf_rnn = inf_rnn_layer(data)
    inf_theta = inf_theta_layer(inf_rnn)
    alpha = alpha_layer(inf_theta)
    states = states_layer(inf_theta)

    # Observation model:
    # - p(x_t) = N(m_t, C_t), where m_t and C_t are state dependent
    #   means/covariances
    means_layer = MeanVectorsLayer(
        config.n_states,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        name="means",
    )
    covs_layer = CovarianceMatricesLayer(
        config.n_states,
        config.n_channels,
        config.learn_covariances,
        config.initial_covariances,
        name="covs",
    )
    ll_loss_layer = CategoricalLogLikelihoodLossLayer(config.n_states, name="ll_loss")

    mu = means_layer(data)  # data not used
    D = covs_layer(data)  # data not used
    ll_loss = ll_loss_layer([data, mu, D, alpha])

    # Model RNN:
    # - p(theta_t | state_<t), predicts logits for the next state based
    #   on a history of states.
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
    kl_div_layer = CategoricalKLDivergenceLayer(name="kl_div")
    kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")

    mod_rnn = mod_rnn_layer(states)
    mod_theta = mod_theta_layer(mod_rnn)
    kl_div = kl_div_layer([inf_theta, mod_theta])
    kl_loss = kl_loss_layer(kl_div)

    return tf.keras.Model(
        inputs=data, outputs=[ll_loss, kl_loss, alpha], name="State-DyNeMo"
    )
