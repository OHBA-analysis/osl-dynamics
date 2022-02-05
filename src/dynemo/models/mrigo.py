"""Model class for a multi-time-scale generative model with Gaussian observations.

"""

from dataclasses import dataclass
from typing import Literal
from tqdm import trange

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from dynemo.models import mgo
from dynemo.models.mod_base import BaseModelConfig, ModelBase
from dynemo.models.inf_mod_base import InferenceModelConfig, InferenceModelBase
from dynemo.inference.layers import (
    InferenceRNNLayer,
    LogLikelihoodLossLayer,
    MeanVectorsLayer,
    DiagonalMatricesLayer,
    CorrelationMatricesLayer,
    MixVectorsLayer,
    MixMatricesLayer,
    ModelRNNLayer,
    NormalizationLayer,
    KLDivergenceLayer,
    KLLossLayer,
    SampleNormalDistributionLayer,
    ThetaActivationLayer,
    ConcatenateLayer,
    MatMulLayer,
    DummyLayer,
)


@dataclass
class Config(BaseModelConfig, InferenceModelConfig):
    """Settings for MRIGO.

    Parameters
    ----------
    n_modes : int
        Number of modes.
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
    inference_dropout_rate : float
        Dropout rate.

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

    theta_normalization : str
        Type of normalization to apply to the posterior samples, theta.
        Either 'layer', 'batch' or None.
    alpha_xform : str
        Functional form of alpha. Either 'gumbel-softmax', 'softmax' or 'softplus'.
    learn_alpha_temperature : bool
        Should we learn the alpha temperature when alpha_xform='softmax' or
        'gumbel-softmax'?
    initial_alpha_temperature : float
        Initial value for the alpha temperature.
    The same parameters are used for beta and gamma time courses.

    learn_means : bool
        Should we make the standard deviation for each mode trainable?
    learn_stds: bool
        Should we make the standard deviation for each mode trainable?
    learn_fcs: bool
        Should we make the functional connectivity for each mode trainable?
    initial_stds: np.ndarray
        Initialisation for mode standard deviations.
    initial_fcs: np.ndarray
        Initialisation for mode functional connectivity matrices.

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

    # Inference network parameters
    inference_rnn: Literal["gru", "lstm"] = None
    inference_n_layers: int = 1
    inference_n_units: int = None
    inference_normalization: Literal[None, "batch", "layer"] = None
    inference_activation: str = None
    inference_dropout_rate: float = 0.0

    # Model network parameters
    model_rnn: Literal["gru", "lstm"] = None
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: Literal[None, "batch", "layer"] = None
    model_activation: str = None
    model_dropout_rate: float = 0.0

    # Observation model parameters
    multiple_scales: bool = True
    learn_means: bool = None
    learn_stds: bool = None
    learn_fcs: bool = None
    initial_means: np.ndarray = None
    initial_stds: np.ndarray = None
    initial_fcs: np.ndarray = None

    def __post_init__(self):
        self.validate_rnn_parameters()
        self.validate_observation_model_parameters()
        self.validate_alpha_parameters()
        self.validate_kl_annealing_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_rnn_parameters(self):
        if self.inference_rnn is None or self.model_rnn is None:
            raise ValueError("Please pass inference_rnn and model_rnn.")

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


class Model(InferenceModelBase):
    """Multi-time-scale RNN Inference/model network and Gaussian Observations (MRIGO).

    Parameters
    ----------
    config : dynemo.models.mrigo.Config
    """

    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_means_stds_fcs(self):
        """Get the mean, standard devation and functional connectivity of each mode.

        Returns
        -------
        means : np.ndarray
            Mode means.
        stds : np.ndarray
            Mode standard deviations.
        fcs : np.ndarray
            Mode functional connectivities.
        """
        return mgo.get_means_stds_fcs(self.model)

    def set_means_stds_fcs(self, means, stds, fcs, update_initializer=True):
        """Set the means, standard deviations, functional connectivities of each mode.

        Parameters
        ----------
        means: np.ndarray
            Mode means with shape (n_modes, n_channels).
        stds: np.ndarray
            Mode standard deviations with shape (n_modes, n_channels) or
            (n_modes, n_channels, n_channels).
        fcs: np.ndarray
            Mode functional connectivities with shape (n_modes, n_channels, n_channels).
        update_initializer: bool
            Do we want to use the passed parameters when we re_initialize
            the model?
        """
        mgo.set_means_stds_fcs(self.model, means, stds, fcs, update_initializer)
    
    def sample_time_courses(self, n_samples: int):
        """Uses the model RNN to sample mode mixing factors, alpha, beta and gamma.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.

        Returns
        -------
        List containing sampled alpha, beta, and gamma.
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
        mean_epsilon = np.random.normal(0, 1, [n_samples + 1, self.config.n_modes]).astype(
            np.float32
        )
        fc_epsilon = np.random.normal(0, 1, [n_samples + 1, self.config.n_modes]).astype(
            np.float32
        )

        # Initialise sequence of underlying logits theta
        mean_theta_norm = np.zeros(
            [self.config.sequence_length, self.config.n_modes],
            dtype=np.float32,
        )
        mean_theta_norm[-1] = np.random.normal(size=self.config.n_modes)
        fc_theta_norm = np.zeros(
            [self.config.sequence_length, self.config.n_modes],
            dtype=np.float32,
        )
        fc_theta_norm[-1] = np.random.normal(size=self.config.n_modes)

        # Sample the mode time courses
        alpha = np.empty([n_samples, self.config.n_modes])
        gamma = np.empty([n_samples, self.config.n_modes])
        for i in trange(n_samples, desc="Sampling mode time courses", ncols=98):
            # If there are leading zeros we trim theta so that we don't pass the zeros
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
            mean_theta_norm[-1] = mean_theta_norm_layer(mean_theta[np.newaxis, np.newaxis, :])
            fc_theta = fc_mod_mu + fc_mod_sigma * fc_epsilon[i]
            fc_theta_norm[-1] = fc_theta_norm_layer(fc_theta[np.newaxis, np.newaxis, :])

            alpha[i] = alpha_layer(mean_theta_norm[-1][np.newaxis, np.newaxis, :])[0, 0]
            gamma[i] = gamma_layer(fc_theta_norm[-1][np.newaxis, np.newaxis, :])[0, 0]
        
        return alpha, alpha, gamma


def _model_structure(config):

    # Layer for input
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )

    #
    # Inference RNN
    #

    # Layers
    inference_input_dropout_layer = layers.Dropout(
        config.inference_dropout_rate, name="data_drop"
    )
    inference_output_layer = InferenceRNNLayer(
        config.inference_rnn,
        config.inference_normalization,
        config.inference_activation,
        config.inference_n_layers,
        config.inference_n_units,
        config.inference_dropout_rate,
        name="inf_rnn",
    )

    # Data flow
    inference_input_dropout = inference_input_dropout_layer(inputs)
    inference_output = inference_output_layer(inference_input_dropout)

    #
    # Mode time course for the mean
    #

    # Layers
    mean_inf_mu_layer = layers.Dense(config.n_modes, name="mean_inf_mu")
    mean_inf_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="mean_inf_sigma"
    )
    mean_theta_layer = SampleNormalDistributionLayer(name="mean_theta")
    mean_theta_norm_layer = NormalizationLayer(
        config.theta_normalization, name="mean_theta_norm"
    )
    alpha_layer = ThetaActivationLayer(
        config.alpha_xform,
        config.initial_alpha_temperature,
        config.learn_alpha_temperature,
        name="alpha",
    )

    # Data flow
    mean_inf_mu = mean_inf_mu_layer(inference_output)
    mean_inf_sigma = mean_inf_sigma_layer(inference_output)
    mean_theta = mean_theta_layer([mean_inf_mu, mean_inf_sigma])
    mean_theta_norm = mean_theta_norm_layer(mean_theta)
    alpha = alpha_layer(mean_theta_norm)

    #
    # Mode time course for the standard deviation
    #
    beta_layer = DummyLayer(name="beta")
    beta = beta_layer(alpha)

    #
    # Mode time course for the FCs
    #

    # Layers
    fc_inf_mu_layer = layers.Dense(config.n_modes, name="fc_inf_mu")
    fc_inf_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="fc_inf_sigma"
    )
    fc_theta_layer = SampleNormalDistributionLayer(name="fc_theta")
    fc_theta_norm_layer = NormalizationLayer(
        config.theta_normalization, name="fc_theta_norm"
    )
    gamma_layer = ThetaActivationLayer(
        config.alpha_xform,
        config.initial_alpha_temperature,
        config.learn_alpha_temperature,
        name="gamma",
    )

    # Data flow
    fc_inf_mu = fc_inf_mu_layer(inference_output)
    fc_inf_sigma = fc_inf_sigma_layer(inference_output)
    fc_theta = fc_theta_layer([fc_inf_mu, fc_inf_sigma])
    fc_theta_norm = fc_theta_norm_layer(fc_theta)
    gamma = gamma_layer(fc_theta_norm)

    #
    # Observation model
    #

    # Layers
    means_layer = MeanVectorsLayer(
        config.n_modes,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        name="means",
    )
    stds_layer = DiagonalMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_stds,
        config.initial_stds,
        name="stds",
    )
    fcs_layer = CorrelationMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_fcs,
        config.initial_fcs,
        name="fcs",
    )
    mix_means_layer = MixVectorsLayer(name="mix_means")
    mix_stds_layer = MixMatricesLayer(name="mix_stds")
    mix_fcs_layer = MixMatricesLayer(name="mix_fcs")
    matmul_layer = MatMulLayer(name="cov")
    ll_loss_layer = LogLikelihoodLossLayer(name="ll_loss")

    # Data flow
    mu = means_layer(inputs)  # inputs not used
    E = stds_layer(inputs)  # inputs not used
    D = fcs_layer(inputs)  # inputs not used

    m = mix_means_layer([alpha, mu])
    G = mix_stds_layer([beta, E])
    F = mix_fcs_layer([gamma, D])
    C = matmul_layer([G, F, G])

    ll_loss = ll_loss_layer([inputs, m, C])

    #
    # Model RNN
    #

    # Layers
    concatenate_layer = ConcatenateLayer(axis=2, name="theta_norm")
    model_input_dropout_layer = layers.Dropout(
        config.model_dropout_rate, name="theta_norm_drop"
    )
    model_output_layer = ModelRNNLayer(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout_rate,
        name="mod_rnn",
    )

    # Data flow
    theta_norm = concatenate_layer([mean_theta_norm, fc_theta_norm])
    model_input_dropout = model_input_dropout_layer(theta_norm)
    model_output = model_output_layer(model_input_dropout)

    #
    # Mode time course for the mean
    #

    # Layers
    mean_mod_mu_layer = layers.Dense(config.n_modes, name="mean_mod_mu")
    mean_mod_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="mean_mod_sigma"
    )
    kl_div_layer_mean = KLDivergenceLayer(name="mean_kl_div")

    # Data flow
    mean_mod_mu = mean_mod_mu_layer(model_output)
    mean_mod_sigma = mean_mod_sigma_layer(model_output)
    mean_kl_div = kl_div_layer_mean(
        [mean_inf_mu, mean_inf_sigma, mean_mod_mu, mean_mod_sigma]
    )

    #
    # Mode time course for the functional connectivity
    #

    # Layers
    fc_mod_mu_layer = layers.Dense(config.n_modes, name="fc_mod_mu")
    fc_mod_sigma_layer = layers.Dense(
        config.n_modes, activation="softplus", name="fc_mod_sigma"
    )
    fc_kl_div_layer = KLDivergenceLayer(name="fc_kl_div")

    # Data flow
    fc_mod_mu = fc_mod_mu_layer(model_output)
    fc_mod_sigma = fc_mod_sigma_layer(model_output)
    fc_kl_div = fc_kl_div_layer([fc_inf_mu, fc_inf_sigma, fc_mod_mu, fc_mod_sigma])

    #
    # Total KL loss
    #
    kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")
    kl_loss = kl_loss_layer([mean_kl_div, fc_kl_div])

    return tf.keras.Model(
        inputs=inputs, outputs=[ll_loss, kl_loss, alpha, beta, gamma], name="MRIGO"
    )
