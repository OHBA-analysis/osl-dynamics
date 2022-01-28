"""Model class for a multi-time-scale generative model with Gaussian observations.

"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from dynemo.models.mod_base import BaseConfig, ModelBase
from dynemo.models.inf_mod_base import InferenceModelConfig, InferenceModelBase
from dynemo.models.layers import (
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
    FillConstantLayer,
    ConcatenateLayer,
    MatMulLayer,
    DummyLayer,
)


@dataclass
class Config(BaseConfig, InferenceModelConfig):
    """Settings for MRIGO.

    Dimension Parameters
    --------------------
    n_modes : int
        Number of modes.
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

    Alpha, Beta, Gamma Parameters
    -----------------------------
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

    Multi-time-scale and Observation Model Parameters
    -------------------------------------------------
    separate_rnns : bool
        Should we have a separate RNN for each time scale?
    fix_std: bool
        Should we have constant std across modes and time?
    tie_mean_std: bool
        Should we tie up the time courses of mean and std?
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

    KL Annealing Parameters
    -----------------------
    do_kl_annealing : bool
        Should we use KL annealing during training?
    kl_annealing_curve : str
        Type of KL annealing curve. Either 'linear' or 'tanh'.
    kl_annealing_sharpness : float
        Parameter to control the shape of the annealing curve if
        kl_annealing_curve='tanh'.
    n_kl_annealing_epochs : int
        Number of epochs to perform KL annealing.

    Initialization Parameters
    -------------------------
    n_init : int
        Number of initializations.
    n_init_epochs : int
        Number of epochs to train each initialization.

    Training Parameters
    -------------------
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
    separate_rnns: bool = True
    fix_std: bool = False
    tie_mean_std: bool = False
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
        self.validate_initialization_parameters()
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
        means_layer = self.model.get_layer("means")
        stds_layer = self.model.get_layer("stds")
        fcs_layer = self.model.get_layer("fcs")
        return means_layer(1).numpy(), stds_layer(1).numpy(), fcs_layer(1).numpy()

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
        if stds.ndim == 3:
            # Only keep the diagonal as a vector
            stds = np.diagonal(stds, axis1=1, axis2=2)

        means = means.astype(np.float32)
        stds = stds.astype(np.float32)
        fcs = fcs.astype(np.float32)

        # Get layers
        means_layer = self.model.get_layer("means")
        stds_layer = self.model.get_layer("stds")
        fcs_layer = self.model.get_layer("fcs")

        # Transform the matrices to layer weights
        diagonals = stds_layer.bijector.inverse(stds)
        flattened_cholesky_factors = fcs_layer.bijector.inverse(fcs)

        # Set values
        means_layer.vectors.assign(means)
        stds_layer.diagonals.assign(diagonals)
        fcs_layer.flattened_cholesky_factors.assign(flattened_cholesky_factors)

        # Update initialisers
        if update_initializer:
            means_layer.initial_value = means
            stds_layer.initial_value = stds
            fcs_layer.initial_value = fcs

            stds_layer.initial_diagonals = diagonals
            fcs_layer.initial_flattened_cholesky_factors = flattened_cholesky_factors

            means_layer.vectors_initializer.initial_value = means
            stds_layer.diagonals_initializer.initial_value = diagonals
            fcs_layer.flattened_cholesky_factors_initializer.initial_value = (
                flattened_cholesky_factors
            )


def _model_structure(config):

    # Number of time courses to learn
    if config.fix_std:
        # Learn alpha and gamma
        n_tcs = 2
    else:
        # Learn alpha, beta and gamma
        n_tcs = 3

    # Number of RNNs
    if config.separate_rnns:
        n_rnns = n_tcs
    else:
        n_rnns = 1

    # Layer for input
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )

    #
    # Inference RNN
    #

    # Layers
    inference_input_dropout_layers = [
        layers.Dropout(config.inference_dropout_rate, name=f"data_drop_{i}")
        for i in range(n_rnns)
    ]
    inference_output_layers = [
        InferenceRNNLayer(
            config.inference_rnn,
            config.inference_normalization,
            config.inference_activation,
            config.inference_n_layers,
            config.inference_n_units,
            config.inference_dropout_rate,
            name=f"inf_rnn_{i}",
        )
        for i in range(n_rnns)
    ]

    # Data flow
    inference_input_dropout = [
        inference_input_dropout_layers[i](inputs) for i in range(n_rnns)
    ]
    inference_output = [
        inference_output_layers[i](inference_input_dropout[i]) for i in range(n_rnns)
    ]
    if n_tcs == 2:
        if n_rnns == 2:
            mean_inference_output = inference_output[0]
            fc_inference_output = inference_output[1]
        else:
            mean_inference_output = inference_output[0]
            std_inference_output = inference_output[0]
            fc_inference_output = inference_output[0]
    elif n_tcs == 3:
        if n_rnns == 3:
            mean_inference_output = inference_output[0]
            std_inference_output = inference_output[1]
            fc_inference_output = inference_output[2]
        else:
            mean_inference_output = inference_output[0]
            std_inference_output = inference_output[0]
            fc_inference_output = inference_output[0]

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
    mean_inf_mu = mean_inf_mu_layer(mean_inference_output)
    mean_inf_sigma = mean_inf_sigma_layer(mean_inference_output)
    mean_theta = mean_theta_layer([mean_inf_mu, mean_inf_sigma])
    mean_theta_norm = mean_theta_norm_layer(mean_theta)
    alpha = alpha_layer(mean_theta_norm)

    #
    # Mode time course for the standard deviation
    #

    # Layers
    if config.fix_std:
        beta_layer = FillConstantLayer(1 / config.n_modes, name="beta")
    elif config.tie_mean_std:
        beta_layer = DummyLayer(name="beta")
    else:
        std_inf_mu_layer = layers.Dense(config.n_modes, name="std_inf_mu")
        std_inf_sigma_layer = layers.Dense(
            config.n_modes, activation="softplus", name="std_inf_sigma"
        )
        std_theta_layer = SampleNormalDistributionLayer(name="std_theta")
        std_theta_norm_layer = NormalizationLayer(
            config.theta_normalization, name="std_theta_norm"
        )
        beta_layer = ThetaActivationLayer(
            config.alpha_xform,
            config.initial_alpha_temperature,
            config.learn_alpha_temperature,
            name="beta",
        )

    # Data flow
    if config.fix_std or config.tie_mean_std:
        beta = beta_layer(alpha)
    else:
        std_inf_mu = std_inf_mu_layer(std_inference_output)
        std_inf_sigma = std_inf_sigma_layer(std_inference_output)
        std_theta = std_theta_layer([std_inf_mu, std_inf_sigma])
        std_theta_norm = std_theta_norm_layer(std_theta)
        beta = beta_layer(std_theta_norm)

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
    fc_inf_mu = fc_inf_mu_layer(fc_inference_output)
    fc_inf_sigma = fc_inf_sigma_layer(fc_inference_output)
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

    # Keep sampled theta separate or concatenate depending on the number of RNNs
    if n_tcs == 2:
        theta_norm = [mean_theta_norm, fc_theta_norm]
    else:
        theta_norm = [mean_theta_norm, std_theta_norm, fc_theta_norm]

    if n_rnns == 1:
        concatenate_layer = ConcatenateLayer(axis=2, name="theta_norm")
        theta_norm = [concatenate_layer(theta_norm)]

    # Layers
    model_input_dropout_layers = [
        layers.Dropout(config.model_dropout_rate, name=f"theta_norm_drop_{i}")
        for i in range(n_rnns)
    ]
    model_output_layers = [
        ModelRNNLayer(
            config.model_rnn,
            config.model_normalization,
            config.model_activation,
            config.model_n_layers,
            config.model_n_units,
            config.model_dropout_rate,
            name=f"mod_rnn_{i}",
        )
        for i in range(n_rnns)
    ]

    # Data flow
    model_input_dropout = [
        model_input_dropout_layers[i](theta_norm[i]) for i in range(n_rnns)
    ]
    model_output = [
        model_output_layers[i](model_input_dropout[i]) for i in range(n_rnns)
    ]

    if n_rnns == 1:
        if n_tcs == 2:
            mean_model_output = model_output[0]
            fc_model_output = model_output[0]
        else:
            mean_model_output = model_output[0]
            std_model_output = model_output[0]
            fc_model_output = model_output[0]
    else:
        if n_tcs == 2:
            mean_model_output = model_output[0]
            fc_model_output = model_output[1]
        else:
            mean_model_output = model_output[0]
            std_model_output = model_output[1]
            fc_model_output = model_output[2]

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
    mean_mod_mu = mean_mod_mu_layer(mean_model_output)
    mean_mod_sigma = mean_mod_sigma_layer(mean_model_output)
    mean_kl_div = kl_div_layer_mean(
        [mean_inf_mu, mean_inf_sigma, mean_mod_mu, mean_mod_sigma]
    )

    #
    # Mode time course for the standard deviation
    #

    if not (config.fix_std or config.tie_mean_std):
        # Layers
        std_mod_mu_layer = layers.Dense(config.n_modes, name="std_mod_mu")
        std_mod_sigma_layer = layers.Dense(
            config.n_modes, activation="softplus", name="std_mod_sigma"
        )
        std_kl_div_layer = KLDivergenceLayer(name="std_kl_div")

        # Data flow
        std_mod_mu = std_mod_mu_layer(std_model_output)
        std_mod_sigma = std_mod_sigma_layer(std_model_output)
        std_kl_div = std_kl_div_layer(
            [std_inf_mu, std_inf_sigma, std_mod_mu, std_mod_sigma]
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
    fc_mod_mu = fc_mod_mu_layer(fc_model_output)
    fc_mod_sigma = fc_mod_sigma_layer(fc_model_output)
    fc_kl_div = fc_kl_div_layer([fc_inf_mu, fc_inf_sigma, fc_mod_mu, fc_mod_sigma])

    #
    # Total KL loss
    #
    kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")
    if config.fix_std or config.tie_mean_std:
        kl_loss = kl_loss_layer([mean_kl_div, fc_kl_div])
    else:
        kl_loss = kl_loss_layer([mean_kl_div, std_kl_div, fc_kl_div])

    return tf.keras.Model(
        inputs=inputs, outputs=[ll_loss, kl_loss, alpha, beta, gamma], name="MRIGO"
    )
