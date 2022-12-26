"""Multi-Dynamic Network Modes (M-DyNeMo) observation model.

"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import osl_dynamics.data.tf as dtf
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.models import dynemo_obs
from osl_dynamics.inference import regularizers
from osl_dynamics.inference.initializers import WeightInitializer
from osl_dynamics.inference.layers import (
    add_epsilon,
    LogLikelihoodLossLayer,
    VectorsLayer,
    DiagonalMatricesLayer,
    CorrelationMatricesLayer,
    MixVectorsLayer,
    MixMatricesLayer,
    MatMulLayer,
)


@dataclass
class Config(BaseModelConfig):
    """Settings for M-DyNeMo observation model.

    Parameters
    ----------
    model_name : str
        Model name.
    n_modes : int
        Number of modes for both power.
    n_fc_modes : int
        Number of modes for FC. If none, set to n_modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the generative model.

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

    model_name: str = "M-DyNeMo-Obs"

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
        self.validate_observation_model_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

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
            print("Warning: n_fc_modes is None, set to n_modes.")


class Model(ModelBase):
    """M-DyNeMo observation model class.

    Parameters
    ----------
    config : osl_dynamics.models.mdynemo_obs.Config
    """

    config_type = Config

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
        return get_means_stds_fcs(self.model)

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
            Mode functional connectivities with shape (n_fc_modes, n_channels, n_channels).
        update_initializer: bool
            Do we want to use the passed parameters when we re_initialize
            the model?
        """
        set_means_stds_fcs(self.model, means, stds, fcs, update_initializer)

    def set_regularizers(self, training_dataset):
        """Set the regularizers of means, stds and fcs based on the training data.

        A multivariate normal prior is applied to the mean vectors with mu=0,
        sigma=diag((range / 2)**2), a log normal prior is applied to the standard
        deviations with mu=0, sigma=sqrt(log(2 * (range))) and a marginal inverse
        Wishart prior is applied to the fcs matrices with nu = n_channels - 1 + 0.1.

        Parameters
        ----------
        training_dataset : tensorflow.data.Dataset
            Training dataset.
        """
        if self.config.learn_means:
            dynemo_obs.set_means_regularizer(self.model, training_dataset)

        if self.config.learn_stds:
            set_stds_regularizer(self.model, training_dataset)

        if self.config.learn_fcs:
            set_fcs_regularizer(self.model, training_dataset)


def _model_structure(config):

    # Layers for inputs
    data = layers.Input(shape=(config.sequence_length, config.n_channels), name="data")
    alpha = layers.Input(shape=(config.sequence_length, config.n_modes), name="alpha")
    gamma = layers.Input(
        shape=(config.sequence_length, config.n_fc_modes), name="gamma"
    )

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each mode as the observation model.
    # - We calculate the likelihood of generating the training data with alpha
    #   and the observation model.

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
    mu = means_layer(data)  # data not used
    E = stds_layer(data)  # data not used
    D = fcs_layer(data)  # data not used

    m = mix_means_layer([alpha, mu])
    G = mix_stds_layer([alpha, E])
    F = mix_fcs_layer([gamma, D])
    C = matmul_layer([G, F, G])

    ll_loss = ll_loss_layer([data, m, C])

    return tf.keras.Model(
        inputs=[data, alpha, gamma], outputs=[ll_loss], name="M-DyNeMo-Obs"
    )


def get_means_stds_fcs(model):
    means_layer = model.get_layer("means")
    stds_layer = model.get_layer("stds")
    fcs_layer = model.get_layer("fcs")

    means = means_layer.vectors
    stds = add_epsilon(
        tf.linalg.diag(stds_layer.bijector(stds_layer.diagonals)),
        stds_layer.epsilon,
        diag=True,
    )
    fcs = add_epsilon(
        fcs_layer.bijector(fcs_layer.flattened_cholesky_factors),
        fcs_layer.epsilon,
        diag=True,
    )
    return means.numpy(), stds.numpy(), fcs.numpy()


def set_means_stds_fcs(model, means, stds, fcs, update_initializer=True):
    if stds.ndim == 3:
        # Only keep the diagonal as a vector
        stds = np.diagonal(stds, axis1=1, axis2=2)

    means = means.astype(np.float32)
    stds = stds.astype(np.float32)
    fcs = fcs.astype(np.float32)

    # Get layers
    means_layer = model.get_layer("means")
    stds_layer = model.get_layer("stds")
    fcs_layer = model.get_layer("fcs")

    # Transform the matrices to layer weights
    diagonals = stds_layer.bijector.inverse(stds)
    flattened_cholesky_factors = fcs_layer.bijector.inverse(fcs)

    # Set values
    means_layer.vectors.assign(means)
    stds_layer.diagonals.assign(diagonals)
    fcs_layer.flattened_cholesky_factors.assign(flattened_cholesky_factors)

    # Update initialisers
    if update_initializer:
        means_layer.vectors_initializer = WeightInitializer(means)
        stds_layer.diagonals_initializer = WeightInitializer(diagonals)
        fcs_layer.flattened_cholesky_factors_initializer = WeightInitializer(
            flattened_cholesky_factors
        )


def set_stds_regularizer(model, training_dataset):
    n_batches = dtf.get_n_batches(training_dataset)
    n_channels = dtf.get_n_channels(training_dataset)
    range_ = dtf.get_range(training_dataset)

    mu = np.zeros([n_channels], dtype=np.float32)
    sigma = np.sqrt(np.log(2 * range_))

    stds_layer = model.get_layer("stds")
    stds_layer.regularizer = regularizers.LogNormal(mu, sigma, n_batches)


def set_fcs_regularizer(model, training_dataset):
    n_batches = dtf.get_n_batches(training_dataset)
    n_channels = dtf.get_n_channels(training_dataset)

    nu = n_channels - 1 + 0.1

    fcs_layer = model.get_layer("fcs")
    fcs_layer.regularizer = regularizers.MarginalInverseWishart(
        nu,
        n_channels,
        n_batches,
    )
