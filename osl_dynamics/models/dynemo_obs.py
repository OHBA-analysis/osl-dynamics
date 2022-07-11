"""Dynamic Network Modes (DyNeMo) observation model.

"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.inference.layers import (
    LogLikelihoodLossLayer,
    MeanVectorsLayer,
    CovarianceMatricesLayer,
    MixVectorsLayer,
    MixMatricesLayer,
)


@dataclass
class Config(BaseModelConfig):
    """Settings for DyNeMo observation model.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the generative model.

    learn_means : bool
        Should we make the mean vectors for each mode trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each mode trainable?
    initial_means : np.ndarray
        Initialisation for mean vectors.
    initial_covariances : np.ndarray
        Initialisation for mode covariances.

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

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None

    def __post_init__(self):
        self.validate_observation_model_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")


class Model(ModelBase):
    """DyNeMo observation model class.

    Parameters
    ----------
    config : osl_dynamics.models.dynemo_obs.Config
    """

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def get_covariances(self):
        """Get the covariances of each mode.

        Returns
        -------
        covariances : np.ndarary
            Mode covariances.
        """
        return get_covariances(self.model)

    def get_means_covariances(self):
        """Get the means and covariances of each mode.

        Returns
        -------
        means : np.ndarary
            Mode means.
        covariances : np.ndarray
            Mode covariances.
        """
        return get_means_covariances(self.model)

    def set_means(self, means, update_initializer=True):
        """Set the means of each mode.

        Parameters
        ----------
        means : np.ndarray
            Mode covariances.
        update_initializer : bool
            Do we want to use the passed means when we re-initialize
            the model?
        """
        set_means(self.model, means, update_initializer)

    def set_covariances(self, covariances, update_initializer=True):
        """Set the covariances of each mode.

        Parameters
        ----------
        covariances : np.ndarray
            Mode covariances.
        update_initializer : bool
            Do we want to use the passed covariances when we re-initialize
            the model?
        """
        set_covariances(self.model, covariances, update_initializer)


def _model_structure(config):

    # Layers for inputs
    data = layers.Input(shape=(config.sequence_length, config.n_channels), name="data")
    alpha = layers.Input(shape=(config.sequence_length, config.n_modes), name="alpha")

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each mode as the observation model.
    # - We calculate the likelihood of generating the training data with alpha
    #   and the observation model.

    # Definition of layers
    means_layer = MeanVectorsLayer(
        config.n_modes,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        name="means",
    )
    covs_layer = CovarianceMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_covariances,
        config.initial_covariances,
        name="covs",
    )
    mix_means_layer = MixVectorsLayer(name="mix_means")
    mix_covs_layer = MixMatricesLayer(name="mix_covs")
    ll_loss_layer = LogLikelihoodLossLayer(name="ll_loss")

    # Data flow
    mu = means_layer(data)  # data not used
    D = covs_layer(data)  # data not used
    m = mix_means_layer([alpha, mu])
    C = mix_covs_layer([alpha, D])
    ll_loss = ll_loss_layer([data, m, C])

    return tf.keras.Model(inputs=[data, alpha], outputs=[ll_loss], name="DyNeMo-Obs")


def get_covariances(model):
    covs_layer = model.get_layer("covs")
    return covs_layer(1).numpy()


def get_means_covariances(model):
    means_layer = model.get_layer("means")
    covs_layer = model.get_layer("covs")
    return means_layer(1).numpy(), covs_layer(1).numpy()


def set_means(model, means, update_initializer=True):
    means = means.astype(np.float32)
    means_layer = model.get_layer("means")
    layer_weights = means_layer.means
    layer_weights.assign(means)

    if update_initializer:
        means_layer.initial_value = means
        means_layer.vectors_initializer.initial_value = means


def set_covariances(model, covariances, update_initializer=True):
    covariances = covariances.astype(np.float32)
    covs_layer = model.get_layer("covs")
    layer_weights = covs_layer.flattened_cholesky_factors
    flattened_cholesky_factors = covs_layer.bijector.inverse(covariances)
    layer_weights.assign(flattened_cholesky_factors)

    if update_initializer:
        covs_layer.initial_value = covariances
        covs_layer.initial_flattened_cholesky_factors = flattened_cholesky_factors
        covs_layer.flattened_cholesky_factors_initializer.initial_value = (
            flattened_cholesky_factors
        )
