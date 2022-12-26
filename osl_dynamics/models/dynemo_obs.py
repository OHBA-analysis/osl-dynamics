"""Dynamic Network Modes (DyNeMo) observation model.

"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import osl_dynamics.data.tf as dtf
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.inference import regularizers
from osl_dynamics.inference.initializers import WeightInitializer
from osl_dynamics.inference.layers import (
    add_epsilon,
    LogLikelihoodLossLayer,
    VectorsLayer,
    CovarianceMatricesLayer,
    MixVectorsLayer,
    MixMatricesLayer,
)


@dataclass
class Config(BaseModelConfig):
    """Settings for DyNeMo observation model.

    Parameters
    ----------
    model_name : str
        Model name.
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
    covariances_epsilon : float
        Error added to mode covariances for numerical stability.
    means_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for mean vectors.
    covariances_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for covariance matrices.

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

    model_name: str = "DyNeMo-Obs"

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None
    covariances_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

    def __post_init__(self):
        self.validate_observation_model_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")

        if self.covariances_epsilon is None:
            if self.learn_covariances:
                self.covariances_epsilon = 1e-6
            else:
                self.covariances_epsilon = 0.0


class Model(ModelBase):
    """DyNeMo observation model class.

    Parameters
    ----------
    config : osl_dynamics.models.dynemo_obs.Config
    """

    config_type = Config

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
            Mode means.
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

    def set_regularizers(self, training_dataset):
        """Set the means and covariances regularizer based on the training data.

        A multivariate normal prior is applied to the mean vectors with mu = 0,
        sigma=diag((range / 2)**2) and an inverse Wishart prior is applied to the
        covariances matrices with nu=n_channels - 1 + 0.1 and psi=diag(range).

        Parameters
        ----------
        training_dataset : tensorflow.data.Dataset
            Training dataset.
        """
        if self.config.learn_means:
            set_means_regularizer(self.model, training_dataset)

        if self.config.learn_covariances:
            set_covariances_regularizer(self.model, training_dataset)


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
    means_layer = VectorsLayer(
        config.n_modes,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        config.means_regularizer,
        name="means",
    )
    covs_layer = CovarianceMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_covariances,
        config.initial_covariances,
        config.covariances_epsilon,
        config.covariances_regularizer,
        name="covs",
    )
    mix_means_layer = MixVectorsLayer(name="mix_means")
    mix_covs_layer = MixMatricesLayer(name="mix_covs")
    ll_loss_layer = LogLikelihoodLossLayer(config.covariances_epsilon, name="ll_loss")

    # Data flow
    mu = means_layer(data)  # data not used
    D = covs_layer(data)  # data not used
    m = mix_means_layer([alpha, mu])
    C = mix_covs_layer([alpha, D])
    ll_loss = ll_loss_layer([data, m, C])

    return tf.keras.Model(inputs=[data, alpha], outputs=[ll_loss], name="DyNeMo-Obs")


def get_covariances(model):
    covs_layer = model.get_layer("covs")
    covs = add_epsilon(
        covs_layer.bijector(covs_layer.flattened_cholesky_factors),
        covs_layer.epsilon,
        diag=True,
    )
    return covs.numpy()


def get_means_covariances(model):
    means_layer = model.get_layer("means")
    covs_layer = model.get_layer("covs")

    means = means_layer.vectors
    covs = add_epsilon(
        covs_layer.bijector(covs_layer.flattened_cholesky_factors), covs_layer.epsilon
    )
    return means.numpy(), covs.numpy()


def set_means(model, means, update_initializer=True, layer_name="means"):
    means = means.astype(np.float32)
    means_layer = model.get_layer(layer_name)
    layer_weights = means_layer.vectors
    layer_weights.assign(means)

    if update_initializer:
        means_layer.vectors_initializer = WeightInitializer(means)


def set_covariances(model, covariances, update_initializer=True, layer_name="covs"):
    covariances = covariances.astype(np.float32)
    covs_layer = model.get_layer(layer_name)
    layer_weights = covs_layer.flattened_cholesky_factors
    flattened_cholesky_factors = covs_layer.bijector.inverse(covariances)
    layer_weights.assign(flattened_cholesky_factors)

    if update_initializer:
        covs_layer.flattened_cholesky_factors_initializer = WeightInitializer(
            flattened_cholesky_factors
        )


def set_means_regularizer(model, training_dataset, layer_name="means"):
    n_batches = dtf.get_n_batches(training_dataset)
    n_channels = dtf.get_n_channels(training_dataset)
    range_ = dtf.get_range(training_dataset)

    mu = np.zeros(n_channels, dtype=np.float32)
    sigma = np.diag((range_ / 2) ** 2)

    means_layer = model.get_layer(layer_name)
    means_layer.regularizer = regularizers.MultivariateNormal(mu, sigma, n_batches)


def set_covariances_regularizer(model, training_dataset, layer_name="covs"):
    n_batches = dtf.get_n_batches(training_dataset)
    n_channels = dtf.get_n_channels(training_dataset)
    range_ = dtf.get_range(training_dataset)

    nu = n_channels - 1 + 0.1
    psi = np.diag(range_)

    covs_layer = model.get_layer(layer_name)
    covs_layer.regularizer = regularizers.InverseWishart(nu, psi, n_batches)
