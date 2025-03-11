"""Fully TF HMM.

"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference.layers import (
    CovarianceMatricesLayer,
    DiagonalMatricesLayer,
    VectorsLayer,
    StaticLossScalingFactorLayer,
    HiddenMarkovStateInferenceLayer,
    SeparateLogLikelihoodLayer,
    SumLogLikelihoodLossLayer,
)
from osl_dynamics.models import obs_mod
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.models.inf_mod_base import (
    MarkovStateInferenceModelConfig,
    MarkovStateInferenceModelBase,
)


@dataclass
class Config(BaseModelConfig, MarkovStateInferenceModelConfig):

    model_name: str = "HMM"

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
        self.validate_observation_model_parameters()
        self.validate_trans_prob_parameters()
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


class Model(MarkovStateInferenceModelBase):

    config_type = Config

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(self.config)

    def set_static_loss_scaling_factor(self, dataset):
        """Set the :code:`n_batches` attribute of the
        :code:`"static_loss_scaling_factor"` layer.

        Parameters
        ----------
        dataset : tf.data.Dataset
            TensorFlow dataset.
        """
        layer_names = [layer.name for layer in self.model.layers]
        if "static_loss_scaling_factor" in layer_names:
            n_batches = dtf.get_n_batches(dataset)
            self.model.get_layer("static_loss_scaling_factor").n_batches = n_batches

    def get_means(self):
        """Get the state means.

        Returns
        -------
        means : np.ndarray
            State means. Shape is (n_states, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "means")

    def get_covariances(self):
        """Get the state covariances.

        Returns
        -------
        covariances : np.ndarray
            State covariances. Shape is (n_states, n_channels, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "covs")

    def get_means_covariances(self):
        """Get the state means and covariances.

        This is a wrapper for :code:`get_means` and :code:`get_covariances`.

        Returns
        -------
        means : np.ndarary
            State means.
        covariances : np.ndarray
            State covariances.
        """
        return self.get_means(), self.get_covariances()

    def get_observation_model_parameters(self):
        """Wrapper for :code:`get_means_covariances`."""
        return self.get_means_covariances()

    def set_means(self, means, update_initializer=True):
        """Set the state means.

        Parameters
        ----------
        means : np.ndarray
            State means. Shape is (n_states, n_channels).
        update_initializer : bool, optional
            Do we want to use the passed means when we re-initialize the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            means,
            layer_name="means",
            update_initializer=update_initializer,
        )

    def set_covariances(self, covariances, update_initializer=True):
        """Set the state covariances.

        Parameters
        ----------
        covariances : np.ndarray
            State covariances. Shape is (n_states, n_channels, n_channels).
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
        self.set_means(means, update_initializer=update_initializer)
        self.set_covariances(covariances, update_initializer=update_initializer)

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
        :code:`config.diagonal_covariances=True`, a log normal prior is applied
        to the diagonal of the covariances matrices with :code:`mu=0`,
        :code:`sigma=sqrt(log(2*range))`, otherwise an inverse Wishart prior is
        applied to the covariances matrices with :code:`nu=n_channels-1+0.1`
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


def _model_structure(config):
    # Inputs
    data = layers.Input(
        shape=(config.sequence_length, config.n_channels),
        dtype=tf.float32,
        name="data",
    )

    # Static loss scaling factor
    static_loss_scaling_factor_layer = StaticLossScalingFactorLayer(
        config.sequence_length,
        config.loss_calc,
        name="static_loss_scaling_factor",
    )
    static_loss_scaling_factor = static_loss_scaling_factor_layer(data)

    # Observation model
    means_layer = VectorsLayer(
        config.n_states,
        config.n_channels,
        config.learn_means,
        config.initial_means,
        config.means_regularizer,
        name="means",
    )
    if config.diagonal_covariances:
        covs_layer = DiagonalMatricesLayer(
            config.n_states,
            config.n_channels,
            config.learn_covariances,
            config.initial_covariances,
            config.covariances_epsilon,
            config.covariances_regularizer,
            name="covs",
        )
    else:
        covs_layer = CovarianceMatricesLayer(
            config.n_states,
            config.n_channels,
            config.learn_covariances,
            config.initial_covariances,
            config.covariances_epsilon,
            config.covariances_regularizer,
            name="covs",
        )
    mu = means_layer(
        data, static_loss_scaling_factor=static_loss_scaling_factor
    )  # data not used
    D = covs_layer(
        data, static_loss_scaling_factor=static_loss_scaling_factor
    )  # data not used

    # Log-likelihood
    ll_layer = SeparateLogLikelihoodLayer(
        config.n_states,
        dtype="float64",
        name="ll",
    )
    ll = ll_layer([data, [mu], [D]])

    # Hidden state inference
    hidden_state_inference_layer = HiddenMarkovStateInferenceLayer(
        config.n_states,
        config.initial_trans_prob,
        config.initial_state_probs,
        config.learn_trans_prob,
        config.learn_initial_state_probs,
        dtype="float64",
        name="hid_state_inf",
    )
    gamma, xi = hidden_state_inference_layer(ll)

    # Loss
    ll_loss_layer = SumLogLikelihoodLossLayer(config.loss_calc, name="ll_loss")
    ll_loss = ll_loss_layer([ll, gamma])

    return tf.keras.Model(inputs=data, outputs=[ll_loss, gamma, xi], name="HMM")
