"""Single-dynamic Adversarial Generator Encoder (SAGE).

"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, utils
from tqdm.auto import trange

from osl_dynamics.inference.layers import (
    AdversarialLogLikelihoodLossLayer,
    ConcatVectorsMatricesLayer,
    CovarianceMatricesLayer,
    InferenceRNNLayer,
    MixMatricesLayer,
    MixVectorsLayer,
    ModelRNNLayer,
    VectorsLayer,
    StaticLossScalingFactorLayer,
)
from osl_dynamics.models import obs_mod
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase

_logger = logging.getLogger("osl-dynamics")


@dataclass
class Config(BaseModelConfig):
    """Settings for SAGE.

    Parameters
    ----------
    model_name : str
        Model name.
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference, generative and
        discriminator network.
    inference_rnn : str
        RNN to use, either :code:`'gru'` or :code:`'lstm'`.
    inference_n_layers : int
        Number of layers.
    inference_n_units : int
        Number of units.
    inference_normalization : str
        Type of normalization to use. Either :code:`None`, :code:`'batch'`
        or :code:`'layer'`.
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

    discriminator_rnn : str
        RNN to use, either :code:`'gru'` or :code:`'lstm'`.
    discriminator_n_layers : int
        Number of layers.
    discriminator_n_units : int
        Number of units.
    discriminator_normalization : str
        Type of normalization to use. Either :code:`None`, :code:`'batch'`
        or :code:`'layer'`.
    discriminator_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. :code:`'relu'`, :code:`'elu'`, etc.
    discriminator_dropout : float
        Dropout rate.
    discriminator_regularizer : str
        Regularizer.

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
    n_epochs : int
        Number of training epochs.
    optimizer : str or tf.keras.optimizers.Optimizer
        Optimizer to use. :code:`'adam'` is recommended.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    """

    model_name: str = "SAGE"

    # Inference network parameters
    inference_rnn: str = "lstm"
    inference_n_layers: int = 1
    inference_n_units: int = None
    inference_normalization: str = None
    inference_activation: str = "elu"
    inference_dropout: float = 0.0
    inference_regularizer: str = None

    # Model network parameters
    model_rnn: str = "lstm"
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: str = None
    model_activation: str = "elu"
    model_dropout: float = 0.0
    model_regularizer: str = None

    # Descriminator network parameters
    discriminator_rnn: str = "lstm"
    discriminator_n_layers: int = 1
    discriminator_n_units: int = None
    discriminator_normalization: str = None
    discriminator_activation: str = "elu"
    discriminator_dropout: float = 0.0
    discriminator_regularizer: str = None

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None
    covariances_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

    def __post_init__(self):
        self.validate_dimension_parameters()
        self.validate_training_parameters()
        self.validate_observation_model_parameters()

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")

        if self.covariances_epsilon is None:
            if self.learn_covariances:
                self.covariances_epsilon = 1e-6
            else:
                self.covariances_epsilon = 0.0


class Model(ModelBase):
    """SAGE model class.

    Parameters
    ----------
    config : osl_dynamics.models.sage.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model for the inference, generator and discriminator
        model and the full SAGE model.
        """
        _logger.info("Build models")
        self.inference_model = _build_inference_model(self.config)
        self.inference_model.summary()
        print()
        self.generator_model = _build_generator_model(self.config)
        self.generator_model.summary()
        print()
        self.discriminator_model = _build_discriminator_model(self.config)
        self.discriminator_model.summary()
        print()

        data = layers.Input(
            shape=(self.config.sequence_length, self.config.n_channels),
            name="data",
        )
        C_m, alpha_posterior = self.inference_model(data)
        alpha_prior = self.generator_model(alpha_posterior)
        discriminator_output_prior = self.discriminator_model(alpha_prior)
        self.model = models.Model(
            data,
            [C_m, discriminator_output_prior],
            name="SAGE",
        )
        self.model.summary()
        print()

    def compile(self):
        """Compile the model."""

        self.discriminator_model.compile(
            loss="binary_crossentropy",
            optimizer=self.config.optimizer.lower(),
            metrics=["accuracy"],
        )
        self.discriminator_model.trainable = False

        # Reconstruction (Likelihood) loss:
        # The first loss corresponds to the likelihood - this tells us how well
        # we are explaining our data according to the current estimate of the
        # generative model, and is given by:
        # L = \sum_{t=1}^{T} log p(Y_t | \theta_t^m = \mu^{m,\theta}_t,
        #                                \theta_t^c = \mu^{c,\theta}_t)
        ll_loss = AdversarialLogLikelihoodLossLayer(
            self.config.n_channels, name="ll_loss"
        )

        # Regularization (Prior) Loss:
        # The second loss regularises the estimate of the latent, time-varying
        # parameters [$\theta^m$, $\theta^c$] using an adaptive prior - this
        # penalises when the posterior estimates of [$\theta^m$, $\theta^c$]
        # deviate from the prior:
        # R = \sum_{t=1}^{T} [
        #     CrossEntropy(\mu^{m,\theta}_t|| \hat{\mu}^{m,\theta}_{t})
        #     + CrossEntropy(\mu^{c,\theta}_t || \hat{\mu}^{c,\theta}_{t})
        # ]

        # Compile the full model
        optimizer = optimizers.get(
            {
                "class_name": self.config.optimizer.lower(),
                "config": {"learning_rate": self.config.learning_rate},
            }
        )
        self.model.compile(
            loss=[ll_loss, "binary_crossentropy"],
            loss_weights=[0.995, 0.005],
            optimizer=optimizer,
        )

    def fit(self, training_data, epochs=None, verbose=1):
        """Train the model.

        Parameters
        ----------
        training_data : tf.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        epochs : int, optional
            Number of epochs to train. Defaults to value in :code:`config`
            if not passed.
        verbose : int, optional
            Should we print a progress bar?

        Returns
        -------
        history : history
            History of :code:`discriminator_loss` and :code:`generator_loss`.
        """
        if epochs is None:
            epochs = self.config.n_epochs

        # Make sure training data is a TensorFlow Dataset
        training_data = self.make_dataset(
            training_data,
            shuffle=True,
            concatenate=True,
        )

        # Path to save the best model weights
        timestr = time.strftime("%Y%m%d-%H%M%S")  # current date-time
        filepath = "tmp/"
        save_filepath = filepath + str(timestr) + "_best_model.h5"

        history = []
        best_val_loss = np.Inf
        for epoch in range(epochs):
            if verbose:
                print("Epoch {}/{}".format(epoch + 1, epochs))
                pb_i = utils.Progbar(len(training_data), stateful_metrics=["D", "G"])

            for idx, batch in enumerate(training_data):
                # Generate real/fake input for the discriminator
                real = np.ones((len(batch["data"]), self.config.sequence_length, 1))
                fake = np.zeros((len(batch["data"]), self.config.sequence_length, 1))
                train_discriminator = self._train_discriminator(real, fake)

                C_m, alpha_posterior = self.inference_model.predict_on_batch(batch)
                alpha_prior = self.generator_model.predict_on_batch(alpha_posterior)

                # Train discriminator, inference and generator model
                discriminator_loss = train_discriminator(
                    alpha_posterior,
                    alpha_prior,
                )
                generator_loss = self.model.train_on_batch(batch, [batch, real])

                if verbose:
                    pb_i.add(
                        1,
                        values=[("D", discriminator_loss[1]), ("G", generator_loss[0])],
                    )

            if generator_loss[0] < best_val_loss:
                self.save_weights(save_filepath)
                _logger.info(
                    "Best model w/ val loss (generator) {} saved to {}".format(
                        generator_loss[0], save_filepath
                    )
                )
                best_val_loss = generator_loss[0]
            val_loss = self.model.test_on_batch([batch], [batch, real])

            history.append({"D": discriminator_loss[1], "G": generator_loss[0]})

        # Load the weights for the best model
        _logger.info(f"Loading best model: {save_filepath}")
        self.load_weights(save_filepath)

        return history

    def _train_discriminator(self, real, fake):
        def train(real_samples, fake_samples):
            self.discriminator_model.trainable = True
            loss_real = self.discriminator_model.train_on_batch(real_samples, real)
            loss_fake = self.discriminator_model.train_on_batch(fake_samples, fake)
            loss = np.add(loss_real, loss_fake) * 0.5
            self.discriminator_model.trainable = False
            return loss

        return train

    def get_alpha(self, inputs, concatenate=False):
        """Mode mixing factors, alpha.

        Parameters
        ----------
        inputs : tf.data.Dataset or osl_dynamics.data.Data
            Prediction data.
        concatenate : bool, optional
            Should we concatenate the output?

        Returns
        -------
        alpha : list or np.ndarray
            Mode mixing factors with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        """
        inputs = self.make_dataset(inputs, concatenate=concatenate)

        _logger.info("Getting alpha:")
        outputs = []
        for dataset in inputs:
            alpha = self.inference_model.predict(dataset)[1]
            alpha = np.concatenate(alpha)
            outputs.append(alpha)

        if concatenate or len(outputs) == 1:
            outputs = np.concatenate(outputs)

        return outputs

    def get_means(self):
        """Get the mode means.

        Returns
        -------
        means : np.ndarray
            Mode means. Shape (n_modes, n_channels).
        """
        return obs_mod.get_observation_model_parameter(
            self.inference_model,
            "means",
        )

    def get_covariances(self):
        """Get the mode covariances.

        Returns
        -------
        covariances : np.ndarary
            Mode covariances. Shape (n_modes, n_channels, n_channels).
        """
        return obs_mod.get_observation_model_parameter(
            self.inference_model,
            "covs",
        )

    def get_means_covariances(self):
        """Get the mode means and covariances.

        This is a wrapper for :code:`get_means` and :code:`get_covariances`.

        Returns
        -------
        means : np.ndarary
            Mode means. Shape (n_modes, n_channels).
        covariances : np.ndarray
            Mode covariances. Shape (n_modes, n_channels, n_channels).
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
        update_initializer : bool, optional
            Do we want to use the passed means when we re-initialize the model?
        """
        obs_mod.set_observation_model_parameter(
            self.inference_model,
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
            self.inference_model,
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

    def sample_alpha(self, alpha=None):
        """Uses the generator to predict the prior, :code:`alpha`.

        Parameters
        ----------
        alpha : np.ndarray, optional
            Shape must be (n_samples, n_modes).

        Returns
        -------
        alpha : np.ndarray
            Predicted :code:`alpha`.
        """
        n_samples = np.shape(alpha)[0]
        alpha_sampled = np.empty(
            [n_samples, self.config.n_modes],
            dtype=np.float32,
        )

        for i in trange(
            n_samples - self.config.sequence_length,
            desc="Predicting mode time course",
        ):
            # Extract the sequence
            alpha_input = alpha[i : i + self.config.sequence_length]
            alpha_input = alpha_input[np.newaxis, :, :]

            # Predict the point estimates for theta one time step in the future
            alpha_sampled[i] = self.generator_model.predict_on_batch(alpha_input)[0, 0]

        return alpha_sampled


def _build_inference_model(config):
    # Inference RNN:
    #   alpha_t = zeta(theta^m_t) where
    #   mu^{m,theta}_t = f(BLSTM(Y,omega^m_e),lambda_e^m)
    #   mu^{c,theta}_t = f(BLSTM(Y,omega^c_e),lambda_e^c)

    # Definition of layers
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )

    # Static loss scaling factor
    static_loss_scaling_factor_layer = StaticLossScalingFactorLayer(
        name="static_loss_scaling_factor"
    )
    static_loss_scaling_factor = static_loss_scaling_factor_layer(inputs)

    data_drop_layer = layers.TimeDistributed(
        layers.Dropout(config.inference_dropout, name="inf_data_drop")
    )
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

    alpha_layer = layers.TimeDistributed(
        layers.Dense(config.n_modes, activation="softmax"), name="inf_alpha"
    )

    # Data flow
    data_drop = data_drop_layer(inputs)
    theta = inf_rnn_layer(data_drop)
    alpha = alpha_layer(theta)

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix
    #   for each mode as the observation model.
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
    concat_means_covs_layer = ConcatVectorsMatricesLayer(name="concat_means_covs")

    # Data flow
    mu = means_layer(
        inputs, static_loss_scaling_factor=static_loss_scaling_factor
    )  # inputs not used
    D = covs_layer(
        inputs, static_loss_scaling_factor=static_loss_scaling_factor
    )  # inputs not used
    m = mix_means_layer([alpha, mu])
    C = mix_covs_layer([alpha, D])
    C_m = concat_means_covs_layer([m, C])

    return models.Model(inputs, [C_m, alpha], name="inference")


def _build_generator_model(config):
    # Model RNN:
    #   alpha_{t} = zeta(theta^m_t}) where
    #   hat{mu}^{m,theta}_t = f(LSTM(theta^m_<t,omega^m_g), lambda_g^m)
    #   hat{mu}^{c,theta}_t = f(LSTM(theta^c_<t,omega^c_g), lambda_g^c)

    # Definition of layers
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_modes),
        name="gen_inp",
    )
    drop_layer = layers.TimeDistributed(
        layers.Dropout(config.model_dropout, name="gen_data_drop")
    )
    mod_rnn_layer = ModelRNNLayer(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout,
        config.model_regularizer,
        name="gen_rnn",
    )
    prior_layer = layers.TimeDistributed(
        layers.Dense(config.n_modes, activation="softmax"),
        name="gen_softmax_alpha",
    )

    # Data flow
    theta_drop = drop_layer(inputs)
    mod_rnn = mod_rnn_layer(theta_drop)
    prior = prior_layer(mod_rnn)

    return models.Model(inputs, prior, name="generator")


def _build_discriminator_model(config):
    # Descriminator RNN:
    #   D_theta^m_t = sigma(f(BLSTM([zeta(hat{mu}^{m,theta}_t),
    #                                zeta(mu^{m,theta}_t)],omega^m_d),
    #                         \lambda_d^m))
    #   D_theta^c_t = sigma(f(BLSTM([zeta(hat{mu}^{c,theta}_t),
    #                                zeta(mu^{c,theta}_t)],omega^c_d),
    #                        \lambda_d^c))

    # Definition of layers
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_modes),
        name="data",
    )
    drop_layer = layers.TimeDistributed(
        layers.Dropout(config.model_dropout, name="dis_data_drop")
    )
    dis_rnn_layer = ModelRNNLayer(
        config.discriminator_rnn,
        config.discriminator_normalization,
        config.discriminator_activation,
        config.discriminator_n_layers,
        config.discriminator_n_units,
        config.discriminator_dropout,
        config.discriminator_regularizer,
        name="dis_rnn",
    )
    sigmoid_layer = layers.TimeDistributed(
        layers.Dense(1, activation="sigmoid"), name="dis_sigmoid"
    )

    # Data flow
    theta_norm_drop = drop_layer(inputs)
    dis_rnn = dis_rnn_layer(theta_norm_drop)
    output = sigmoid_layer(dis_rnn)

    return models.Model(inputs, output, name="discriminator")
