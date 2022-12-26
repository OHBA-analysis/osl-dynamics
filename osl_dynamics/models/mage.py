"""Multi-dynamic Adversarial Generator Encoder (MAGE).

"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, utils
from tqdm import trange

from osl_dynamics.models import mdynemo_obs
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.inference.layers import (
    InferenceRNNLayer,
    VectorsLayer,
    DiagonalMatricesLayer,
    CorrelationMatricesLayer,
    MixVectorsLayer,
    MixMatricesLayer,
    MatMulLayer,
    ModelRNNLayer,
    ConcatVectorsMatricesLayer,
    AdversarialLogLikelihoodLossLayer,
)


@dataclass
class Config(BaseModelConfig):
    """Settings for MAGE.

    Parameters
    ----------
    model_name : str
        Model name.
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference, generative and discriminator network.

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

    discriminator_rnn : str
        RNN to use, either 'gru' or 'lstm'.
    discriminator_n_layers : int
        Number of layers.
    discriminator_n_units : int
        Number of units.
    discriminator_normalization : str
        Type of normalization to use. Either None, 'batch' or 'layer'.
    discriminator_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. 'relu', 'elu', etc.
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
    stds_epsilon : float
        Error added to mode stds for numerical stability.
    fcs_epsilon : float
        Error added to mode fcs for numerical stability.

    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
    n_epochs : int
        Number of training epochs.
    optimizer : str or tensorflow.keras.optimizers.Optimizer
        Optimizer to use. 'adam' is recommended.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    """

    model_name: str = "MAGE"

    # Inference network parameters
    inference_rnn: Literal["gru", "lstm"] = "lstm"
    inference_n_layers: int = 1
    inference_n_units: int = None
    inference_normalization: Literal[None, "batch", "layer"] = None
    inference_activation: str = "elu"
    inference_dropout: float = 0.0
    inference_regularizer: str = None

    # Model network parameters
    model_rnn: Literal["gru", "lstm"] = "lstm"
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: Literal[None, "batch", "layer"] = None
    model_activation: str = "elu"
    model_dropout: float = 0.0
    model_regularizer: str = None

    # Descriminator network parameters
    discriminator_rnn: Literal["gru", "lstm"] = "lstm"
    discriminator_n_layers: int = 1
    discriminator_n_units: int = None
    discriminator_normalization: Literal[None, "batch", "layer"] = None
    discriminator_activation: str = "elu"
    discriminator_dropout: float = 0.0
    discriminator_regularizer: str = None

    # Observation model parameters
    learn_means: bool = None
    learn_stds: bool = None
    learn_fcs: bool = None
    initial_means: np.ndarray = None
    initial_stds: np.ndarray = None
    initial_fcs: np.ndarray = None
    stds_epsilon: float = None
    fcs_epsilon: float = None
    multiple_dynamics: bool = True

    def __post_init__(self):
        self.validate_dimension_parameters()
        self.validate_training_parameters()
        self.validate_rnn_parameters()
        self.validate_observation_model_parameters()

    def validate_rnn_parameters(self):
        if self.inference_n_units is None:
            raise ValueError("Please pass inference_n_units.")

        if self.model_n_units is None:
            raise ValueError("Please pass model_n_units.")

        if self.discriminator_n_units is None:
            raise ValueError("Please pass discriminator_n_units.")

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


class Model(ModelBase):
    """MAGE model class.

    Parameters
    ----------
    config : osl_dynamics.models.mage.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model for the inference, generator and discriminator model
        and the full MAGE model.
        """
        print("Build models")
        self.inference_model = _build_inference_model(self.config)
        self.inference_model.summary()
        print()

        self.generator_model_mean = _build_generator_model(self.config, name="alpha")
        self.generator_model_cov = _build_generator_model(self.config, name="gamma")
        self.generator_model_cov.summary()
        self.generator_model_mean.summary()
        print()

        self.discriminator_model_mean = _build_discriminator_model(
            self.config, name="alpha"
        )
        self.discriminator_model_cov = _build_discriminator_model(
            self.config, name="gamma"
        )
        self.discriminator_model_mean.summary()
        self.discriminator_model_cov.summary()
        print()

        # Connect the inputs and outputs
        data = layers.Input(
            shape=(self.config.sequence_length, self.config.n_channels), name="data"
        )
        C_m, alpha_posterior, gamma_posterior = self.inference_model(data)
        alpha_prior = self.generator_model_mean(alpha_posterior)
        gamma_prior = self.generator_model_cov(gamma_posterior)
        discriminator_output_alpha = self.discriminator_model_mean(alpha_prior)
        discriminator_output_gamma = self.discriminator_model_cov(gamma_prior)

        self.model = models.Model(
            data,
            [C_m, discriminator_output_alpha, discriminator_output_gamma],
            name="MAGE",
        )
        self.model.summary()
        print()

    def compile(self):
        """Compile the model."""

        self.discriminator_model_mean.compile(
            loss="binary_crossentropy",
            optimizer=self.config.optimizer.lower(),
            metrics=["accuracy"],
        )
        self.discriminator_model_mean.trainable = False

        self.discriminator_model_cov.compile(
            loss="binary_crossentropy",
            optimizer=self.config.optimizer.lower(),
            metrics=["accuracy"],
        )
        self.discriminator_model_cov.trainable = False

        # Reconstruction (Likelihood) loss:
        # The first loss corresponds to the likelihood - this tells us how well we
        # are explaining our data according to the current estimate of the
        # generative model, and is given by:
        # L = \sum_{t=1}^{T} log p(Y_t | \theta_t^m = \mu^{m,\theta}_t,
        #                                \theta_t^c = \mu^{c,\theta}_t)
        ll_loss = AdversarialLogLikelihoodLossLayer(
            self.config.n_channels, name="ll_loss"
        )

        # Regularization (Prior) Loss:
        # The second loss regularises the estimate of the latent, time-varying
        # parameters [$\theta^m$, $\theta^c$] using an adaptive prior - this penalises
        # when the posterior estimates of [$\theta^m$, $\theta^c$] deviate from the
        # prior:
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
            loss=[ll_loss, "binary_crossentropy", "binary_crossentropy"],
            loss_weights=[0.990, 0.005, 0.005],
            optimizer=optimizer,
        )

    def fit(self, training_data, epochs=None, verbose=1):
        """Train the model.

        Parameters
        ----------
        training_data : tensorflow.data.Dataset or osl_dynamics.data.Data
            Training data.
        epochs : int
            Number of epochs to train. Defaults to value in config if not passed.
        verbose : int
            Should we print a progress bar?

        Returns
        -------
        history : history
            History of discriminator_loss and generator_loss.
        """
        if epochs is None:
            epochs = self.config.n_epochs

        # Make sure training data is a TensorFlow Dataset
        training_data = self.make_dataset(training_data, shuffle=True, concatenate=True)

        # Path to save the best model weights
        timestr = time.strftime("%Y%m%d-%H%M%S")  # current date-time
        filepath = "tmp/"
        save_filepath = filepath + str(timestr) + "_best_model.h5"

        history = []
        best_val_loss = np.Inf
        for epoch in range(epochs):
            if verbose:
                print("Epoch {}/{}".format(epoch + 1, epochs))
                pb_i = utils.Progbar(
                    len(training_data), stateful_metrics=["D_m", "D_c", "G"]
                )

            for idx, batch in enumerate(training_data):
                # Generate real/fake input for the discriminator
                real = np.ones((len(batch["data"]), self.config.sequence_length, 1))
                fake = np.zeros((len(batch["data"]), self.config.sequence_length, 1))
                train_discriminator_mean = self._train_discriminator(
                    real, fake, self.discriminator_model_mean
                )
                train_discriminator_cov = self._train_discriminator(
                    real, fake, self.discriminator_model_cov
                )

                (
                    C_m,
                    alpha_posterior,
                    gamma_posterior,
                ) = self.inference_model.predict_on_batch(batch)
                alpha_prior = self.generator_model_mean.predict_on_batch(
                    alpha_posterior
                )
                gamma_prior = self.generator_model_cov.predict_on_batch(gamma_posterior)

                # Train discriminator, inference and generator model
                discriminator_loss_mean = train_discriminator_mean(
                    alpha_posterior, alpha_prior
                )
                discriminator_loss_cov = train_discriminator_cov(
                    gamma_posterior, gamma_prior
                )
                generator_loss = self.model.train_on_batch(batch, [batch, real, real])

                if verbose:
                    pb_i.add(
                        1,
                        values=[
                            ("D_m", discriminator_loss_mean[1]),
                            ("D_c", discriminator_loss_cov[1]),
                            ("G", generator_loss[0]),
                        ],
                    )

            if generator_loss[0] < best_val_loss:
                self.save_weights(save_filepath)
                print(
                    "Best model w/ val loss (generator) {} saved to {}".format(
                        generator_loss[0], save_filepath
                    )
                )
                best_val_loss = generator_loss[0]
            val_loss = self.model.test_on_batch([batch], [batch, real, real])

            history.append(
                {
                    "D_m": discriminator_loss_mean[1],
                    "D_c": discriminator_loss_cov[1],
                    "G": generator_loss[0],
                }
            )

        # Load the weights for the best model
        print("Loading best model:", save_filepath)
        self.load_weights(save_filepath)

        return history

    def _train_discriminator(self, real, fake, discriminator):
        def train(real_samples, fake_samples):
            discriminator.trainable = True
            loss_real = discriminator.train_on_batch(real_samples, real)
            loss_fake = discriminator.train_on_batch(fake_samples, fake)
            loss = np.add(loss_real, loss_fake) * 0.5
            discriminator.trainable = False
            return loss

        return train

    def get_mode_time_courses(
        self,
        inputs,
        concatenate=False,
    ):
        """Get mode time courses.

        This method is used to get mode time courses for the multi-time-scale model.

        Parameters
        ----------
        inputs : tensorflow.data.Dataset or osl_dynamics.data.Data
            Prediction data.
        concatenate : bool
            Should we concatenate alpha for each subject?

        Returns
        -------
        alpha : list or np.ndarray
            Alpha time course with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        gamma : list or np.ndarray
            Gamma time course with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        """
        if not self.config.multiple_dynamics:
            raise ValueError("Please use get_alpha for a single time scale model.")

        inputs = self.make_dataset(inputs, concatenate=concatenate)

        print("Getting mode time courses:")
        outputs_alpha = []
        outputs_gamma = []
        for dataset in inputs:
            alpha, gamma = self.inference_model.predict(dataset)[1:]

            alpha = np.concatenate(alpha)
            gamma = np.concatenate(gamma)

            outputs_alpha.append(alpha)
            outputs_gamma.append(gamma)

        if concatenate or len(outputs_alpha) == 1:
            outputs_alpha = np.concatenate(outputs_alpha)
            outputs_gamma = np.concatenate(outputs_gamma)

        return outputs_alpha, outputs_gamma

    def sample_mode_time_courses(self, alpha=None, gamma=None):
        """Uses the generator to predict the prior alpha and gamma.

        Parameters
        ----------
        alpha : np.ndarray
            Shape must be (n_samples, n_modes).

        gamma : np.ndarray
            Shape must be (n_samples, n_modes).

        Returns
        -------
        alpha : tuple of np.ndarray
            Sampled alpha.
        gamma : tuple of np.ndarray
            Sampled gamma.
        """
        n_samples = np.shape(alpha)[0]
        alpha_sampled = np.empty([n_samples, self.config.n_modes], dtype=np.float32)
        gamma_sampled = np.empty([n_samples, self.config.n_modes], dtype=np.float32)

        for i in trange(
            n_samples - self.config.sequence_length,
            desc="Predicting mode time course",
            ncols=98,
        ):
            # Extract the sequence
            alpha_input = alpha[i : i + self.config.sequence_length]
            alpha_input = alpha_input[np.newaxis, :, :]

            gamma_input = gamma[i : i + self.config.sequence_length]
            gamma_input = gamma_input[np.newaxis, :, :]

            # Predict the point estimates for theta one time step in the future
            alpha_sampled[i] = self.generator_model_mean.predict_on_batch(alpha_input)[
                0, 0
            ]
            gamma_sampled[i] = self.generator_model_cov.predict_on_batch(gamma_input)[
                0, 0
            ]

        return alpha_sampled, gamma_sampled

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
        return mdynemo_obs.get_means_stds_fcs(self.inference_model)

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
        mdynemo_obs.set_means_stds_fcs(
            self.inference_model, means, stds, fcs, update_initializer
        )


def _build_inference_model(config):
    # Inference RNN:
    #   alpha_t = zeta(theta^m_t}) where
    #   mu^{m,theta}_t = f(BLSTM(Y,omega^m_e),lambda_e^m)
    #   mu^{c,theta}_t = f(BLSTM(Y,omega^c_e),lambda_e^c)

    # Definition of layers
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_channels), name="data"
    )
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

    gamma_layer = layers.TimeDistributed(
        layers.Dense(config.n_modes, activation="softmax"), name="inf_gamma"
    )

    # Data flow
    data_drop = data_drop_layer(inputs)
    theta = inf_rnn_layer(data_drop)
    alpha = alpha_layer(theta)
    gamma = gamma_layer(theta)

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
        name="means",
    )

    stds_layer = DiagonalMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_stds,
        config.initial_stds,
        config.stds_epsilon,
        name="stds",
    )

    fcs_layer = CorrelationMatricesLayer(
        config.n_modes,
        config.n_channels,
        config.learn_fcs,
        config.initial_fcs,
        config.fcs_epsilon,
        name="fcs",
    )

    mix_means_layer = MixVectorsLayer(name="mix_means")
    mix_stds_layer = MixMatricesLayer(name="mix_stds")
    mix_fcs_layer = MixMatricesLayer(name="mix_fcs")
    matmul_layer = MatMulLayer(name="cov")
    concat_means_covs_layer = ConcatVectorsMatricesLayer(name="concat_means_covs")

    # Data flow
    mu = means_layer(inputs)  # inputs not used
    E = stds_layer(inputs)  # inputs not used
    D = fcs_layer(inputs)  # inputs not used

    m = mix_means_layer([alpha, mu])
    G = mix_stds_layer([alpha, E])
    F = mix_fcs_layer([gamma, D])
    C = matmul_layer([G, F, G])
    C_m = concat_means_covs_layer([m, C])

    return models.Model(inputs, [C_m, alpha, gamma], name="inference")


def _build_generator_model(config, name):
    # Model RNN:
    #   alpha_t = zeta(theta^m_t) where
    #   hat{mu}^{m,theta}_t = f(LSTM(theta^m_<t,omega^m_g),lambda_g^m)
    #   hat{mu}^{c,theta}_t = f(LSTM(theta^c_<t,omega^c_g),lambda_g^c)

    # Input
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_modes),
        name="gen_inp_" + name,
    )

    # Definition of layers
    drop_layer = layers.TimeDistributed(
        layers.Dropout(config.model_dropout, name="gen_data_drop_" + name)
    )
    mod_rnn_layer = ModelRNNLayer(
        config.model_rnn,
        config.model_normalization,
        config.model_activation,
        config.model_n_layers,
        config.model_n_units,
        config.model_dropout,
        config.model_regularizer,
        name="gen_rnn_" + name,
    )
    prior_layer = layers.TimeDistributed(
        layers.Dense(config.n_modes, activation="softmax"), name="gen_softmax_" + name
    )

    # Data flow
    theta_drop = drop_layer(inputs)
    mod_rnn = mod_rnn_layer(theta_drop)
    prior = prior_layer(mod_rnn)

    return models.Model(inputs, prior, name="generator_" + name)


def _build_discriminator_model(config, name):
    # Descriminator RNN:
    #   D_{theta^m_t} = sigma(f(BLSTM([zeta(hat{mu}^{m,theta}_t), zeta(mu^{m,theta}_t)],omega^m_d), lambda_d^m))
    #   D_{theta^c_t} = sigma(f(BLSTM([zeta(hat{mu}^{c,theta}_t), zeta(mu^{c,theta}_t)],omega^c_d), lambda_d^c))

    # Definition of layers
    inputs = layers.Input(
        shape=(config.sequence_length, config.n_modes), name="dis_inp_" + name
    )

    drop_layer = layers.TimeDistributed(
        layers.Dropout(config.model_dropout, name="dis_data_drop_" + name)
    )
    dis_rnn_layer = ModelRNNLayer(
        config.discriminator_rnn,
        config.discriminator_normalization,
        config.discriminator_activation,
        config.discriminator_n_layers,
        config.discriminator_n_units,
        config.discriminator_dropout,
        config.discriminator_regularizer,
        name="dis_rnn_" + name,
    )
    sigmoid_layer = layers.TimeDistributed(
        layers.Dense(1, activation="sigmoid"), name="dis_sigmoid_" + name
    )

    # Data flow
    theta_norm_drop = drop_layer(inputs)
    dis_rnn = dis_rnn_layer(theta_norm_drop)
    output = sigmoid_layer(dis_rnn)

    return models.Model(inputs, output, name="discriminator_" + name)
