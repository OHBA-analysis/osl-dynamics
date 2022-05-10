"""SAGE model.
"""

from dataclasses import dataclass
from typing import Literal, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.python.data import Dataset
from osl_dynamics.data import Data
from tqdm import trange
from osl_dynamics.models import dynemo_obs
import tensorflow_datasets as tfds
from osl_dynamics.inference.layers import (
    InferenceRNNLayer,
    MeanVectorsLayer,
    CovarianceMatricesLayer,
    MixVectorsLayer,
    MixMatricesLayer,
    ModelRNNLayer,
    NormalizationLayer,
    MixVectorsMatricesLayer,
    SampleNormalDistributionLayer,
    sageLogLikelihoodLossLayer,
)

@dataclass
class Config():
    """Settings for sage.

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
    inference_dropout : float
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
    model_dropout : float
        Dropout rate.


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

    # Inference network parameters
    inference_rnn: Literal["gru", "lstm"] = "lstm"
    inference_n_layers: int = 1
    inference_n_units: int = None
    inference_normalization: Literal[None, "batch", "layer"] = None
    inference_activation: str = None
    inference_dropout: float = 0.0

    # Model network parameters
    model_rnn: Literal["gru", "lstm"] = "lstm"
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: Literal[None, "batch", "layer"] = None
    model_activation: str = None
    model_dropout: float = 0.0

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None

    # Training parameters
    batch_size: int = None
    learning_rate: float = None
    gradient_clip: float = None
    n_epochs: int = None
    optimizer: tf.keras.optimizers.Optimizer = "adam"
    multi_gpu: bool = False
    strategy: str = None

    # Dimension parameters
    n_modes: int = None
    n_states: int = None
    n_channels: int = None
    sequence_length: int = None


class SAGE():
    """Sage model class.

    Parameters
    ----------
    config : osl_dynamics.models.sage.Config
    """

    def __init__(self, config):
        self.config = config

        print("Build models…")
        self._build_inference_model()
        self._build_generator_model()
        self._build_discriminator_model()
        self._build_and_compile_sage()

    def _make_dataset(self, inputs: Data):
        """Make a dataset.

        Parameters
        ----------
        inputs : osl_dynamics.data.Data
            Data object.

        Returns
        -------
        tensorflow.data.Dataset
            Tensorflow dataset that can be used for training.
        """
        if isinstance(inputs, Data):
            return inputs.dataset(self.config.sequence_length, shuffle=False)
        if isinstance(inputs, Dataset):
            return [inputs]
        if isinstance(inputs, str):
            return [Data(inputs).dataset(self.config.sequence_length, shuffle=False)]
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 2:
                return [
                    Data(inputs).dataset(self.config.sequence_length, shuffle=False)
                ]
            if inputs.ndim == 3:
                return [
                    Data(subject).dataset(self.config.sequence_length, shuffle=False)
                    for subject in inputs
                ]
        if check_iterable_type(inputs, Dataset):
            return inputs
        if check_iterable_type(inputs, str):
            datasets = [
                Data(subject).dataset(self.config.sequence_length, shuffle=False)
                for subject in inputs
            ]
            return datasets

    def _build_inference_model(self):

        print("Building Inference Model..")
        
        inputs = layers.Input(
            shape=(self.config.sequence_length, 
            self.config.n_channels), 
            name="data")

        # Inference RNN:
        # - Learns q(theta) ~ N(theta | inf_mu, inf_sigma), where
        #     - inf_mu    ~ affine(RNN(inputs_<=t))
        #     - inf_sigma ~ softplus(RNN(inputs_<=t))

        # Definition of layers

        data_drop_layer = layers.TimeDistributed(layers.Dropout(self.config.inference_dropout, name="data_drop"))
        inf_rnn_layer = InferenceRNNLayer(
            self.config.inference_rnn,
            self.config.inference_normalization,
            self.config.inference_activation,
            self.config.inference_n_layers,
            self.config.inference_n_units,
            self.config.inference_dropout,
            name="inf_rnn",
        )

        alpha_layer =  layers.TimeDistributed(layers.Dense(self.config.n_modes, 
            activation='softmax',
            name="alpha_inf"))

        # Data flow
        data_drop = data_drop_layer(inputs)
        inf_rnn = inf_rnn_layer(data_drop)
        alpha = alpha_layer(inf_rnn)


        # Observation model:
        # - We use a multivariate normal with a mean vector and covariance matrix for
        #   each mode as the observation model.
        # - We calculate the likelihood of generating the training data with alpha
        #   and the observation model.

        # Definition of layers
        means_layer = MeanVectorsLayer(
            self.config.n_modes,
            self.config.n_channels,
            self.config.learn_means,
            self.config.initial_means,
            name="means",
        )
        covs_layer = CovarianceMatricesLayer(
            self.config.n_modes,
            self.config.n_channels,
            self.config.learn_covariances,
            self.config.initial_covariances,
            name="covs",
        )
        mix_means_layer = MixVectorsLayer(name="mix_means")
        mix_covs_layer = MixMatricesLayer(name="mix_covs")
        mix_means_covs_layer = MixVectorsMatricesLayer(name="mix_means_covs")

        # Data flow
        mu = means_layer(inputs)  # inputs not used
        D = covs_layer(inputs)  # inputs not used
        m = mix_means_layer([alpha, mu])
        C = mix_covs_layer([alpha, D])
        C_m = mix_means_covs_layer([m, C])
        

        self.inference_model = Model(inputs, [C_m, alpha])
        self.inference_model.summary()

    def _build_generator_model(self):
        print("Building Generator Model…")

        # Model RNN:
        # - Learns p(theta_t |theta_<t) ~ N(theta_t | mod_mu, mod_sigma), where
        #     - mod_mu    ~ affine(RNN(theta_<t))
        #     - mod_sigma ~ softplus(RNN(theta_<t))

        # Definition of layers
        generator_input = layers.Input(shape=(self.config.sequence_length, 
            self.config.n_modes),
            name="generator_input")


        theta_norm_drop_layer = layers.TimeDistributed(layers.Dropout(self.config.model_dropout,
            name="theta_norm_drop_generator"))


        mod_rnn_layer = ModelRNNLayer(
            self.config.model_rnn,
            self.config.model_normalization,
            self.config.model_activation,
            self.config.model_n_layers,
            self.config.model_n_units,
            self.config.model_dropout,
            name="mod_rnn",
        )

        alpha_layer =  layers.TimeDistributed(layers.Dense(self.config.n_modes, 
            activation='softmax',
            name="alpha_gen"))


        # Data flow
        theta_norm_drop = theta_norm_drop_layer(generator_input)
        alpha_prior = mod_rnn_layer(theta_norm_drop)
        alpha_prior_softmax = alpha_layer(alpha_prior)

        self.generator_model = Model(generator_input, alpha_prior_softmax)

        self.generator_model.summary()


    def _build_discriminator_model(self):
        print("Building Discriminator…")

        # Definition of layers
        discriminator_input = layers.Input(shape=(self.config.sequence_length,
            self.config.n_modes),
            name="data")


        theta_norm_drop_layer = layers.TimeDistributed(layers.Dropout(self.config.model_dropout, 
            name="theta_norm_drop_descriminator"))
        
        mod_rnn_layer = ModelRNNLayer(
            self.config.model_rnn,
            self.config.model_normalization,
            self.config.model_activation,
            self.config.model_n_layers,
            self.config.model_n_units,
            self.config.model_dropout,
            name="des_rnn",
        )

        sigmoid_layer = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))
        
        # Data flow
        theta_norm_drop = theta_norm_drop_layer(discriminator_input)
        discriminator_sequence = mod_rnn_layer(theta_norm_drop)
        sigmoid_layer_softmax = sigmoid_layer(discriminator_sequence)

        self.discriminator_model = Model(discriminator_input, sigmoid_layer_softmax)
        self.discriminator_model.summary()

    def _build_and_compile_sage(self):

        print("Compile Discriminator…")

        self.discriminator_model.compile(loss='binary_crossentropy',
            optimizer=self.config.optimizer,
            metrics=['accuracy'])

        self.discriminator_model.trainable = False

        print("Conecting models…")
        real_input = layers.Input(shape=(self.config.sequence_length,
            self.config.n_channels),
            name="data")


        C_m, alpha_posterior = self.inference_model(real_input)
        alpha_prior = self.generator_model(alpha_posterior)
        discriminator_output_prior = self.discriminator_model(alpha_prior) 
        discriminator_output_posterior = self.discriminator_model(alpha_posterior)

        self.sage = Model(real_input, [C_m, discriminator_output_prior, discriminator_output_posterior],
            name = 'sage_model')

        log_kl_loss = sageLogLikelihoodLossLayer(self.config.n_channels, name="ll_loss")
        log_kl_loss.__name__ = 'log_lh' # need to fix this as error without it

        self.sage.compile(loss=[log_kl_loss, 'binary_crossentropy','binary_crossentropy'],
            loss_weights=[0.998, 0.001, 0.001],
            optimizer=self.config.optimizer)


    def _discriminator_training(self, real, fake):
        def train(real_samples,fake_samples):
            self.discriminator_model.trainable = True
            loss_real = self.discriminator_model.train_on_batch(real_samples, real)
            loss_fake = self.discriminator_model.train_on_batch(fake_samples, fake)
            loss = np.add(loss_real, loss_fake) * 0.5
            self.discriminator_model.trainable = False

            return loss
        return train

    def train(self, train_data):
        
        real = np.ones((self.config.batch_size,self.config.sequence_length, 1))
        fake = np.zeros((self.config.batch_size,self.config.sequence_length, 1))
        
        # Batch-wise training
        train_discriminator = self._discriminator_training(real, fake)
        history = []

        for epoch in range(self.config.n_epochs):
            
            for idx, batch in enumerate(train_data):
                C_m, alpha_posterior = self.inference_model.predict_on_batch(batch)
                alpha_prior = self.generator_model.predict_on_batch(alpha_posterior)

                discriminator_loss = train_discriminator(alpha_prior,alpha_posterior)
                #  Train Generator
                generator_loss = self.sage.train_on_batch(batch, [batch, real,fake])

            # Plot the progress
            print ("———————————————————")
            print ("******************Epoch {}***************************".format(epoch))
            print ("Discriminator loss: {}".format(discriminator_loss[1]))
            print ("Generator loss: {}".format(generator_loss[0]))
            print ("———————————————————")

            history.append({"D":discriminator_loss[1],"G":generator_loss[1]})

        return history


    def get_alpha(self, inputs, concatenate: bool = True) -> Union[list, np.ndarray]:
        """Mode mixing factors, alpha.

        Parameters
        ----------
        inputs : tensorflow.data.Dataset
            Prediction dataset.

        Returns
        -------
        list or np.ndarray
            Mode mixing factors with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        """
        outputs = []
        for dataset in inputs:
            alpha = self.inference_model.predict(dataset)[1]
            alpha = np.concatenate(alpha)
            outputs.append(alpha)

        if concatenate or len(outputs_alpha) == 1:
            outputs = np.concatenate(outputs)

        return outputs

