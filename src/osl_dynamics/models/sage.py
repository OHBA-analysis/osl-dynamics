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
import time, os
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
    inference_activation: str = "elu"
    inference_dropout: float = 0.0

    # Model network parameters
    model_rnn: Literal["gru", "lstm"] = "lstm"
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: Literal[None, "batch", "layer"] = None
    model_activation: str = "elu"
    model_dropout: float = 0.0

    # Descriminator network parameters
    des_rnn: Literal["gru", "lstm"] = "lstm"
    des_n_layers: int = 1
    des_n_units: int = None
    des_normalization: Literal[None, "batch", "layer"] = None
    des_activation: str = "elu"
    des_dropout: float = 0.0

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
        # - \alpha_{t} = \zeta({\theta^{m}_{t}}) where
        #   \mu^{m,\theta}_t  = f(LSTM_{bi}(Y,\omega^m_e),\lambda_e^m)
        #  \mu^{c,\theta}_t = f(LSTM_{bi}(Y,\omega^c_e),\lambda_e^c)

        # Definition of layers

        data_drop_layer = layers.TimeDistributed(layers.Dropout(self.config.inference_dropout, name="data_drop_inf"))
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
            activation='softmax'),
            name="alpha_inf")

        # Data flow
        data_drop = data_drop_layer(inputs)
        theta = inf_rnn_layer(data_drop)
        alpha = alpha_layer(theta)


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
        # - \alpha_{t} = \zeta({\theta^{m}_{t}}) where
        #    \hat{\mu}^{m,\theta}_{t} = f (LSTM_{uni} (\theta^{m}_{<t},\omega^m_g), \lambda_g^m)
        #     \hat{\mu}^{c,\theta}_{t}   = f (LSTM_{uni} (\theta^{c}_{<t},\omega^c_g), \lambda_g^c)

        # Definition of layers
        generator_input = layers.Input(shape=(self.config.sequence_length, 
            self.config.n_modes),
            name="generator_input")


        drop_layer = layers.TimeDistributed(layers.Dropout(self.config.model_dropout,
            name="data_drop_gen"))


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
            activation='softmax'),
            name="alpha_gen")

        # Data flow
        theta_drop = drop_layer(generator_input)
        theta_drop_prior = mod_rnn_layer(theta_drop)
        alpha_prior = alpha_layer(theta_drop_prior)

        self.generator_model = Model(generator_input, alpha_prior)
        self.generator_model.summary()


    def _build_discriminator_model(self):
        print("Building Discriminator…")

        # Descriminator RNN:
        #     D_{\theta^m_t} = \sigma (f (LSTM_{bi}([\zeta(\hat{\mu}^{m,\theta}_{t}), \zeta(\mu^{m,\theta}_t)],\omega^m_d), \lambda_d^m))
        #     D_{\theta^c_t} = \sigma (f (LSTM_{bi}([\zeta(\hat{\mu}^{c,\theta}_{t}), \zeta(\mu^{c,\theta}_t)],\omega^c_d), \lambda_d^c))

        # Definition of layers
        discriminator_input = layers.Input(shape=(self.config.sequence_length,
            self.config.n_modes),
            name="data")


        drop_layer = layers.TimeDistributed(layers.Dropout(self.config.model_dropout, 
            name="data_drop_des"))
        
        des_rnn_layer = ModelRNNLayer(
            self.config.des_rnn,
            self.config.des_normalization,
            self.config.des_activation,
            self.config.des_n_layers,
            self.config.des_n_units,
            self.config.model_dropout,
            name="des_rnn",
        )

        sigmoid_layer = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))
        
        # Data flow
        theta_norm_drop = drop_layer(discriminator_input)
        discriminator_sequence = des_rnn_layer(theta_norm_drop)
        discriminator_output = sigmoid_layer(discriminator_sequence)

        self.discriminator_model = Model(discriminator_input, discriminator_output)
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

        self.sage = Model(real_input, [C_m, discriminator_output_prior],
            name = 'sage_model')

        # Reconstruction (Likelihood) loss:
        #   -The first loss corresponds to the likelihood - this tells us how well we are explaining our data 
        #   -according to the current estimate of the generative model, and is given by:
        #   -L =  \sum_{t=1}^{T}\, log \,p(Y_t | \theta_t^m = \mu^{m,\theta}_t, \theta_t^c = \mu^{c,\theta}_t )

        log_kl_loss = sageLogLikelihoodLossLayer(self.config.n_channels, name="ll_loss")
        log_kl_loss.__name__ = 'll_loss' # need to fix this as error without it

        # Regularization (Prior) Loss:
        #   -The second loss regularises the estimate of the latent, time-varying parameters [$\theta^m$, $\theta^c$] using an adaptive prior
        #   -this penalises when the posterior estimates of [$\theta^m$, $\theta^c$] deviate from the prior:
        #   -R = \sum_{t=1}^{T}  [\textnormal{CrossEntropy} \,(\mu^{m,\theta}_t || \hat{\mu}^{m,\theta}_{t}) + \textnormal{CrossEntropy}\, (\mu^{c,\theta}_t || \hat{\mu}^{c,\theta}_{t})]

        self.sage.compile(loss=[log_kl_loss, 'binary_crossentropy'],
            loss_weights=[0.995, 0.005],
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate))


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
        
        """Train the Model

        Parameters
        ----------
        inputs : tensorflow.data.Dataset
            Prediction dataset.

        Returns
        -------
        history: History of discriminator_loss and generator_loss
        """

        # Saving the Model Weights
        timestr = time.strftime("%Y%m%d-%H%M%S") # current date-time
        filepath = os.getcwd() + "/tmp/" 
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)    

        save_filepath = filepath + str(timestr) + "_best_model.h5"

        # Generating real/fake input for the descriminator
        real = np.ones((self.config.batch_size,self.config.sequence_length, 1))
        fake = np.zeros((self.config.batch_size,self.config.sequence_length, 1))
        
        # Batch-wise trainingq
        train_discriminator = self._discriminator_training(real, fake)
        history = []
        best_val_loss = 9999999

        for epoch in range(self.config.n_epochs):
            
            for idx, batch in enumerate(train_data):
                C_m, alpha_posterior = self.inference_model.predict_on_batch(batch)
                alpha_prior = self.generator_model.predict_on_batch(alpha_posterior)

                discriminator_loss = train_discriminator(alpha_posterior, alpha_prior)
                #  Train Generator
                generator_loss = self.sage.train_on_batch(batch, [batch, real])

            if generator_loss[0] < best_val_loss:
                self.sage.save_weights(save_filepath) 
                print("Best model w/ val loss (generator) {} saved to {}".
                    format(generator_loss[0], save_filepath))
                best_val_loss = generator_loss[0]  
            val_loss = self.sage.test_on_batch([batch], [batch,real])
            
            # Plot the progress
            print ("———————————————————")
            print ("******************Epoch {}***************************".format(epoch))
            print ("Discriminator loss: {}".format(discriminator_loss[1]))
            print ("Generator loss: {}".format(generator_loss[0]))
            print ("———————————————————")

            history.append({"D":discriminator_loss[1],"G":generator_loss[0]})

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

    def get_covariances(self):
        """Get the covariances of each mode.

        Returns
        -------
        np.ndarary
            Mode covariances.
        """
        return dynemo_obs.get_covariances(self.inference_model)

    def get_means_covariances(self):
        """Get the means and covariances of each mode.

        Returns
        -------
        means : np.ndarary
            Mode means.
        covariances : np.ndarray
            Mode covariances.
        """
        return dynemo_obs.get_means_covariances(self.inference_model)

    def gen_alpha(self, alpha: np.ndarray = None) -> np.ndarray:
        """Uses the Generator RNN to predict the prior alphas.

        Parameters
        ----------
        alpha : np.ndarray
            Shape must be
            (n_samples, n_modes).

        Returns
        -------
        np.ndarray
            Predicted alpha.
        """

        # number of samples
        n_samples = np.shape(alpha)[0]
        alpha_sampled = np.empty([n_samples, self.config.n_modes], dtype=np.float32)

        for i in trange(n_samples-self.config.sequence_length, desc="Predicting mode time course", ncols=98):
            # Extract the sequence
            alpha_input = alpha[i:i+self.config.sequence_length]
            alpha_input = alpha_input[np.newaxis,:,:]
            # Predict the point estimates for theta one time step
            # in the future,
            alpha_sampled[i] = self.generator_model.predict_on_batch(alpha_input)[0,0]

        return alpha_sampled
