"""Model class for a generative model with Gaussian observations.

"""

import logging
from operator import lt

import numpy as np
from tensorflow.keras import Model, layers
from tensorflow.nn import softplus
from tqdm import trange
from vrad.inference.functions import (
    cholesky_factor,
    cholesky_factor_to_full_matrix,
    trace_normalize,
)
from vrad.models import BaseModel
from vrad.models.layers import (
    DummyLayer,
    InferenceRNNLayers,
    KLDivergenceLayer,
    LogLikelihoodLayer,
    MeansCovsLayer,
    MixMeansCovsLayer,
    ModelRNNLayers,
    SampleNormalDistributionLayer,
    StateMixingFactorsLayer,
)
from vrad.utils.misc import check_arguments

_logger = logging.getLogger("VRAD")


class RNNGaussian(BaseModel):
    """Inference RNN and generative model with Gaussian observations.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    learn_means : bool
        Should we learn the mean vector for each state?
    learn_covariances : bool
        Should we learn the covariance matrix for each state?
    rnn_type : str
        RNN to use, either 'lstm' or 'gru'.
    rnn_normalization : str
        Type of normalization to use in the inference network and generative model.
        Either 'layer', 'batch' or None.
    n_layers_inference : int
        Number of layers in the inference network.
    n_layers_model : int
        Number of layers in the generative model neural network.
    n_units_inference : int
        Number of units/neurons in the inference network.
    n_units_model : int
        Number of units/neurons in the generative model neural network.
    dropout_rate_inference : float
        Dropout rate in the inference network.
    dropout_rate_model : float
        Dropout rate in the generative model neural network.
    theta_normalization : str
        Type of normalization to apply to the posterior samples, theta_t.
        Either 'layer', 'batch' or None.
    alpha_xform : str
        Functional form of alpha_t. Either 'categorical', 'softmax', 'softplus' or
        'relu'.
    alpha_temperature : float
        Temperature parameter for when alpha_xform = 'softmax' or 'categorical'.
    learn_alpha_scaling : bool
        Should we learn a scaling for alpha?
    normalize_covariances : bool
        Should we trace normalize the state covariances?
    do_annealing : bool
        Should we use KL annealing during training?
    annealing_sharpness : float
        Parameter to control the annealing curve.
    n_epochs_annealing : int
        Number of epochs to perform annealing.
    learning_rate : float
        Learning rate for updating model parameters/weights.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    initial_means : np.ndarray
        Initial values for the state means. Should have shape (n_states, n_channels).
    initial_covariances : np.ndarray
        Initial values for the state covariances. Should have shape (n_states,
        n_channels, n_channels).
    """

    def __init__(
        self,
        n_states: int,
        n_channels: int,
        sequence_length: int,
        learn_means: bool,
        learn_covariances: bool,
        rnn_type: str,
        rnn_normalization: str,
        n_layers_inference: int,
        n_layers_model: int,
        n_units_inference: int,
        n_units_model: int,
        dropout_rate_inference: float,
        dropout_rate_model: float,
        theta_normalization: str,
        alpha_xform: str,
        alpha_temperature: float,
        learn_alpha_scaling: bool,
        normalize_covariances: bool,
        do_annealing: bool,
        annealing_sharpness: float,
        n_epochs_annealing: int,
        learning_rate: float,
        multi_gpu: bool = False,
        strategy: str = None,
        initial_means: np.ndarray = None,
        initial_covariances: np.ndarray = None,
    ):
        # Validation
        if alpha_xform not in ["categorical", "softmax", "softplus", "relu"]:
            raise ValueError(
                "alpha_xform must be 'categorical', 'softmax', 'softplus' or 'relu'."
            )

        # Parameters related to the observation model
        self.learn_means = learn_means
        self.learn_covariances = learn_covariances
        self.initial_means = initial_means
        self.initial_covariances = initial_covariances
        self.alpha_xform = alpha_xform
        self.alpha_temperature = alpha_temperature
        self.learn_alpha_scaling = learn_alpha_scaling
        self.normalize_covariances = normalize_covariances

        # Initialise the model base class
        # This will build and compile the keras model
        super().__init__(
            n_states=n_states,
            n_channels=n_channels,
            sequence_length=sequence_length,
            rnn_type=rnn_type,
            rnn_normalization=rnn_normalization,
            n_layers_inference=n_layers_inference,
            n_layers_model=n_layers_model,
            n_units_inference=n_units_inference,
            n_units_model=n_units_model,
            dropout_rate_inference=dropout_rate_inference,
            dropout_rate_model=dropout_rate_model,
            theta_normalization=theta_normalization,
            do_annealing=do_annealing,
            annealing_sharpness=annealing_sharpness,
            n_epochs_annealing=n_epochs_annealing,
            learning_rate=learning_rate,
            multi_gpu=multi_gpu,
            strategy=strategy,
        )

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(
            n_states=self.n_states,
            n_channels=self.n_channels,
            sequence_length=self.sequence_length,
            rnn_type=self.rnn_type,
            rnn_normalization=self.rnn_normalization,
            n_layers_inference=self.n_layers_inference,
            n_layers_model=self.n_layers_model,
            n_units_inference=self.n_units_inference,
            n_units_model=self.n_units_model,
            dropout_rate_inference=self.dropout_rate_inference,
            dropout_rate_model=self.dropout_rate_model,
            theta_normalization=self.theta_normalization,
            learn_means=self.learn_means,
            learn_covariances=self.learn_covariances,
            initial_means=self.initial_means,
            initial_covariances=self.initial_covariances,
            alpha_xform=self.alpha_xform,
            alpha_temperature=self.alpha_temperature,
            learn_alpha_scaling=self.learn_alpha_scaling,
            normalize_covariances=self.normalize_covariances,
        )

    def predict(self, *args, **kwargs):
        """Wrapper for the standard keras predict method.

        Returns
        -------
        dict
            Dictionary with labels for each prediction.
        """
        predictions = self.model.predict(*args, *kwargs)
        return_names = ["ll_loss", "kl_loss", "alpha_t"]
        predictions_dict = dict(zip(return_names, predictions))
        return predictions_dict

    def predict_states(self, inputs, *args, **kwargs):
        """State mixing factors, alpha_t.
        
        Parameters
        ----------
        inputs : tensorflow.data.Dataset
            Prediction dataset.

        Returns
        -------
        np.ndarray
            State mixing factors with shape (n_samples, n_states).
        """
        inputs = self._make_dataset(inputs)
        outputs = []
        for dataset in inputs:
            alpha_t = self.predict(dataset, *args, **kwargs)["alpha_t"]
            alpha_t = np.concatenate(alpha_t)
            outputs.append(alpha_t)
        return outputs

    def losses(self, dataset):
        """Calculates the log-likelihood and KL loss for a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate losses for.

        Returns
        -------
        ll_loss : float
            Negative log-likelihood loss.
        kl_loss : float
            KL divergence loss.
        """
        if isinstance(dataset, list):
            predictions = [self.predict(subject) for subject in dataset]
            ll_loss = np.sum([np.sum(p["ll_loss"]) for p in predictions])
            kl_loss = np.sum([np.sum(p["kl_loss"]) for p in predictions])
        else:
            predictions = self.predict(dataset)
            ll_loss = np.sum(predictions["ll_loss"])
            kl_loss = np.sum(predictions["kl_loss"])
        return ll_loss, kl_loss

    def free_energy(self, dataset):
        """Calculates the variational free energy of a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate the variational free energy for.

        Returns
        -------
        float
            Variational free energy for the dataset.
        """
        ll_loss, kl_loss = self.losses(dataset)
        free_energy = ll_loss + kl_loss
        return free_energy

    def initialize_means_covariances(
        self,
        n_initializations,
        n_epochs_initialization,
        training_dataset,
        verbose=1,
        use_tqdm=False,
        tqdm_class=None,
    ):
        """Initialize the means and covariances.
    
        The model is trained for a few epochs and the model with the best
        free energy is chosen.

        Parameters
        ----------
        n_initializations : int
            Number of initializations.
        n_epochs_initialization : int
            Number of epochs to train the model.
        training_dataset : tensorflow.data.Dataset
            Dataset to use for training.
        verbose : int
            Show verbose (1) or not (0).
        use_tqdm : bool
            Should we use a tqdm progress bar instead of the usual Tensorflow output?
        tqdm_class : tqdm
            Tqdm class for the progress bar.
        """
        if n_initializations is None or n_initializations == 0:
            _logger.warning(
                "Number of initializations was set to zero. "
                + "Skipping initialization."
            )
            return

        # Pick the initialization with the lowest free energy
        best_free_energy = np.Inf
        for n in range(n_initializations):
            print(f"Initialization {n}")
            self.reset_model()
            history = self.fit(
                training_dataset,
                epochs=n_epochs_initialization,
                verbose=verbose,
                use_tqdm=use_tqdm,
                tqdm_class=tqdm_class,
            )
            free_energy = history.history["loss"][-1]
            if free_energy < best_free_energy:
                best_initialization = n
                best_free_energy = free_energy
                best_weights = self.model.get_weights()
                best_optimizer = self.model.optimizer

        print(f"Using initialization {best_initialization}")
        self.compile(optimizer=best_optimizer)
        self.model.set_weights(best_weights)
        if self.do_annealing:
            self.annealing_factor.assign(0.0)

    def burn_in(self, *args, **kwargs):
        """Burn-in training phase.

        Fits the model with means and covariances non-trainable.
        """
        if check_arguments(args, kwargs, 3, "epochs", 1, lt):
            _logger.warning(
                "Number of burn-in epochs is less than 1. Skipping burn-in."
            )
            return

        # Make means and covariances non-trainable and compile
        means_covs_layer = self.model.get_layer("means_covs")
        means_covs_layer.trainable = False
        self.compile()

        # Train the model
        self.fit(*args, **kwargs)

        # Make means and covariances trainable again and compile
        means_covs_layer.trainable = True
        self.compile()

    def get_means_covariances(self, alpha_scale=True):
        """Get the means and covariances of each state.

        Parameters
        ----------
        alpah_scale : bool
            Should we apply alpha scaling? Default is True.

        Returns
        -------
        means : np.ndarray
            State means.
        covariances : np.ndarary
            State covariances.
        """
        # Get the means and covariances from the MeansCovsLayer
        means_covs_layer = self.model.get_layer("means_covs")
        means = means_covs_layer.means.numpy()
        cholesky_covariances = means_covs_layer.cholesky_covariances
        covariances = cholesky_factor_to_full_matrix(cholesky_covariances).numpy()

        # Normalise covariances
        if self.normalize_covariances:
            covariances = trace_normalize(covariances)

        # Apply alpha scaling
        if alpha_scale:
            alpha_scaling = self.get_alpha_scaling()
            means *= alpha_scaling[:, np.newaxis]
            covariances *= alpha_scaling[:, np.newaxis, np.newaxis]

        return means, covariances

    def set_means_covariances(self, means=None, covariances=None):
        """Set the means and covariances of each state.

        Parameters
        ----------
        means : np.ndarray
            State means.
        covariances : np.ndarray
            State covariances.
        """
        means_covs_layer = self.model.get_layer("means_covs")
        layer_weights = means_covs_layer.get_weights()

        # Replace means in the layer weights
        if means is not None:
            for i in range(len(layer_weights)):
                if layer_weights[i].shape == means.shape:
                    layer_weights[i] = means

        # Replace covariances in the layer weights
        if covariances is not None:
            for i in range(len(layer_weights)):
                if layer_weights[i].shape == covariances.shape:
                    layer_weights[i] = cholesky_factor(covariances)

        # Set the weights of the layer
        means_covs_layer.set_weights(layer_weights)

    def get_alpha_scaling(self):
        """Get the alpha scaling of each state.

        Returns
        ----------
        alpha_scaling : bool
            Alpha scaling for each state.
        """
        mix_means_covs_layer = self.model.get_layer("mix_means_covs")
        alpha_scaling = mix_means_covs_layer.alpha_scaling.numpy()
        alpha_scaling = softplus(alpha_scaling).numpy()
        return alpha_scaling

    def sample_state_time_course(self, n_samples):
        """Uses the model RNN to sample a state time course.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.

        Returns
        -------
        sampled_stc : np.ndarray
            Sampled state time course.
        """
        # Get layers
        model_rnn_layer = self.model.get_layer("model_rnn")
        mu_theta_jt_layer = self.model.get_layer("mu_theta_jt")
        log_sigma_theta_jt_layer = self.model.get_layer("log_sigma_theta_jt")
        theta_t_norm_layer = self.model.get_layer("theta_t_norm")
        alpha_t_layer = self.model.get_layer("alpha_t")

        # State time course and sequence of the underlying logits theta_t
        sampled_stc = np.zeros([n_samples, self.n_states])
        theta_t_norm = np.zeros([self.sequence_length, self.n_states], dtype=np.float32)

        # Normally distributed random numbers used to sample the logits theta_t
        epsilon = np.random.normal(0, 1, [n_samples + 1, self.n_states]).astype(
            np.float32
        )

        # Activate the first state for the first sample
        theta_t_norm[-1, 0] = 1

        # Sample state time course
        for i in trange(n_samples, desc="Sampling state time course", ncols=98):

            # If there are leading zeros we trim theta_t so that we don't pass the zeros
            trimmed_theta_t = theta_t_norm[~np.all(theta_t_norm == 0, axis=1)][
                np.newaxis, :, :
            ]

            # Predict the probability distribution function for theta_t one time step
            # in the future,
            # p(theta_t|theta_<t) ~ N(mu_theta_jt, sigma_theta_jt)
            model_rnn = model_rnn_layer(trimmed_theta_t)
            mu_theta_jt = mu_theta_jt_layer(model_rnn)[0, -1]
            log_sigma_theta_jt = log_sigma_theta_jt_layer(model_rnn)[0, -1]
            sigma_theta_jt = np.exp(log_sigma_theta_jt)

            # Shift theta_t one time step to the left
            theta_t_norm = np.roll(theta_t_norm, -1, axis=0)

            # Sample from the probability distribution function
            theta_t = mu_theta_jt + sigma_theta_jt * epsilon[i]
            theta_t_norm[-1] = theta_t_norm_layer(theta_t[np.newaxis, np.newaxis, :])[0]

            # Calculate the state mixing factors
            alpha_t = alpha_t_layer(theta_t_norm[-1][np.newaxis, np.newaxis, :])

            # Hard classify the state time course
            sampled_stc[i, np.argmax(alpha_t)] = 1

        return sampled_stc


def _model_structure(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    rnn_type: str,
    rnn_normalization: str,
    n_layers_inference: int,
    n_layers_model: int,
    n_units_inference: int,
    n_units_model: int,
    dropout_rate_inference: float,
    dropout_rate_model: float,
    theta_normalization: str,
    learn_means: bool,
    learn_covariances: bool,
    initial_means: np.ndarray,
    initial_covariances: np.ndarray,
    alpha_xform: str,
    alpha_temperature: float,
    learn_alpha_scaling: bool,
    normalize_covariances: bool,
):
    """Model structure.

    Parameters
    ----------
    n_states : int
        Numeber of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    rnn_type : int
        RNN to use, either 'lstm' or 'gru'.
    rnn_normalization : str
        Type of normalization to use in the inference network and generative model.
        Either 'layer', 'batch' or None.
    n_layers_inference : int
        Number of layers in the inference network.
    n_layers_model : int
        Number of layers in the generative model neural network.
    n_units_inference : int
        Number of units/neurons in the inference network.
    n_units_model : int
        Number of units/neurons in the generative model neural network.
    dropout_rate_inference : float
        Dropout rate in the inference network.
    dropout_rate_model : float
        Dropout rate in the generative model neural network.
    theta_normalization : str
        Type of normalization to apply to the posterior samples, theta_t.
        Either 'layer', 'batch' or None.
    learn_means : bool
        Should we learn the mean vector for each state?
    learn_covariances : bool
        Should we learn the covariance matrix for each state?
    initial_means : np.ndarray
        Initial values for the state means. Should have shape (n_states, n_channels).
    initial_covariances : np.ndarray
        Initial values for the state covariances. Should have shape (n_states,
        n_channels, n_channels).
    alpha_xform : str
        Functional form of alpha_t. Either 'categorical', 'softmax', 'softplus' or
        'relu'.
    alpha_temperature : float
        Temperature parameter for when alpha_xform = 'softmax' or 'categorical'.
    learn_alpha_scaling : bool
        Should we learn a scaling for alpha?
    normalize_covariances : bool
        Should we trace normalize the state covariances?

    Returns
    -------
    tensorflow.keras.Model
        Keras model built using the functional API.
    """

    # Layer for input
    inputs = layers.Input(shape=(sequence_length, n_channels), name="data")

    # Inference RNN:
    # - Learns q(theta_t) ~ N(m_theta_t, s_theta_t), where
    #     - m_theta_t     ~ affine(RNN(inputs_<=t))
    #     - log_s_theta_t ~ affine(RNN(inputs_<=t))

    # Definition of layers
    inference_input_dropout_layer = layers.Dropout(
        dropout_rate_inference, name="data_drop"
    )
    inference_output_layers = InferenceRNNLayers(
        rnn_type,
        rnn_normalization,
        n_layers_inference,
        n_units_inference,
        dropout_rate_inference,
        name="inference_rnn",
    )
    m_theta_t_layer = layers.Dense(n_states, name="m_theta_t")
    log_s_theta_t_layer = layers.Dense(n_states, name="log_s_theta_t")

    # Layers to sample theta_t from q(theta_t) and to convert to state mixing
    # factors alpha_t
    theta_t_layer = SampleNormalDistributionLayer(name="theta_t")
    if theta_normalization == "layer":
        theta_t_norm_layer = layers.LayerNormalization(name="theta_t_norm")
    elif theta_normalization == "batch":
        theta_t_norm_layer = layers.BatchNormalization(name="theta_t_norm")
    else:
        theta_t_norm_layer = DummyLayer(name="theta_t_norm")
    alpha_t_layer = StateMixingFactorsLayer(
        alpha_xform, alpha_temperature, name="alpha_t"
    )

    # Data flow
    inference_input_dropout = inference_input_dropout_layer(inputs)
    inference_output = inference_output_layers(inference_input_dropout)
    m_theta_t = m_theta_t_layer(inference_output)
    log_s_theta_t = log_s_theta_t_layer(inference_output)
    theta_t = theta_t_layer([m_theta_t, log_s_theta_t])
    theta_t_norm = theta_t_norm_layer(theta_t)
    alpha_t = alpha_t_layer(theta_t_norm)

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each state as the observation model.
    # - We calculate the likelihood of generating the training data with alpha_t
    #   and the observation model.

    # Definition of layers
    means_covs_layer = MeansCovsLayer(
        n_states,
        n_channels,
        learn_means=learn_means,
        learn_covariances=learn_covariances,
        normalize_covariances=normalize_covariances,
        initial_means=initial_means,
        initial_covariances=initial_covariances,
        name="means_covs",
    )
    mix_means_covs_layer = MixMeansCovsLayer(
        n_states, n_channels, learn_alpha_scaling, name="mix_means_covs"
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    mu_j, D_j = means_covs_layer(inputs)  # inputs not used
    m_t, C_t = mix_means_covs_layer([alpha_t, mu_j, D_j])
    ll_loss = ll_loss_layer([inputs, m_t, C_t])

    # Model RNN:
    # - Learns p(theta_t|theta_<t) ~ N(mu_theta_jt, sigma_theta_jt), where
    #     - mu_theta_jt        ~ affine(RNN(theta_<t))
    #     - log_sigma_theta_jt ~ affine(RNN(theta_<t))

    # Definition of layers
    model_input_dropout_layer = layers.Dropout(
        dropout_rate_model, name="theta_t_norm_drop"
    )
    model_output_layers = ModelRNNLayers(
        rnn_type,
        rnn_normalization,
        n_layers_model,
        n_units_model,
        dropout_rate_model,
        name="model_rnn",
    )
    mu_theta_jt_layer = layers.Dense(n_states, name="mu_theta_jt")
    log_sigma_theta_jt_layer = layers.Dense(n_states, name="log_sigma_theta_jt")
    kl_loss_layer = KLDivergenceLayer(name="kl")

    # Data flow
    model_input_dropout = model_input_dropout_layer(theta_t_norm)
    model_output = model_output_layers(model_input_dropout)
    mu_theta_jt = mu_theta_jt_layer(model_output)
    log_sigma_theta_jt = log_sigma_theta_jt_layer(model_output)
    kl_loss = kl_loss_layer([m_theta_t, log_s_theta_t, mu_theta_jt, log_sigma_theta_jt])

    return Model(inputs=inputs, outputs=[ll_loss, kl_loss, alpha_t])
