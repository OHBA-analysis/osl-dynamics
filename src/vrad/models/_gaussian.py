"""Model class for a generative model with Gaussian observations.

"""
import logging
from operator import lt

import numpy as np
from tensorflow import zeros
from tensorflow.keras import Model, layers
from tensorflow.nn import softmax, softplus
from tqdm import trange
from vrad.inference.functions import (
    cholesky_factor,
    cholesky_factor_to_full_matrix,
    trace_normalize,
)
from vrad.models import BaseModel
from vrad.models.layers import (
    InferenceRNNLayers,
    KLDivergenceLayer,
    LogLikelihoodLayer,
    MeansCovsLayer,
    MixMeansCovsLayer,
    ModelRNNLayers,
    SampleNormalDistributionLayer,
    TrainableVariablesLayer,
)
from vrad.utils.misc import check_arguments

_logger = logging.getLogger("VRAD")


class RNNGaussian(BaseModel):
    """Inference RNN and generative model with Gaussian observations."""

    def __init__(
        self,
        n_states: int,
        n_channels: int,
        sequence_length: int,
        learn_means: bool,
        learn_covariances: bool,
        n_layers_inference: int,
        n_layers_model: int,
        n_units_inference: int,
        n_units_model: int,
        dropout_rate_inference: float,
        dropout_rate_model: float,
        alpha_xform: str,
        learn_alpha_scaling: bool,
        normalize_covariances: bool,
        do_annealing: bool,
        annealing_sharpness: float,
        n_epochs_annealing: int,
        learning_rate: float = 0.01,
        multi_gpu: bool = False,
        strategy: str = None,
        initial_means: np.ndarray = None,
        initial_covariances: np.ndarray = None,
    ):
        # Parameters related to the observation model
        self.learn_means = learn_means
        self.learn_covariances = learn_covariances
        self.initial_means = initial_means
        self.initial_covariances = initial_covariances
        self.alpha_xform = alpha_xform
        self.learn_alpha_scaling = learn_alpha_scaling
        self.normalize_covariances = normalize_covariances

        # Initialise the model base class
        # This will build and compile the keras model
        super().__init__(
            n_states=n_states,
            n_channels=n_channels,
            sequence_length=sequence_length,
            n_layers_inference=n_layers_inference,
            n_layers_model=n_layers_model,
            n_units_inference=n_units_inference,
            n_units_model=n_units_model,
            dropout_rate_inference=dropout_rate_inference,
            dropout_rate_model=dropout_rate_model,
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
            n_layers_inference=self.n_layers_inference,
            n_layers_model=self.n_layers_model,
            n_units_inference=self.n_units_inference,
            n_units_model=self.n_units_model,
            dropout_rate_inference=self.dropout_rate_inference,
            dropout_rate_model=self.dropout_rate_model,
            learn_means=self.learn_means,
            learn_covariances=self.learn_covariances,
            initial_means=self.initial_means,
            initial_covariances=self.initial_covariances,
            alpha_xform=self.alpha_xform,
            learn_alpha_scaling=self.learn_alpha_scaling,
            normalize_covariances=self.normalize_covariances,
        )

    def predict(self, *args, **kwargs):
        """Wrapper for the standard keras predict method.

        Returns a dictionary with labels for each prediction.
        """
        predictions = self.model.predict(*args, *kwargs)
        return_names = [
            "ll_loss",
            "kl_loss",
            "theta_t",
            "m_theta_t",
            "log_s2_theta_t",
            "mu_theta_jt",
            "log_sigma2_theta_j",
        ]
        predictions_dict = dict(zip(return_names, predictions))
        return predictions_dict

    def predict_states(self, inputs, *args, **kwargs):
        """Return the probability for each state at each time point."""
        inputs = self._make_dataset(inputs)
        outputs = []
        for dataset in inputs:
            m_theta_t = self.predict(dataset, *args, **kwargs)["m_theta_t"]
            m_theta_t = np.concatenate(m_theta_t)
            if self.alpha_xform == "softmax":
                alpha = softmax(m_theta_t).numpy()
            elif self.alpha_xform == "softplus":
                alpha = softplus(m_theta_t).numpy()
            outputs.append(alpha)
        return outputs

    def free_energy(self, dataset, return_all=False):
        """Calculates the variational free energy of a model."""
        predictions = self.predict(dataset)
        ll_loss = np.mean(predictions["ll_loss"])
        kl_loss = np.mean(predictions["kl_loss"])
        if return_all:
            return ll_loss + kl_loss, ll_loss, kl_loss
        else:
            return ll_loss + kl_loss

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
        self.fit(*args, **kwargs, no_annealing=True)

        # Make means and covariances trainable again and compile
        means_covs_layer.trainable = True
        self.compile()

    def get_means_covariances(self):
        """Get the means and covariances of each state."""

        # Get the means and covariances from the MeansCovsLayer
        means_covs_layer = self.model.get_layer("means_covs")
        means = means_covs_layer.means.numpy()
        cholesky_covariances = means_covs_layer.cholesky_covariances.numpy()
        covariances = cholesky_factor_to_full_matrix(cholesky_covariances)

        # Normalise covariances
        if self.normalize_covariances:
            covariances = trace_normalize(covariances)

        # Apply alpha scaling
        alpha_scaling = self.get_alpha_scaling()
        means *= alpha_scaling.reshape(-1, 1)
        covariances *= alpha_scaling.reshape(-1, 1, 1)

        return means, covariances

    def set_means_covariances(self, means=None, covariances=None):
        """Set the means and covariances of each state."""
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
        """Get the alpha scaling of each state."""
        mix_means_covs_layer = self.model.get_layer("mix_means_covs")
        alpha_scaling = mix_means_covs_layer.alpha_scaling.numpy()
        alpha_scaling = softplus(alpha_scaling).numpy()
        return alpha_scaling

    def sample_state_time_course(self, n_samples=1, sequence_length=None):
        """Uses the model RNN to sample a state time course."""
        if sequence_length is None:
            sequence_length = self.sequence_length

        # Get layers
        model_rnn_layer = self.model.get_layer("model_rnn")
        mu_theta_jt_layer = self.model.get_layer("mu_theta_jt")

        # Calculate the standard deviation of the probability distribution function
        # This has been learnt for each state during training
        log_sigma_theta_j = self.model.get_layer("log_sigma_theta_j").get_weights()[0]
        sigma_theta_j = np.exp(log_sigma_theta_j)

        # State time course and sequence of latent states
        sampled_stc = np.zeros([n_samples, self.n_states])
        theta_t = np.zeros([sequence_length, self.n_states], dtype=np.float32)

        # Normally distributed random numbers used to sample the state time course
        epsilon = np.random.normal(0, 1, [n_samples + 1, self.n_states])

        # Randomly select the first theta_t assuming zero means
        theta_t[-1] = sigma_theta_j * epsilon[-1]

        # Get the alpha scaling so we can calculate alpha_t from theta_t
        alpha_scaling = self.get_alpha_scaling()

        # Sample state time course
        for i in trange(n_samples, desc="Sampling state time course"):

            # If there are leading zeros we trim theta_t so that we don't pass the zeros
            trimmed_theta_t = theta_t[~np.all(theta_t == 0, axis=1)][np.newaxis, :, :]

            # Predict the probability distribution function for theta_t one time step
            # in the future, p(theta_t|theta_<t) ~ N(mu_theta_jt, sigma_theta_j)
            model_rnn = model_rnn_layer(trimmed_theta_t)
            mu_theta_jt = mu_theta_jt_layer(model_rnn)[0, -1]

            # Shift theta_t one time step to the left
            theta_t = np.roll(theta_t, -1, axis=0)

            # Sample from the probability distribution function
            theta_t[-1] = mu_theta_jt + sigma_theta_j * epsilon[i]

            # Calculate the state probabilities
            alpha_t = softmax(theta_t[-1]) * alpha_scaling

            # Hard classify the state time course
            sampled_stc[i, np.argmax(alpha_t)] = 1

        return sampled_stc


def _model_structure(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    n_layers_inference: int,
    n_layers_model: int,
    n_units_inference: int,
    n_units_model: int,
    dropout_rate_inference: float,
    dropout_rate_model: float,
    learn_means: bool,
    learn_covariances: bool,
    initial_means: np.ndarray,
    initial_covariances: np.ndarray,
    alpha_xform: str,
    learn_alpha_scaling: bool,
    normalize_covariances: bool,
):
    # Layer for input
    inputs = layers.Input(shape=(sequence_length, n_channels), name="data")

    # Inference RNN:
    # - Learns q(theta_t) ~ N(m_theta_t, s^2_theta_t), where
    #     - m_theta_t     ~ affine(RNN(inputs_<=t))
    #     - log_s_theta_t ~ affine(RNN(inputs_<=t))

    # Definition of layers
    input_normalisation_layer = layers.LayerNormalization()
    inference_input_dropout_layer = layers.Dropout(dropout_rate_inference)
    inference_output_layers = InferenceRNNLayers(
        n_layers_inference,
        n_units_inference,
        dropout_rate_inference,
        name="inference_rnn",
    )
    m_theta_t_layer = layers.Dense(n_states, activation="linear", name="m_theta_t")
    log_s_theta_t_layer = layers.Dense(
        n_states, activation="linear", name="log_s_theta_t",
    )

    # Layer to sample theta_t from q(theta_t)
    theta_t_layer = SampleNormalDistributionLayer(name="theta_t")

    # Inference RNN data flow
    inputs_norm = input_normalisation_layer(inputs)
    inputs_norm_dropout = inference_input_dropout_layer(inputs_norm)
    inference_output = inference_output_layers(inputs_norm_dropout)
    m_theta_t = m_theta_t_layer(inference_output)
    log_s_theta_t = log_s_theta_t_layer(inference_output)
    theta_t = theta_t_layer([m_theta_t, log_s_theta_t])

    # Model RNN:
    # - Learns p(theta_t|theta_<t) ~ N(mu_theta_jt, sigma^2_theta_j), where
    #     - mu_theta_jt       ~ affine(RNN(theta_<t))
    #     - log_sigma_theta_j = trainable constant

    # Definition of layers
    model_input_dropout_layer = layers.Dropout(dropout_rate_model)
    model_output_layers = ModelRNNLayers(
        n_layers_model, n_units_model, dropout_rate_model, name="model_rnn",
    )
    mu_theta_jt_layer = layers.Dense(n_states, activation="linear", name="mu_theta_jt")
    log_sigma_theta_j_layer = TrainableVariablesLayer(
        n_states, name="log_sigma_theta_j"
    )

    # Layers for the means and covariances for the observation model of each state
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
        n_states, n_channels, alpha_xform, learn_alpha_scaling, name="mix_means_covs"
    )

    # Layers to calculate the negative of the log likelihood and KL divergence
    ll_loss_layer = LogLikelihoodLayer(name="ll")
    kl_loss_layer = KLDivergenceLayer(name="kl")

    # Model RNN data flow
    model_input_dropout = model_input_dropout_layer(theta_t)
    model_output = model_output_layers(model_input_dropout)
    mu_theta_jt = mu_theta_jt_layer(model_output)
    log_sigma_theta_j = log_sigma_theta_j_layer(inputs)  # inputs not used
    mu_j, D_j = means_covs_layer(inputs)  # inputs not used
    m_t, C_t = mix_means_covs_layer([theta_t, mu_j, D_j])
    ll_loss = ll_loss_layer([inputs, m_t, C_t])
    kl_loss = kl_loss_layer([m_theta_t, log_s_theta_t, mu_theta_jt, log_sigma_theta_j])

    outputs = [
        ll_loss,
        kl_loss,
        theta_t,
        m_theta_t,
        log_s_theta_t,
        mu_theta_jt,
        log_sigma_theta_j,
    ]

    return Model(inputs=inputs, outputs=outputs)
