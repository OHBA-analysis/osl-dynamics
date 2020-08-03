"""Model class for a generative model with Gaussian observations.

"""

import numpy as np
from tensorflow.keras import Model, layers
from tensorflow.python import zeros
from vrad.inference.functions import cholesky_factor, cholesky_factor_to_full_matrix
from vrad.models import BaseModel
from vrad.models.layers import (
    InferenceRNNLayers,
    KLDivergenceLayer,
    LogLikelihoodLayer,
    MixMeansCovsLayer,
    ModelRNNLayers,
    MultivariateNormalLayer,
    ReparameterizationLayer,
    TrainableVariablesLayer,
)


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
        n_epochs_burnin: int = 0,
        learning_rate: float = 0.01,
        clip_normalization: float = None,
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
            n_epochs_burnin=n_epochs_burnin,
            learning_rate=learning_rate,
            clip_normalization=clip_normalization,
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

    def get_means_covariances(self):
        """Get the means and covariances of each state"""
        mvn_layer = self.model.get_layer("mvn")
        means = mvn_layer.means.numpy()
        cholesky_covariances = mvn_layer.cholesky_covariances.numpy()
        covariances = cholesky_factor_to_full_matrix(cholesky_covariances)
        return means, covariances

    def set_means_covariances(self, means=None, covariances=None):
        """Set the means and covariances of each state."""
        mvn_layer = self.model.get_layer("mvn")
        layer_weights = mvn_layer.get_weights()

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
        mvn_layer.set_weights(layer_weights)

    def get_alpha_scaling(self):
        """Get the alpha scaling of each state."""
        mix_means_covs_layer = self.model.get_layer("mix_means_covs")
        alpha_scaling = mix_means_covs_layer.alpha_scaling.numpy()
        return alpha_scaling


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

    # Inference RNN
    # - q(theta_t)     ~ N(m_theta_t, s2_theta_t)
    # - m_theta_t      ~ affine(RNN(Y_<=t))
    # - log_s2_theta_t ~ affine(RNN(Y_<=t))

    # Definition of layers
    input_normalisation_layer = layers.LayerNormalization()
    inference_input_dropout_layer = layers.Dropout(dropout_rate_inference)
    inference_output_layers = InferenceRNNLayers(
        n_layers_inference, n_units_inference, dropout_rate_inference
    )
    m_theta_t_layer = layers.Dense(n_states, activation="linear", name="m_theta_t")
    log_s2_theta_t_layer = layers.Dense(
        n_states, activation="linear", name="log_s2_theta_t"
    )

    # Layer to generate a sample from q(theta_t) ~ N(m_theta_t, log_s2_theta_t) via the
    # reparameterisation trick
    theta_t_layer = ReparameterizationLayer(name="theta_t")

    # Inference RNN data flow
    inputs_norm = input_normalisation_layer(inputs)
    inputs_norm_dropout = inference_input_dropout_layer(inputs_norm)
    inference_output = inference_output_layers(inputs_norm_dropout)
    m_theta_t = m_theta_t_layer(inference_output)
    log_s2_theta_t = log_s2_theta_t_layer(inference_output)
    theta_t = theta_t_layer([m_theta_t, log_s2_theta_t])

    # Model RNN
    # - p(theta_t|theta_<t) ~ N(mu_theta_jt, sigma2_theta_j)
    # - mu_theta_jt         ~ affine(RNN(theta_<t))
    # - log_sigma2_theta_j  = trainable constant

    # Definition of layers
    model_input_dropout_layer = layers.Dropout(dropout_rate_model)
    model_output_layers = ModelRNNLayers(
        n_layers_model, n_units_model, dropout_rate_model
    )
    mu_theta_jt_layer = layers.Dense(n_states, activation="linear", name="mu_theta_jt")
    log_sigma2_theta_j_layer = TrainableVariablesLayer(
        [n_states],
        initial_values=zeros(n_states),
        trainable=True,
        name="log_sigma2_theta_j",
    )

    # Layers for the means and covariances for observation model of each state
    observation_means_covs_layer = MultivariateNormalLayer(
        n_states,
        n_channels,
        learn_means=learn_means,
        learn_covariances=learn_covariances,
        normalize_covariances=normalize_covariances,
        initial_means=initial_means,
        initial_covariances=initial_covariances,
        name="mvn",
    )
    mix_means_covs_layer = MixMeansCovsLayer(
        n_states, n_channels, alpha_xform, learn_alpha_scaling, name="mix_means_covs"
    )

    # Layers to calculate the negative of the log likelihood and KL divergence
    log_likelihood_layer = LogLikelihoodLayer(name="ll")
    kl_loss_layer = KLDivergenceLayer(name="kl")

    # Model RNN data flow
    model_input_dropout = model_input_dropout_layer(theta_t)
    model_output = model_output_layers(model_input_dropout)
    mu_theta_jt = mu_theta_jt_layer(model_output)
    log_sigma2_theta_j = log_sigma2_theta_j_layer(inputs)  # inputs not used
    mu_j, D_j = observation_means_covs_layer(inputs)  # inputs not used
    m_t, C_t = mix_means_covs_layer([theta_t, mu_j, D_j])
    ll_loss = log_likelihood_layer([inputs, m_t, C_t])
    kl_loss = kl_loss_layer(
        [m_theta_t, log_s2_theta_t, mu_theta_jt, log_sigma2_theta_j]
    )

    outputs = [
        ll_loss,
        kl_loss,
        theta_t,
        m_theta_t,
        log_s2_theta_t,
        mu_theta_jt,
        log_sigma2_theta_j,
    ]

    return Model(inputs=inputs, outputs=outputs)
