"""Inference network and generative model.

"""

import numpy as np
from tensorflow.keras import Model, layers, models, optimizers
from tensorflow.python import Variable, zeros
from tensorflow.python.distribute.distribution_strategy_context import get_strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from vrad.inference.callbacks import AnnealingCallback, BurninCallback
from vrad.inference.functions import cholesky_factor, cholesky_factor_to_full_matrix
from vrad.inference.initializers import reinitialize_model_weights
from vrad.inference.layers import (
    InferenceRNNLayers,
    KLDivergenceLayer,
    LogLikelihoodLayer,
    MixMeansCovsLayer,
    ModelRNNLayers,
    MultivariateNormalLayer,
    ReparameterizationLayer,
    TrainableVariablesLayer,
)
from vrad.utils.misc import listify


class Model:
    """Main model used in V-RAD.

    Contains the inference RNN and generative model.
    Acts as a wrapper for a standard Keras model. The Keras model can be accessed
    through the model.keras attribute.
    """

    def __init__(
        n_states: int,
        n_channels: int,
        sequence_length: int,
        learn_means: bool,
        learn_covariances: bool,
        initial_means: np.ndarray,
        initial_covariances: np.ndarray,
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
        n_epochs_burnin: int,
        learning_rate: float,
        clip_normalization: float,
        multi_gpu: bool = False,
        strategy: str = None,
    ):
        # Number of latent states and dimensionality of the data
        self.n_states = n_states
        self.n_channels = n_channels

        # Model hyperparameters
        self.sequence_length = sequence_length
        self.n_layers_inference = n_layers_inference
        self.n_layers_model = n_layers_model
        self.n_units_inference = n_units_inference
        self.n_units_model = n_units_model
        self.dropout_rate_inference = dropout_rate_inference
        self.dropout_rate_model = dropout_rate_model

        # Parameters related to the means and covariances
        self.learn_means = learn_means
        self.learn_covariances = learn_covariances
        self.initial_means = initial_means
        self.initial_covariances = initial_covariances
        self.alpha_xform = alpha_xform
        self.learn_alpha_scaling = learn_alpha_scaling
        self.normalize_covariances = normalize_covariances

        # KL annealing and burn-in
        self.do_annealing = do_annealing
        self.annealing_factor = Variable(0.0) if do_annealing else Variable(1.0)
        self.annealing_sharpness = annealing_sharpness
        self.n_epochs_annealing = n_epochs_annealing
        self.n_epochs_burnin = n_epochs_burnin

        # Callbacks
        self.burnin_callback = BurninCallback(epochs=self.n_epochs_burnin)
        self.annealing_callback = AnnealingCallback(
            annealing_factor=self.annealing_factor,
            annealing_sharpness=self.annealing_sharpness,
            n_epochs_annealing=self.n_epochs_annealing,
        )

        # Training Parameters
        self.learning_rate = learning_rate
        self.clip_normalization = clip_normalization

        # Stretegy for distributed learning
        if multi_gpu:
            self.strategy = MirroredStrategy()
        elif strategy is None:
            self.strategy = get_strategy()

        # Build the model
        self.build_model()

        # Compile the model
        self.compile()

        def build_model(self):
            """Builds a keras model."""
            self.keras = _model_structure(
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

        def compile(self, optimizer=None):
            """Wrapper for the standard keras compile method.
            
            Sets up the optimiser and loss functions.
            """
            # Setup optimizer
            if optimizer == None:
                optimizer = optimizers.Adam(
                    learning_rate=self.learning_rate, clipnorm=self.clip_normalization,
                )

            # Loss functions
            loss = [
                _ll_loss_fn,
                _create_kl_loss_fn(annealing_factor=self.annealing_factor),
            ]

            # Compile
            self.keras.compile(optimizer=optimizer, loss=loss)

        def fit(self, *args, use_tqdm=False, tqdm_class=None, **kwargs):
            """Wrapper for the standard keras fit method.

            Adds callbacks for KL annealing and burn-in.
            """
            args = list(args)

            # Add annealing, burn-in and tqdm callbacks
            additional_callbacks = [self.annealing_callback, self.burnin_callback]
            if use_tqdm:
                if tqdm_class is not None:
                    tqdm_callback = TqdmCallback(verbose=0, tqdm_class=tqdm_class)
                else:
                    tqdm_callback = TqdmCallback(verbose=0, tqdm_class=tqdm)
                additional_callbacks.append(tqdm_callback)
            if len(args) > 5:
                args[5] = listify(args[5]) + additional_callbacks
            if "callbacks" in kwargs:
                kwargs["callbacks"] = (
                    listify(kwargs["callbacks"]) + additional_callbacks
                )
            else:
                kwargs["callbacks"] = additional_callbacks

            # Train the model
            return self.keras.fit(*args, **kwargs)

        def predict(self, *args, **kwargs):
            """Wrapper for the standard keras predict method.

            Returns a dictionary with labels for each prediction.
            """
            predictions = self.keras.predict(*args, *kwargs)
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

        def predict_states(self, *args, **kwargs):
            """Infers the latent state time course."""
            m_theta_t = self.keras.predict(*args, **kwargs)["m_theta_t"]
            return np.concatenate(m_theta_t)

        def free_energy(self, dataset, return_all=False):
            """Calculates the variational free energy of a model."""
            predictions = self.predict(dataset)
            ll_loss = np.mean(predictions["ll_loss"])
            kl_loss = np.mean(predictions["kl_loss"])
            if return_all:
                return ll_loss + kl_loss, ll_loss, kl_loss
            else:
                return ll_loss + kl_loss

        def reset_model(self):
            """Reset the model as if you've built a new model.

            Resets the model weights, optimizer and annealing factor.
            """
            self.compile()
            reinitialize_model_weights(self.keras)
            if self.do_annealing:
                self.annealing_factor.assign(0.0)

        def initialize_means_covariances(
            self,
            n_initializations,
            n_epochs_initialization,
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
                self.fit(
                    epochs=n_epochs_initialization,
                    verbose=verbose,
                    use_tqdm=use_tqm,
                    tqdm_class=tqdm_class,
                )
                free_energy = self.keras.evaluate(args[0], verbose=0)[0]
                if free_energy < best_free_energy:
                    best_initialization = n
                    best_free_energy = free_energy
                    best_weights = keras.get_weights()
                    best_optimizer = keras.optimizer

            print(f"Using initialization {best_initialization}")
            self.compile(optimizer=best_optimizer)
            self.keras.set_weights(best_weights)
            if self.do_annealing:
                self.annealing_factor.assign(0.0)

        def get_means_covariances(self):
            """Get the means and covariances of each state"""
            mvn_layer = self.keras.get_layer("mvn")
            means = mvn_layer.means.numpy()
            cholesky_covariances = mvn_layer.cholesky_covariances.numpy()
            covariances = cholesky_factor_to_full_matrix(cholesky_covariances)
            return means, covariances

        def set_means_covariances(self, means=None, covariances=None):
            """Set the means and covariances of each state."""
            mvn_layer = self.keras.get_layer("mvn")
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
            mix_means_covs_layer = self.keras.get_layer("mix_means_covs")
            alpha_scaling = mix_means_covs_layer.alpha_scaling.numpy()
            return alpha_scaling

        def save_weights(self, filepath):
            """Save weights of the model."""
            self.keras.save_weights(filepath)

        def load_weights(self, filepath):
            """Load weights of the model from a file."""
            with strategy.scope():
                self.keras.load_weights(filepath)


def _ll_loss_fn(y_true, ll_loss):
    """The first output of the model is the negative log likelihood
       so we just need to return it."""
    return ll_loss


def _create_kl_loss_fn(annealing_factor):
    def _kl_loss_fn(y_true, kl_loss):
        """Second output of the model is the KL divergence loss.
        We multiply with an annealing factor."""
        return annealing_factor * kl_loss

    return _kl_loss_fn


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
