"""Base class for models in V-RAD.

"""
from abc import abstractmethod

import numpy as np
from tensorflow import Variable
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.distribute.distribution_strategy_context import get_strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from vrad.inference.callbacks import (
    AnnealingCallback,
    BurninCallback,
    SaveBestCallback,
    tensorboard_run_logdir,
)
from vrad.inference.initializers import reinitialize_model_weights
from vrad.inference.losses import KullbackLeiblerLoss, LogLikelihoodLoss
from vrad.utils.misc import listify, replace_argument


class BaseModel:
    """Base class for models.

    Acts as a wrapper for a standard Keras model.
    """

    def __init__(
        self,
        n_states: int,
        n_channels: int,
        sequence_length: int,
        n_layers_inference: int,
        n_layers_model: int,
        n_units_inference: int,
        n_units_model: int,
        dropout_rate_inference: float,
        dropout_rate_model: float,
        do_annealing: bool,
        annealing_sharpness: float,
        n_epochs_annealing: int,
        n_epochs_burnin: int,
        learning_rate: float,
        clip_normalization: float,
        multi_gpu: bool,
        strategy: str,
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

        # Strategy for distributed learning
        self.strategy = strategy
        if multi_gpu:
            self.strategy = MirroredStrategy()
        elif strategy is None:
            self.strategy = get_strategy()

        # Build and compile the model
        self.model = None

        with self.strategy.scope():
            self.build_model()
        self.compile()

        self._identifier = np.random.randint(100000)

    # Allow access to the keras model attributes
    def __getattr__(self, attr):
        return getattr(self.model, attr)

    @abstractmethod
    def build_model(self):
        """Build a keras model."""
        pass

    def compile(self, optimizer=None):
        """Wrapper for the standard keras compile method.
        
        Sets up the optimiser and loss functions.
        """
        # Setup optimizer
        if optimizer is None:
            optimizer = optimizers.Adam(
                learning_rate=self.learning_rate, clipnorm=self.clip_normalization,
            )

        # Loss functions
        ll_loss = LogLikelihoodLoss()
        kl_loss = KullbackLeiblerLoss(self.annealing_factor)
        loss = [ll_loss, kl_loss]

        # Compile
        self.model.compile(optimizer=optimizer, loss=loss)

    def create_callbacks(
        self, use_tqdm, tqdm_class, use_tensorboard, tensorboard_dir, save_best_after
    ):

        # Callback for KL annealing
        annealing_callback = AnnealingCallback(
            annealing_factor=self.annealing_factor,
            annealing_sharpness=self.annealing_sharpness,
            n_epochs_annealing=self.n_epochs_annealing,
        )

        # Callback for a burn-in training period
        burnin_callback = BurninCallback(epochs=self.n_epochs_burnin)

        additional_callbacks = [annealing_callback, burnin_callback]

        # Callback to display a progress bar with tqdm
        if use_tqdm:
            if tqdm_class is not None:
                tqdm_callback = TqdmCallback(verbose=0, tqdm_class=tqdm_class)
            else:
                tqdm_callback = TqdmCallback(verbose=0, tqdm_class=tqdm)

            additional_callbacks.append(tqdm_callback)

        # Callback for Tensorboard visulisation
        if use_tensorboard:
            if tensorboard_dir is not None:
                tensorboard_cb = TensorBoard(tensorboard_dir)
            else:
                tensorboard_cb = TensorBoard(tensorboard_run_logdir())

            additional_callbacks.append(tensorboard_cb)

        # Callback to save the best model after a certain number of epochs
        if save_best_after is not None:
            save_best_callback = SaveBestCallback(
                save_best_after=save_best_after,
                filepath=f"/tmp/model_weights/best_{self._identifier}",
            )

            additional_callbacks.append(save_best_callback)

        return additional_callbacks

    def fit(
        self,
        *args,
        use_tqdm=False,
        tqdm_class=None,
        use_tensorboard=None,
        tensorboard_dir=None,
        save_best_after=None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Adds callbacks for KL annealing and burn-in then trains the model.
        """
        if use_tqdm:
            args, kwargs = replace_argument(self.model.fit, "verbose", 0, args, kwargs)

        args, kwargs = replace_argument(
            func=self.model.fit,
            name="callbacks",
            item=self.create_callbacks(
                use_tqdm, tqdm_class, use_tensorboard, tensorboard_dir, save_best_after
            ),
            args=args,
            kwargs=kwargs,
            append=True,
        )

        # Train the model
        return self.model.fit(*args, **kwargs)

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

    def predict_states(self, *args, **kwargs):
        """Infers the latent state time course."""
        m_theta_t = self.predict(*args, **kwargs)["m_theta_t"]
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
        reinitialize_model_weights(self.model)
        if self.do_annealing:
            self.annealing_factor.assign(0.0)

    def load_weights(self, filepath):
        """Load weights of the model from a file."""
        with self.strategy.scope():
            self.model.load_weights(filepath)
