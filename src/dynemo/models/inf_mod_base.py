"""Base class for an inference model.

"""

import logging
from typing import Tuple, Union

import numpy as np
from tensorflow import Variable
from tensorflow.keras import optimizers
from dynemo.inference import callbacks, initializers, losses
from dynemo.utils.misc import replace_argument

_logger = logging.getLogger("DyNeMo")


class InferenceModelBase:
    """Base class for an inference model.

    Parameters
    ----------
    config : dynemo.models.Config
    """

    def __init__(self, config):

        # KL annealing
        self.kl_annealing_factor = (
            Variable(0.0) if config.do_kl_annealing else Variable(1.0)
        )

    def compile(self, optimizer=None):
        """Wrapper for the standard keras compile method."""

        # Loss function
        ll_loss = losses.ModelOutputLoss()
        kl_loss = losses.ModelOutputLoss(self.kl_annealing_factor)
        loss = [ll_loss, kl_loss]

        # Optimiser
        if optimizer is None:
            if self.config.optimizer.lower() == "adam":
                optimizer = optimizers.Adam(
                    learning_rate=self.config.learning_rate,
                    clipnorm=self.config.gradient_clip,
                )
            elif self.config.optimizer.lower() == "rmsprop":
                optimizer = optimizers.RMSprop(
                    learning_rate=self.config.learning_rate,
                    clipnorm=self.config.gradient_clip,
                )

        # Compile
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(
        self,
        *args,
        kl_annealing_callback=None,
        alpha_temperature_annealing_callback=None,
        use_tqdm=False,
        tqdm_class=None,
        use_tensorboard=None,
        tensorboard_dir=None,
        save_best_after=None,
        save_filepath=None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Adds callbacks and then trains the model.

        Parameters
        ----------
        kl_annealing_callback : bool
            Should we update the KL annealing factor during training?
        alpha_temperature_annealing_callback : bool
            Should we update the alpha temperature annealing factor during training?
        use_tqdm : bool
            Should we use a tqdm progress bar instead of the usual output from
            tensorflow.
        tqdm_class : tqdm
            Class for the tqdm progress bar.
        use_tensorboard : bool
            Should we use TensorBoard?
        tensorboard_dir : str
            Path to the location to save the TensorBoard log files.
        save_best_after : int
            Epoch number after which we should save the best model. The best model is
            that which achieves the lowest loss.
        save_filepath : str
            Path to save the best model to.

        Returns
        -------
        history
            The training history.
        """
        if use_tqdm:
            args, kwargs = replace_argument(self.model.fit, "verbose", 0, args, kwargs)

        additional_callbacks = []

        if kl_annealing_callback is None:
            kl_annealing_callback = self.config.do_kl_annealing

        if kl_annealing_callback:
            kl_annealing_callback = callbacks.KLAnnealingCallback(
                kl_annealing_factor=self.kl_annealing_factor,
                curve=self.config.kl_annealing_curve,
                annealing_sharpness=self.config.kl_annealing_sharpness,
                n_annealing_epochs=self.config.n_kl_annealing_epochs,
                n_cycles=self.config.n_kl_annealing_cycles,
            )
            additional_callbacks.append(kl_annealing_callback)

        if alpha_temperature_annealing_callback is None:
            alpha_temperature_annealing_callback = (
                self.config.do_alpha_temperature_annealing
            )

        if alpha_temperature_annealing_callback:
            alpha_temperature_annealing_callback = (
                callbacks.AlphaTemperatureAnnealingCallback(
                    initial_alpha_temperature=self.config.initial_alpha_temperature,
                    final_alpha_temperature=self.config.final_alpha_temperature,
                    n_annealing_epochs=self.config.n_alpha_temperature_annealing_epochs,
                )
            )
            additional_callbacks.append(alpha_temperature_annealing_callback)

        args, kwargs = replace_argument(
            func=self.model.fit,
            name="callbacks",
            item=self.create_callbacks(
                use_tqdm,
                tqdm_class,
                use_tensorboard,
                tensorboard_dir,
                save_best_after,
                save_filepath,
                additional_callbacks,
            ),
            args=args,
            kwargs=kwargs,
            append=True,
        )

        return self.model.fit(*args, **kwargs)

    def initialize(
        self,
        training_dataset,
        epochs,
        n_init,
        **kwargs,
    ):
        """Initialize the means and covariances.

        The model is trained for a few epochs and the model with the best
        free energy is chosen.

        Parameters
        ----------
        training_dataset : tensorflow.data.Dataset
            Dataset to use for training.
        epochs : int
            Number of epochs to train the model.
        n_init : int
            Number of initializations.

        Returns
        -------
        history
            The training history of the best initialization.
        """
        if n_init is None or n_init == 0:
            _logger.warning(
                "Number of initializations was set to zero. "
                + "Skipping initialization."
            )
            return

        # Pick the initialization with the lowest free energy
        best_free_energy = np.Inf
        for n in range(n_init):
            print(f"Initialization {n}")
            self.reset_weights()
            self.compile()
            history = self.fit(
                training_dataset,
                epochs=epochs,
                **kwargs,
            )
            free_energy = history.history["loss"][-1]
            if free_energy < best_free_energy:
                best_initialization = n
                best_free_energy = free_energy
                best_weights = self.model.get_weights()
                best_optimizer = self.model.optimizer
                best_history = history

        print(f"Using initialization {best_initialization}")
        self.reset_weights()
        self.model.set_weights(best_weights)
        self.compile(optimizer=best_optimizer)

        return best_history

    def reset_weights(self, keep=None):
        """Reset the model as if you've built a new model."""
        initializers.reinitialize_model_weights(self.model, keep)
        if self.config.do_kl_annealing:
            self.kl_annealing_factor.assign(0.0)
        if self.config.do_alpha_temperature_annealing:
            alpha_layer = self.model.get_layer("alpha")
            alpha_layer.alpha_temperature.assign(self.config.initial_alpha_temperature)

    def predict(self, *args, **kwargs) -> dict:
        """Wrapper for the standard keras predict method.

        Returns
        -------
        dict
            Dictionary with labels for each prediction.
        """
        if not self.config.multiple_scale:
            predictions = self.model.predict(*args, *kwargs)
            return_names = ["ll_loss", "kl_loss", "alpha"]
            predictions_dict = dict(zip(return_names, predictions))

        else:
            predictions = self.model.predict(*args, *kwargs)
            return_names = ["ll_loss", "kl_loss", "alpha", "beta", "gamma"]
            predictions_dict = dict(zip(return_names, predictions))
        return predictions_dict

    def get_alpha(
        self, inputs, *args, concatenate: bool = False, **kwargs
    ) -> Union[list, np.ndarray]:
        """Mode mixing factors, alpha.

        Parameters
        ----------
        inputs : tensorflow.data.Dataset
            Prediction dataset.
        concatenate : bool
            Should we concatenate alpha for each subject? Optional, default
            is False.

        Returns
        -------
        np.ndarray
            Mode mixing factors with shape (n_subjects, n_samples,
            n_modes) or (n_samples, n_modes).
        """
        inputs = self._make_dataset(inputs)
        outputs_alpha = []
        outputs_beta = []
        outputs_gamma = []
        for dataset in inputs:
            predictions_dict = self.predict(dataset, *args, **kwargs)
            alpha = predictions_dict["alpha"]
            alpha = np.concatenate(alpha)
            outputs_alpha.append(alpha)

            if self.config.multiple_scale:
                beta = predictions_dict["beta"]
                gamma = predictions_dict["gamma"]
                beta = np.concatenate(beta)
                gamma = np.concatenate(gamma)
                outputs_beta.append(beta)
                outputs_gamma.append(gamma)

                if concatenate or len(outputs_beta) == 1:
                    outputs_beta = np.concatenate(outputs_beta)
        
                if concatenate or len(outputs_gamma) == 1:
                    outputs_gamma = np.concatenate(outputs_gamma)

            if concatenate or len(outputs_alpha) == 1:
                outputs_alpha = np.concatenate(outputs_alpha)
        

        if self.config.multiple_scale:
            return [outputs_alpha, outputs_beta, outputs_gamma]
        return outputs_alpha

    def losses(self, dataset, return_sum: bool = False) -> Tuple[float, float]:
        """Calculates the log-likelihood and KL loss for a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate losses for.
        return_sum : bool
            Should we return the loss for each batch summed? Otherwise we return
            the mean. Optional, default is False.

        Returns
        -------
        ll_loss : float
            Negative log-likelihood loss.
        kl_loss : float
            KL divergence loss.
        """
        if return_sum:
            mean_or_sum = np.sum
        else:
            mean_or_sum = np.mean
        if isinstance(dataset, list):
            predictions = [self.predict(subject) for subject in dataset]
            ll_loss = mean_or_sum([mean_or_sum(p["ll_loss"]) for p in predictions])
            kl_loss = mean_or_sum([mean_or_sum(p["kl_loss"]) for p in predictions])
        else:
            predictions = self.predict(dataset)
            ll_loss = mean_or_sum(predictions["ll_loss"])
            kl_loss = mean_or_sum(predictions["kl_loss"])
        return ll_loss, kl_loss

    def free_energy(self, dataset, return_sum: bool = False) -> float:
        """Calculates the variational free energy of a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate the variational free energy for.
        return_sum : bool
            Should we return the free energy for each batch summed? Otherwise
            we return the mean. Optional, default is False.

        Returns
        -------
        float
            Variational free energy for the dataset.
        """
        ll_loss, kl_loss = self.losses(dataset, return_sum=return_sum)
        free_energy = ll_loss + kl_loss
        return free_energy

    def get_alpha_temperature(self) -> float:
        """Alpha temperature used in the model.

        Returns
        -------
        float
            Alpha temperature.
        """
        alpha_layer = self.model.get_layer("alpha")
        alpha_temperature = alpha_layer.alpha_temperature.numpy()
        return alpha_temperature
