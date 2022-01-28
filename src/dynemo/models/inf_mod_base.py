"""Base class for an inference model.

"""

import logging
from dataclasses import dataclass
from typing import Tuple, Union, Literal

import numpy as np
from tensorflow.keras import optimizers
from dynemo.inference import callbacks, initializers
from dynemo.utils.misc import replace_argument

_logger = logging.getLogger("DyNeMo")


class InferenceModelBase:
    """Base class for an inference model.

    Parameters
    ----------
    config : dynemo.models.Config
    """

    def compile(self, optimizer=None):
        """Wrapper for the standard keras compile method.

        Parameters
        ----------
        optimizer : str or tensorflow.keras.optimizers.Optimizer
            Optimizer to use when compiling. Optional, if None
            a new optimizer is created.
        """
        if optimizer is None:
            optimizer = optimizers.get(
                {
                    "class_name": self.config.optimizer.lower(),
                    "config": {
                        "learning_rate": self.config.learning_rate,
                        "clipnorm": self.config.gradient_clip,
                    },
                }
            )
        self.model.compile(optimizer)

    def fit(
        self,
        *args,
        kl_annealing_callback: bool = None,
        use_tqdm: bool = False,
        tqdm_class=None,
        use_tensorboard: bool = None,
        tensorboard_dir: str = None,
        save_best_after: int = None,
        save_filepath: str = None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Adds callbacks and then trains the model.

        Parameters
        ----------
        kl_annealing_callback : bool
            Should we update the KL annealing factor during training?
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
            # Check config to see if we should do KL annealing
            kl_annealing_callback = self.config.do_kl_annealing

        if kl_annealing_callback:
            kl_annealing_callback = callbacks.KLAnnealingCallback(
                curve=self.config.kl_annealing_curve,
                annealing_sharpness=self.config.kl_annealing_sharpness,
                n_annealing_epochs=self.config.n_kl_annealing_epochs,
            )
            additional_callbacks.append(kl_annealing_callback)

        # Update arguments to pass to the fit method
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
        epochs: int,
        n_init: int,
        **kwargs,
    ):
        """Multi-start training.

        The model is trained for a few epochs with different random initializations
        for weights and the model with the best free energy is kept.

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

    def reset_weights(self, keep: list = None):
        """Reset the model as if you've built a new model.

        Parameters
        ----------
        keep : list of str
            Layer names to NOT reset. Optional.
        """
        initializers.reinitialize_model_weights(self.model, keep)
        if self.config.do_kl_annealing:
            kl_loss_layer = self.model.get_layer("kl_loss")
            kl_loss_layer.annealing_factor.assign(0.0)
        if self.config.do_alpha_temperature_annealing:
            alpha_layer = self.model.get_layer("alpha")
            alpha_layer.temperature.assign(self.config.initial_alpha_temperature)

    def predict(self, *args, **kwargs) -> dict:
        """Wrapper for the standard keras predict method.

        Returns
        -------
        dict
            Dictionary with labels for each prediction.
        """
        predictions = self.model.predict(*args, *kwargs)
        return_names = ["ll_loss", "kl_loss", "alpha"]
        if self.config.multiple_scales:
            return_names += ["beta", "gamma"]
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
        list or np.ndarray
            Mode mixing factors with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        """
        if self.config.multiple_scales:
            return self.get_mode_time_courses(
                inputs, *args, concatenate=concatenate, **kwargs
            )

        inputs = self._make_dataset(inputs)
        outputs = []
        for dataset in inputs:
            alpha = self.predict(dataset, *args, **kwargs)["alpha"]
            alpha = np.concatenate(alpha)
            outputs.append(alpha)

        if concatenate or len(outputs) == 1:
            outputs = np.concatenate(outputs)

        return outputs

    def get_mode_time_courses(
        self, inputs, *args, concatenate: bool = False, **kwargs
    ) -> Tuple[
        Union[list, np.ndarray], Union[list, np.ndarray], Union[list, np.ndarray]
    ]:
        """Get mode time courses.

        This method is used to get mode time courses for the multi-time-scale model.

        Parameters
        ----------
        inputs : tensorflow.data.Dataset
            Prediction dataset.
        concatenate : bool
            Should we concatenate alpha for each subject? Optional, default
            is False.

        Returns
        -------
        list or np.ndarray
            Alpha time course with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        list or np.ndarray
            Beta time course with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        list or np.ndarray
            Gamma time course with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        """
        if not self.config.multiple_scales:
            raise ValueError("Please use get_alpha for a single time scale model.")

        inputs = self._make_dataset(inputs)

        outputs_alpha = []
        outputs_beta = []
        outputs_gamma = []
        for dataset in inputs:
            predictions = self.predict(dataset, *args, **kwargs)

            alpha = predictions["alpha"]
            beta = predictions["beta"]
            gamma = predictions["gamma"]

            alpha = np.concatenate(alpha)
            beta = np.concatenate(beta)
            gamma = np.concatenate(gamma)

            outputs_alpha.append(alpha)
            outputs_beta.append(beta)
            outputs_gamma.append(gamma)

        if concatenate or len(outputs_alpha) == 1:
            outputs_alpha = np.concatenate(outputs_alpha)
            outputs_beta = np.concatenate(outputs_beta)
            outputs_gamma = np.concatenate(outputs_gamma)

        return outputs_alpha, outputs_beta, outputs_gamma

    def losses(self, dataset) -> Tuple[float, float]:
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
            ll_loss = np.mean([np.mean(p["ll_loss"]) for p in predictions])
            kl_loss = np.mean([np.mean(p["kl_loss"]) for p in predictions])

        else:
            predictions = self.predict(dataset)
            ll_loss = np.mean(predictions["ll_loss"])
            kl_loss = np.mean(predictions["kl_loss"])

        return ll_loss, kl_loss

    def free_energy(self, dataset) -> float:
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

    def get_alpha_temperature(self) -> float:
        """Alpha temperature used in the model.

        Returns
        -------
        float
            Alpha temperature.
        """
        alpha_layer = self.model.get_layer("alpha")
        temperature = alpha_layer.temperature.numpy()
        return temperature


@dataclass
class InferenceModelConfig:
    """Settings needed for the inference model."""

    # Alpha parameters
    theta_normalization: Literal[None, "batch", "layer"] = None
    alpha_xform: Literal["gumbel-softmax", "softmax", "softplus"] = None
    learn_alpha_temperature: bool = None
    initial_alpha_temperature: float = None

    # KL annealing parameters
    do_kl_annealing: bool = None
    kl_annealing_curve: Literal["linear", "tanh"] = None
    kl_annealing_sharpness: float = None
    n_kl_annealing_epochs: int = None

    # Initialisation parameters
    n_init: int = None
    n_init_epochs: int = None

    def validate_alpha_parameters(self):
        if self.alpha_xform not in ["gumbel-softmax", "softmax", "softplus"]:
            raise ValueError(
                "alpha_xform must be 'gumbel-softmax', 'softmax' or 'softplus'."
            )

        if "softmax" in self.alpha_xform:
            if self.initial_alpha_temperature is None:
                raise ValueError("initial_alpha_temperature must be passed.")

            if self.initial_alpha_temperature <= 0:
                raise ValueError("initial_alpha_temperature must be greater than zero.")

        elif self.alpha_xform == "softplus":
            self.initial_alpha_temperature = 1.0  # not used in the model
            self.learn_alpha_temperature = False

    def validate_kl_annealing_parameters(self):
        if self.do_kl_annealing is None:
            raise ValueError("do_kl_annealing must be passed.")

        if self.do_kl_annealing:
            if self.kl_annealing_curve is None:
                raise ValueError(
                    "If we are performing KL annealing, "
                    "kl_annealing_curve must be passed."
                )

            if self.kl_annealing_curve not in ["linear", "tanh"]:
                raise ValueError("KL annealing curve must be 'linear' or 'tanh'.")

            if self.kl_annealing_curve == "tanh":
                if self.kl_annealing_sharpness is None:
                    raise ValueError(
                        "kl_annealing_sharpness must be passed if "
                        + "kl_annealing_curve='tanh'."
                    )

                if self.kl_annealing_sharpness < 0:
                    raise ValueError("KL annealing sharpness must be positive.")

            if self.n_kl_annealing_epochs is None:
                raise ValueError(
                    "If we are performing KL annealing, n_kl_annealing_epochs must be "
                    + "passed."
                )

            if self.n_kl_annealing_epochs < 1:
                raise ValueError(
                    "Number of KL annealing epochs must be greater than zero."
                )

    def validate_initialization_parameters(self):
        if self.n_init is not None:
            if self.n_init < 1:
                raise ValueError("n_init must be one or greater.")

            if self.n_init_epochs is None:
                raise ValueError(
                    "If we are initializing the model, n_init_epochs must be passed."
                )

            if self.n_init_epochs < 1:
                raise ValueError("n_init_epochs must be one or greater.")
