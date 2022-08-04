"""Base classes for models with inference.

"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from osl_dynamics.models.mod_base import ModelBase
from osl_dynamics.inference import callbacks, initializers
from osl_dynamics.utils.misc import replace_argument


@dataclass
class VariationalInferenceModelConfig:
    """Settings needed for the inference model."""

    # Alpha parameters
    theta_normalization: Literal[None, "batch", "layer"] = None
    learn_alpha_temperature: bool = None
    initial_alpha_temperature: float = None

    # KL annealing parameters
    do_kl_annealing: bool = False
    kl_annealing_curve: Literal["linear", "tanh"] = None
    kl_annealing_sharpness: float = None
    n_kl_annealing_epochs: int = None

    def validate_alpha_parameters(self):
        if self.initial_alpha_temperature is None:
            raise ValueError("initial_alpha_temperature must be passed.")

        if self.initial_alpha_temperature <= 0:
            raise ValueError("initial_alpha_temperature must be greater than zero.")

    def validate_kl_annealing_parameters(self):
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
                    "If we are performing KL annealing, "
                    + "n_kl_annealing_epochs must be passed."
                )

            if self.n_kl_annealing_epochs < 1:
                raise ValueError(
                    "Number of KL annealing epochs must be greater than zero."
                )


class VariationalInferenceModelBase(ModelBase):
    """Base class for a variational inference model."""

    def fit(
        self,
        *args,
        kl_annealing_callback=None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Parameters
        ----------
        *args
            Arguments for ModelBase.fit().
        kl_annealing_callback : bool
            Should we update the KL annealing factor during training?
        **kwargs
            Keyword arguments for ModelBase.fit()

        Returns
        -------
        history : history
            The training history.
        """
        if kl_annealing_callback is None:
            # Check config to see if we should do KL annealing
            kl_annealing_callback = self.config.do_kl_annealing

        if kl_annealing_callback:
            kl_annealing_callback = callbacks.KLAnnealingCallback(
                curve=self.config.kl_annealing_curve,
                annealing_sharpness=self.config.kl_annealing_sharpness,
                n_annealing_epochs=self.config.n_kl_annealing_epochs,
            )

            # Update arguments to pass to the fit method
            args, kwargs = replace_argument(
                self.model.fit,
                "callbacks",
                [kl_annealing_callback],
                args,
                kwargs,
                append=True,
            )

        return super().fit(*args, **kwargs)

    def multistart_initialization(
        self,
        training_data,
        n_epochs,
        n_init,
        **kwargs,
    ):
        """Multi-start initialization.

        The model is trained for a few epochs with different random initializations
        for weights and the model with the best free energy is kept.

        Parameters
        ----------
        training_data : tensorflow.data.Dataset or osl_dynamics.data.Data
            Dataset to use for training.
        n_epochs : int
            Number of epochs to train the model.
        n_init : int
            Number of initializations.

        Returns
        -------
        history : history
            The training history of the best initialization.
        """
        if n_init is None or n_init == 0:
            print(
                "Number of initializations was set to zero. "
                + "Skipping initialization."
            )
            return

        # Pick the initialization with the lowest free energy
        best_free_energy = np.Inf
        for n in range(n_init):
            print(f"Initialization {n}")
            self.reset()
            history = self.fit(
                training_data,
                epochs=n_epochs,
                **kwargs,
            )
            free_energy = history.history["loss"][-1]
            if free_energy < best_free_energy:
                best_initialization = n
                best_free_energy = free_energy
                best_weights = self.model.get_weights()
                best_history = history

        print(f"Using initialization {best_initialization}")
        self.reset()
        self.set_weights(best_weights)

        return best_history

    def reset_kl_annealing_factor(self):
        """Sets the KL annealing factor to zero.

        This method assumes there is a keras layer named 'kl_loss' in the model.
        """
        kl_loss_layer = self.model.get_layer("kl_loss")
        kl_loss_layer.annealing_factor.assign(0.0)

    def reset_weights(self, keep=None):
        """Reset the model as if you've built a new model.

        Parameters
        ----------
        keep : list of str
            Layer names to NOT reset.
        """
        initializers.reinitialize_model_weights(self.model, keep)
        if self.config.do_kl_annealing:
            self.reset_kl_annealing_factor()

    def predict(self, *args, **kwargs) -> dict:
        """Wrapper for the standard keras predict method.

        Returns
        -------
        predictions : dict
            Dictionary with labels for each prediction.
        """
        predictions = self.model.predict(*args, *kwargs)
        return_names = ["ll_loss", "kl_loss", "alpha"]
        if self.config.multiple_dynamics:
            return_names.append("gamma")
        predictions_dict = dict(zip(return_names, predictions))

        return predictions_dict

    def get_alpha(self, inputs, concatenate=False):
        """Mode mixing factors, alpha.

        Parameters
        ----------
        inputs : tensorflow.data.Dataset or osl_dynamics.data.Data
            Prediction data.
        concatenate : bool
            Should we concatenate alpha for each subject?

        Returns
        -------
        alpha : list or np.ndarray
            Mode mixing factors with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        """
        if self.config.multiple_dynamics:
            return self.get_mode_time_courses(
                inputs, *args, concatenate=concatenate, **kwargs
            )

        inputs = self.make_dataset(inputs, concatenate=concatenate)

        print("Getting alpha:")
        outputs = []
        for dataset in inputs:
            alpha = self.predict(dataset)["alpha"]
            alpha = np.concatenate(alpha)
            outputs.append(alpha)

        if concatenate or len(outputs) == 1:
            outputs = np.concatenate(outputs)

        return outputs

    def get_mode_time_courses(self, inputs, concatenate=False):
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
            predictions = self.predict(dataset)

            alpha = predictions["alpha"]
            gamma = predictions["gamma"]

            alpha = np.concatenate(alpha)
            gamma = np.concatenate(gamma)

            outputs_alpha.append(alpha)
            outputs_gamma.append(gamma)

        if concatenate or len(outputs_alpha) == 1:
            outputs_alpha = np.concatenate(outputs_alpha)
            outputs_gamma = np.concatenate(outputs_gamma)

        return outputs_alpha, outputs_gamma

    def losses(self, dataset):
        """Calculates the log-likelihood and KL loss for a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset or osl_dynamics.data.Data
            Dataset to calculate losses for.

        Returns
        -------
        ll_loss : float
            Negative log-likelihood loss.
        kl_loss : float
            KL divergence loss.
        """
        dataset = self.make_dataset(dataset, concatenate=True)
        print("Getting losses:")
        predictions = self.predict(dataset)
        ll_loss = np.mean(predictions["ll_loss"])
        kl_loss = np.mean(predictions["kl_loss"])
        return ll_loss, kl_loss

    def free_energy(self, dataset):
        """Calculates the variational free energy of a dataset.

        Note, this method returns a free energy which may have a significantly
        smaller KL loss. This is because during training we sample from the
        posterior, however, when we're evaluating the model, we take the maximum
        a posteriori estimate (posterior mean). This has the effect of giving a
        lower KL loss for a given dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset or osl_dynamics.data.Data.
            Dataset to calculate the variational free energy for.

        Returns
        -------
        free_energy : float
            Variational free energy for the dataset.
        """
        dataset = self.make_dataset(dataset, concatenate=True)
        ll_loss, kl_loss = self.losses(dataset)
        free_energy = ll_loss + kl_loss
        return free_energy
