"""Custom Tensorflow callbacks.

"""

import numpy as np
import tensorflow as tf
from tensorflow import tanh
from tensorflow.python.keras import callbacks
from dynemo import inference


class AlphaTemperatureAnnealingCallback(callbacks.Callback):
    """Callback to update the alpha temperature during training.

    Parameters
    ----------
    initial_alpha_temperature : float
        Alpha temperature for the theta activation function.
    final_alpha_temperature : float
        Final value for the alpha temperature.
    n_annealing_epochs : int
        Number of epochs to apply annealing.
    """

    def __init__(
        self,
        initial_alpha_temperature: float,
        final_alpha_temperature: float,
        n_annealing_epochs: int,
    ):
        super().__init__()
        self.initial_alpha_temperature = initial_alpha_temperature
        self.final_alpha_temperature = final_alpha_temperature
        self.n_annealing_epochs = n_annealing_epochs
        self.alpha_temperature_gradient = (
            final_alpha_temperature - initial_alpha_temperature
        ) / n_annealing_epochs

    def on_epoch_end(self, epoch, logs=None):
        """Action to perform at the end of an epoch.

        Parameters
        ---------
        epochs : int
            Integer, index of epoch.
        logs : dict
            Results for this training epoch, and for the validation epoch if
            validation is performed.
        """
        alpha_layer = self.model.get_layer("alpha")
        epoch += 1  # epoch goes from 0 to n_epochs - 1, so we add 1
        if epoch < self.n_annealing_epochs:
            new_value = (
                self.initial_alpha_temperature + epoch * self.alpha_temperature_gradient
            )
            alpha_layer.alpha_temperature.assign(new_value)
        else:
            alpha_layer.alpha_temperature.assign(self.final_alpha_temperature)


class DiceCoefficientCallback(callbacks.Callback):
    """Callback to calculate a dice coefficient during training."""

    def __init__(
        self,
        prediction_dataset: tf.data.Dataset,
        ground_truth_mode_time_course: np.ndarray,
    ):
        super().__init__()
        self.prediction_dataset = prediction_dataset
        self.gtstc = ground_truth_mode_time_course
        self.n_modes = ground_truth_mode_time_course.shape[-1]

    def on_epoch_end(self, epoch, logs=None):
        """Action to perform at the end of an epoch.

        Parameters
        ---------
        epochs : int
            Integer, index of epoch.
        logs : dict
            Results for this training epoch, and for the validation epoch if
            validation is performed.
        """
        [_, _, alpha] = self.model.predict(self.prediction_dataset)
        pstc = inference.modes.time_courses(
            alpha, concatenate=True, n_modes=self.n_modes
        )
        pstc, gtstc = inference.modes.match_modes(pstc, self.gtstc)
        dice = inference.metrics.dice_coefficient(pstc, gtstc)
        logs["dice"] = dice
        print(f" - dice: {dice}", end="")


class KLAnnealingCallback(callbacks.Callback):
    """Callback to update the KL annealing factor during training.

    Parameters
    ----------
    kl_annealing_factor : tf.Variable
        Annealing factor for the KL term in the loss function.
    curve : str
        Shape of the annealing curve. Either 'linear' or 'tanh'.
    annealing_sharpness : float
        Parameter to control the shape of the annealing curve.
    n_annealing_epochs : int
        Number of epochs to apply annealing.
    n_cycles : int
        Number of times to perform KL annealing with n_annealing_epochs.
    """

    def __init__(
        self,
        kl_annealing_factor: tf.Variable,
        curve: str,
        annealing_sharpness: float,
        n_annealing_epochs: int,
        n_cycles: int,
    ):
        if curve not in ["linear", "tanh"]:
            raise NotImplementedError(curve)

        super().__init__()
        self.kl_annealing_factor = kl_annealing_factor
        self.curve = curve
        self.annealing_sharpness = annealing_sharpness
        self.n_annealing_epochs = n_annealing_epochs
        self.n_cycles = n_cycles
        self.n_epochs_one_cycle = n_annealing_epochs // n_cycles

    def on_epoch_end(self, epoch, logs=None):
        """Action to perform at the end of an epoch.

        Parameters
        ---------
        epochs : int
            Integer, index of epoch.
        logs : dict
            Results for this training epoch, and for the validation epoch if
            validation is performed.
        """
        epoch += 1  # epoch goes from 0 to n_epochs - 1, so we add 1
        if epoch < self.n_annealing_epochs:
            epoch = epoch % self.n_epochs_one_cycle
            if self.curve == "tanh":
                new_value = (
                    0.5
                    * tanh(
                        self.annealing_sharpness
                        * (epoch - 0.5 * self.n_epochs_one_cycle)
                        / self.n_epochs_one_cycle
                    )
                    + 0.5
                )
            elif self.curve == "linear":
                new_value = epoch / self.n_epochs_one_cycle
            self.kl_annealing_factor.assign(new_value)
        else:
            self.kl_annealing_factor.assign(1.0)


class SaveBestCallback(callbacks.ModelCheckpoint):
    """Callback to save the best model.

    The best model is determined as the model with the lowest loss.

    Parameters
    ----------
    save_best_after : int
        Epoch number after which to save the best model.
    """

    def __init__(self, save_best_after: int, *args, **kwargs):
        self.save_best_after = save_best_after

        kwargs.update(
            dict(
                save_weights_only=True,
                monitor="loss",
                mode="min",
                save_best_only=True,
            )
        )

        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        """Action to perform at the end of an epoch.

        Parameters
        ---------
        epochs : int
            Integer, index of epoch.
        logs : dict
            Results for this training epoch, and for the validation epoch if
            validation is performed.
        """
        self.epochs_since_last_save += 1
        if epoch >= self.save_best_after:
            if self.save_freq == "epoch":
                self._save_model(epoch=epoch, logs=logs)

    def on_train_end(self, logs=None):
        """Action to perform at the end of training.

        Parameters
        ----------
        logs : dict
            Results for this training epoch, and for the validation epoch if
            validation is performed.
        """
        self.model.load_weights(self.filepath)
