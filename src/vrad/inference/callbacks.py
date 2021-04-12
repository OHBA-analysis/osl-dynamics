"""Custom Tensorflow callbacks.

"""

import tensorflow as tf
from tensorflow import tanh
from tensorflow.python.keras import callbacks


class AlphaTemperatureAnnealingCallback(callbacks.Callback):
    """Callback to update the alpha temperature during training.

    Parameters
    ----------
    initial_alpha_temperature : float
        Alpha temperature for the theta activation function.
    final_alpha_temperature : float
        Final value for the alpha temperature.
    n_epochs_annealing : int
        Number of epochs to apply annealing.
    """

    def __init__(
        self,
        initial_alpha_temperature: float,
        final_alpha_temperature: float,
        n_epochs_annealing: int,
    ):
        super().__init__()
        self.initial_alpha_temperature = initial_alpha_temperature
        self.final_alpha_temperature = final_alpha_temperature
        self.n_epochs_annealing = n_epochs_annealing
        self.alpha_temperature_gradient = (
            final_alpha_temperature - initial_alpha_temperature
        ) / n_epochs_annealing

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
        if epoch < self.n_epochs_annealing:
            new_value = (
                self.initial_alpha_temperature + epoch * self.alpha_temperature_gradient
            )
            alpha_layer.alpha_temperature = new_value
        else:
            alpha_layer.alpha_temperature = self.final_alpha_temperature


class KLAnnealingCallback(callbacks.Callback):
    """Callback to update the KL annealing factor during training.

    The loss function during training is calculated as loss = ll_loss +
    kl_annealing_factor * kl_loss, where the annealing factor is calculated as
    0.5*tanh(annealing_sharpness*epoch - 0.5*n_epochs_annealing)/n_epochs_annealing
    + 0.5.

    Parameters
    ----------
    kl_annealing_factor : tf.Variable
        Annealing factor for the KL term in the loss function.
    annealing_sharpness : float
        Parameter to control the shape of the annealing curve.
    n_epochs_annealing : int
        Number of epochs to apply annealing.
    """

    def __init__(
        self,
        kl_annealing_factor: tf.Variable,
        annealing_sharpness: float,
        n_epochs_annealing: int,
    ):
        super().__init__()
        self.kl_annealing_factor = kl_annealing_factor
        self.annealing_sharpness = annealing_sharpness
        self.n_epochs_annealing = n_epochs_annealing

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
        if epoch < self.n_epochs_annealing:
            new_value = (
                0.5
                * tanh(
                    self.annealing_sharpness
                    * (epoch - 0.5 * self.n_epochs_annealing)
                    / self.n_epochs_annealing
                )
                + 0.5
            )
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
