"""Custom Tensorflow callbacks.

"""

from tensorflow import tanh
from tensorflow.python.keras import callbacks


class AnnealingCallback(callbacks.Callback):
    """Callback to update the annealing factor during training.

    The loss function during training is calculated as loss = ll_loss +
    annealing_factor * kl_loss, where the annealing factor is calculated as
    0.5*tanh(annealing_sharpness*epoch - 0.5*n_epochs_annealing)/n_epochs_annealing
    + 0.5.

    Parameters
    ----------
    annealing_factor : float
        Annealing factor for the KL term in the loss function.
    annealing_sharpness : float
        Parameter to control the shape of the annealing curve.
    n_epochs_annealing : int
        Number of epochs to apply annealing.

    """

    def __init__(
        self,
        annealing_factor: float,
        annealing_sharpness: float,
        n_epochs_annealing: int,
    ):
        super().__init__()
        self.annealing_factor = annealing_factor
        self.annealing_sharpness = annealing_sharpness
        self.n_epochs_annealing = n_epochs_annealing

    def on_epoch_end(self, epoch, logs=None):
        new_value = (
            0.5
            * tanh(
                self.annealing_sharpness
                * (epoch - 0.5 * self.n_epochs_annealing)
                / self.n_epochs_annealing
            )
            + 0.5
        )
        self.annealing_factor.assign(new_value)


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
                save_weights_only=True, monitor="loss", mode="min", save_best_only=True,
            )
        )

        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if epoch >= self.save_best_after:
            if self.save_freq == "epoch":
                self._save_model(epoch=epoch, logs=logs)

    def on_train_end(self, logs=None):
        self.model.load_weights(self.filepath)
