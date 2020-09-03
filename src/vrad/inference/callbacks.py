"""A series of Tensorflow callbacks.

"""

import logging
import os
import time
from abc import ABC
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import tanh
from tensorflow.python.keras import callbacks
from vrad import array_ops
from vrad.inference import metrics, states

_logger = logging.getLogger("VRAD")


class SaveBestCallback(callbacks.ModelCheckpoint):
    def __init__(self, save_best_after, *args, **kwargs):
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


class SavePredictionCallback(callbacks.Callback):
    def __init__(self, prediction_dataset, dir_name, save_frequency=1):
        super().__init__()

        self.prediction_dataset = prediction_dataset
        self.save_frequency = save_frequency

        path = Path(dir_name, time.strftime("%Y%m%d_%H%M%S"))
        path.mkdir(parents=True, exist_ok=True)

        self.pattern = str(path / "predict_epoch_{:03d}")

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_frequency == 0:
            np.save(
                self.pattern.format(epoch),
                self.model.predict_states(self.prediction_dataset),
            )


class AnnealingCallback(callbacks.Callback):
    """Used to update the annealing factor during training."""

    def __init__(self, annealing_factor, annealing_sharpness, n_epochs_annealing):
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


class Callback(ABC):
    def __init__(self):
        pass

    def epoch_end(self, *args, **kwargs):
        pass

    def tqdm_update(self, *args, **kwargs):
        pass


class SaveCallback(Callback):
    def __init__(self, trainer, basename, predict_dataset, frequency: int = 1):
        super().__init__()
        self.trainer = trainer
        self.basename = basename
        self.dataset = predict_dataset
        self.frequency = frequency

        Path(basename).mkdir(parents=True, exist_ok=True)

    def epoch_end(self):
        if self.trainer.epoch % self.frequency == 0:
            fn_length = len(str(self.trainer.n_epochs))
            np.save(
                f"{self.basename}/{str(self.trainer.epoch).zfill(fn_length)}",
                self.trainer.predict_latent_variable(self.dataset),
            )


class ComparisonCallback(Callback):
    def __init__(self, trainer, comparison_array, predict_dataset):
        super().__init__()
        self.trainer = trainer
        self.comparison_array = comparison_array
        self.predict_dataset = predict_dataset
        self.dice_history = []
        self.max_dice = 0
        self.max_dice_stc = None

    def plot_loss_dice(self):
        fig, axis = plt.subplots(1)
        (loss_line,) = axis.plot(self.trainer.loss_history[1:], c="k")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis_2 = axis.twinx()
        (dice_line,) = axis_2.plot(self.dice_history, c="tab:red")
        axis_2.set_ylabel("Dice coefficient")
        plt.legend([loss_line, dice_line], ["loss", "dice"])
        plt.show()

    def epoch_end(self):
        pred_stc = self.trainer.predict_latent_variable(self.predict_dataset)
        aligned_comp_array, aligned_pred_stc = array_ops.align_arrays(
            self.comparison_array, pred_stc
        )
        try:
            matched_comp_array, matched_pred_stc = states.match_states(
                aligned_comp_array, aligned_pred_stc
            )
            self.trainer.dice = metrics.dice_coefficient(
                matched_comp_array, matched_pred_stc
            )
        except ValueError:
            self.trainer.dice = np.nan

        if len(self.dice_history) > 0:
            if self.trainer.dice > max(self.dice_history):
                self.max_dice_stc = matched_pred_stc.copy()

        self.dice_history.append(self.trainer.dice)

    def tqdm_update(self, *args, **kwargs):
        if len(self.dice_history) > 0:
            self.trainer.post_fix.update({"dice": self.dice_history[-1]})


def tensorboard_run_logdir():
    """Creates a directory name to store TensorBoard logs."""
    root_logdir = os.path.join(os.curdir, "logs")
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
