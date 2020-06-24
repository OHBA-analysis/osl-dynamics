from abc import ABC
from pathlib import Path

import numpy as np
import vrad.inference.metrics
from matplotlib import pyplot as plt
from tensorflow.python import tanh
from tensorflow.python.keras import callbacks
from vrad import array_ops
from vrad.inference.layers import MVNLayer


class AnnealingCallback(callbacks.Callback):
    """Used to update the annealing factor during training."""

    def __init__(self, annealing_factor, annealing_sharpness, n_epochs_annealing):
        super().__init__()
        self.annealing_factor = annealing_factor
        self.annealing_sharpness = annealing_sharpness
        self.n_epochs_annealing = n_epochs_annealing

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
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


class BurninCallback(callbacks.Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.mvn = None

    def on_train_begin(self, logs=None):
        self.mvn = self.model.layers[
            [isinstance(layer, MVNLayer) for layer in self.model.layers].index(True)
        ]

    def on_epoch_begin(self, epoch, logs=None):
        self.mvn.burnin.assign(epoch < self.epochs)


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
            matched_comp_array, matched_pred_stc = array_ops.match_states(
                aligned_comp_array, aligned_pred_stc
            )
            self.trainer.dice = vrad.inference.metrics.dice_coefficient(
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
