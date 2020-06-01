from abc import ABC
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from taser import array_ops


class Callback(ABC):
    def __init__(self):
        pass

    def epoch_end(self, *args, **kwargs):
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

    def plot_loss_dice(self):
        fig, axis = plt.subplots(1)
        axis.plot(self.trainer.loss_history[1:], c='k', label="loss")
        axis_2 = axis.twinx()
        axis_2.plot(self.dice_history, c='tab:red', label="dice")
        plt.legend()
        plt.show()

    def epoch_end(self):
        pred_stc = self.trainer.predict_latent_variable(self.predict_dataset)
        aligned_comp_array, aligned_pred_stc = array_ops.align_arrays(
            self.comparison_array, pred_stc)
        matched_comp_array, matched_pred_stc = array_ops.match_states(
            aligned_comp_array, aligned_pred_stc)
        self.trainer.dice = array_ops.dice_coefficient(matched_comp_array,
                                                       matched_pred_stc)
        self.dice_history.append(self.trainer.dice)