"""Custom Tensorflow callbacks.

"""

import numpy as np
from tensorflow import tanh
from tensorflow.python.keras import callbacks
from osl_dynamics import inference


class DiceCoefficientCallback(callbacks.Callback):
    """Callback to calculate a dice coefficient during training.

    Parameters
    ----------
    prediction_dataset : tensorflow.data.Dataset
        Dataset to use to calculate outputs of the model.
    ground_truth_mode_time_course : np.ndarray
        2D or 3D numpy array containing the ground truth mode time
        course of the training data.
    mode_names : list of str
        Names for the mode time courses.
    """

    def __init__(
        self,
        prediction_dataset,
        ground_truth_mode_time_course,
        mode_names=None,
    ):
        super().__init__()
        self.prediction_dataset = prediction_dataset
        if ground_truth_mode_time_course.ndim == 2:
            # We're training a single time scale model
            self.n_time_courses = 1
            self.gtmtc = ground_truth_mode_time_course[np.newaxis, ...]
        elif ground_truth_mode_time_course.ndim == 3:
            # We're training a multi-time-scale model
            self.n_time_courses = ground_truth_mode_time_course.shape[0]
            self.gtmtc = ground_truth_mode_time_course
        else:
            raise ValueError(
                "A 2D or 3D numpy array must be pass for ground_truth_mode_time_course."
            )
        if mode_names is not None:
            if len(mode_names) != self.n_time_courses:
                raise ValueError(
                    "Mismatch between the number of mode_names and time courses."
                )
        self.mode_names = mode_names
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

        # Predict time courses
        predictions = self.model.predict(self.prediction_dataset, verbose=0)
        tc = predictions[2:]  # first two outputs are losses, rest are time courses
        if len(tc) != self.n_time_courses:
            raise ValueError(
                "Mismatch between number of ground truth and predicted time courses."
            )

        # For each time course calculate the dice with respect to the ground truth
        dices = []
        for i in range(self.n_time_courses):
            pmtc = inference.modes.argmax_time_courses(
                tc[i], concatenate=True, n_modes=self.n_modes
            )
            pmtc, gtmtc = inference.modes.match_modes(pmtc, self.gtmtc[i])
            dice = inference.metrics.dice_coefficient(pmtc, gtmtc)
            dices.append(dice)

        # Add dice to the training history and print to screen
        if self.n_time_courses == 1:
            logs["dice"] = dices[0]
        else:
            for i in range(self.n_time_courses):
                if self.mode_names is not None:
                    key = "dice_" + self.mode_names[i]
                else:
                    key = "dice" + str(i)
                logs[key] = dices[i]


class KLAnnealingCallback(callbacks.Callback):
    """Callback to update the KL annealing factor during training.

    This callback assumes there is a keras layer named 'kl_loss' in the model.

    Parameters
    ----------
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
        curve,
        annealing_sharpness,
        n_annealing_epochs,
        n_cycles=1,
    ):
        if curve not in ["linear", "tanh"]:
            raise NotImplementedError(curve)

        super().__init__()
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

        # Calculate new value
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
        else:
            new_value = 1.0

        # Update the annealing factor in the layer that calculates the KL loss
        kl_loss_layer = self.model.get_layer("kl_loss")
        kl_loss_layer.annealing_factor.assign(new_value)


class SaveBestCallback(callbacks.ModelCheckpoint):
    """Callback to save the best model.

    The best model is determined as the model with the lowest loss.

    Parameters
    ----------
    save_best_after : int
        Epoch number after which to save the best model.
    """

    def __init__(self, save_best_after, *args, **kwargs):
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
