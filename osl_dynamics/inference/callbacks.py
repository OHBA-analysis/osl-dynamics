"""Custom Tensorflow callbacks.

"""

import numpy as np
from tensorflow import tanh
from tensorflow.keras import callbacks

from osl_dynamics import inference


class DiceCoefficientCallback(callbacks.Callback):
    """Callback to calculate a Dice coefficient during training.

    Parameters
    ----------
    prediction_dataset : tf.data.Dataset
        Dataset to use to calculate outputs of the model.
    ground_truth_time_course : np.ndarray
        2D or 3D numpy array containing the ground truth state/mode time
        course of the training data. Shape must be (n_time_courses, n_samples,
        n_modes) or (n_samples, n_modes).
    names : list of str, optional
        Names for the time courses. Shape must be (n_time_courses,).
    """

    def __init__(
        self,
        prediction_dataset,
        ground_truth_time_course,
        names=None,
    ):
        super().__init__()
        self.prediction_dataset = prediction_dataset
        if ground_truth_time_course.ndim == 2:
            # We're training a single time scale model
            self.n_time_courses = 1
            self.gttc = ground_truth_time_course[np.newaxis, ...]
        elif ground_truth_time_course.ndim == 3:
            # We're training a multi-time-scale model
            self.n_time_courses = ground_truth_time_course.shape[0]
            self.gttc = ground_truth_time_course
        else:
            raise ValueError(
                "A 2D or 3D numpy array must be pass for ground_truth_time_course."
            )
        if names is not None:
            if len(names) != self.n_time_courses:
                raise ValueError(
                    "Mismatch between the number of names and time courses."
                )
        self.names = names
        self.n_modes = ground_truth_time_course.shape[-1]

    def on_epoch_end(self, epoch, logs=None):
        """Action to perform at the end of an epoch.

        Parameters
        ---------
        epochs : int
            Integer, index of epoch.
        logs : dict, optional
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

        # For each time course calculate the dice with respect to the
        # ground truth
        dices = []
        for i in range(self.n_time_courses):
            pmtc = inference.modes.argmax_time_courses(
                tc[i], concatenate=True, n_modes=self.n_modes
            )
            pmtc, gttc = inference.modes.match_modes(pmtc, self.gttc[i])
            dice = inference.metrics.dice_coefficient(pmtc, gttc)
            dices.append(dice)

        # Add dice to the training history and print to screen
        if self.n_time_courses == 1:
            logs["dice"] = dices[0]
        else:
            for i in range(self.n_time_courses):
                if self.names is not None:
                    key = "dice_" + self.names[i]
                else:
                    key = "dice" + str(i)
                logs[key] = dices[i]


class KLAnnealingCallback(callbacks.Callback):
    """Callback to update the KL annealing factor during training.

    This callback assumes there is a keras layer named :code:`'kl_loss'`
    in the model.

    Parameters
    ----------
    curve : str
        Shape of the annealing curve. Either :code:`'linear'` or :code:`'tanh'`.
    annealing_sharpness : float
        Parameter to control the shape of the annealing curve.
    n_annealing_epochs : int
        Number of epochs to apply annealing.
    n_cycles : int, optional
        Number of times to perform KL annealing with :code:`n_annealing_epochs`.
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
        logs : dict, optional
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

        # Annealing factor for gamma sampling
        if "means_dev_mag" in self.model.layers:
            means_dev_mag_layer = self.model.get_layer("means_dev_mag")
            means_dev_mag_layer.annealing_factor.assign(new_value)

        if "covs_dev_mag" in self.model.layers:
            covs_dev_mag_layer = self.model.get_layer("covs_dev_mag")
            covs_dev_mag_layer.annealing_factor.assign(new_value)


class EMADecayCallback(callbacks.Callback):
    """Callback to update the decay rate in an Exponential Moving Average optimizer.

    :code:`decay = (100 * epoch / n_epochs + 1 + delay) ** -forget`

    Parameters
    ----------
    delay : float
    forget : float
    """

    def __init__(self, delay, forget, n_epochs):
        super().__init__()
        self.delay = delay
        self.forget = forget
        self.n_epochs = n_epochs

    def on_epoch_end(self, epoch, logs=None):
        """Action to perform at the end of an epoch.

        Parameters
        ---------
        epochs : int
            Integer, index of epoch.
        logs : dict, optional
            Results for this training epoch, and for the validation epoch if
            validation is performed.
        """

        # Calculate new value
        new_value = (100 * epoch / self.n_epochs + 1 + self.delay) ** -self.forget

        # Print new value during training
        logs["rho"] = new_value

        # Update the decay parameter in the optimizer
        # Here we are assuming a MarkovStateModelOptimizer is being used
        ema_optimizer = self.model.optimizer.ema_optimizer
        ema_optimizer.decay = new_value


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
        logs : dict, optional
            Results for this training epoch, and for the validation epoch if
            validation is performed.
        """
        self.epochs_since_last_save += 1
        if epoch >= self.save_best_after:
            if self.save_freq == "epoch":
                self._save_model(epoch=epoch, logs=logs, batch=None)

    def on_train_end(self, logs=None):
        """Action to perform at the end of training.

        Parameters
        ----------
        logs : dict, optional
            Results for this training epoch, and for the validation epoch if
            validation is performed.
        """
        self.model.load_weights(self.filepath)
