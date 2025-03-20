"""Custom Tensorflow callbacks."""

import os
import numpy as np
import tensorflow as tf
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


class GumbelSoftmaxAnnealingCallback(tf.keras.callbacks.Callback):
    """Callback to anneal the temperature of a Gumbel-Softmax distribution.

    Parameters
    ----------
    curve : str
        Shape of the annealing curve.
        Can be either :code:`'linear'` or :code:`'exp'`.
    layer_name : str
        Name of the Gumbel-Softmax layer.
    n_epochs : int
        Total number of epochs.
    start_temperature : float, optional
        Starting temperature for the annealing.
    end_temperature : float, optional
        Ending temperature for the annealing.
    slope : float
        Slope of the curve. Only used when :code:`curve='exp'`.
    """

    def __init__(
        self,
        curve,
        layer_name,
        n_epochs,
        start_temperature=1.0,
        end_temperature=0.0,
        slope=0.014,
    ):
        self.curve = curve
        self.layer_name = layer_name
        self.n_epochs = n_epochs
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.slope = slope

        # Precompute temperatures for linear decay
        if self.curve == "linear":
            self.temperatures = np.linspace(
                start_temperature, end_temperature, n_epochs
            )

    def set_model(self, model):
        # Cache the Gumbel-Softmax layer when the model is set
        super().set_model(model)
        self.gumbel_softmax_layer = model.get_layer(self.layer_name)

    def on_epoch_begin(self, epoch, logs=None):
        if self.curve == "linear":
            temperature = self.temperatures[epoch]
        if self.curve == "exp":
            temperature = max(
                self.end_temperature,
                self.start_temperature * np.exp(-self.slope * epoch),
            )

        self.gumbel_softmax_layer.temperature.assign(temperature)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["temperature"] = float(self.gumbel_softmax_layer.temperature.numpy())


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
        kl_loss_layer = self.model.get_layer(
            "encoder"
        ).temporal_prior_layer.kl_loss_layer
        kl_loss_layer.annealing_factor.assign(new_value)

        # Annealing factor for gamma sampling
        if "means_dev_mag" in self.model.layers:
            means_dev_mag_layer = self.model.get_layer("means_dev_mag")
            means_dev_mag_layer.annealing_factor.assign(new_value)

        if "covs_dev_mag" in self.model.layers:
            covs_dev_mag_layer = self.model.get_layer("covs_dev_mag")
            covs_dev_mag_layer.annealing_factor.assign(new_value)

        logs["kl_factor"] = new_value


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


class CheckpointCallback(callbacks.Callback):
    """Callback to create checkpoints during training.

    Parameters
    ----------
    save_freq : int
        Frequency (in epochs) at which to save the model.
    """

    def __init__(self, save_freq, checkpoint_dir):
        super().__init__()
        self.save_freq = save_freq
        self.checkpoint = None
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = f"{checkpoint_dir}/ckpt"

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoint is None:
            self.checkpoint = tf.train.Checkpoint(
                model=self.model, optimizer=self.model.optimizer
            )
        if (epoch + 1) % self.save_freq == 0:
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)


class TensorBoardCallback(callbacks.TensorBoard):
    """Callback to log training information to TensorBoard.

    This callback extends `tf.keras.callbacks.TensorBoard` by also logging the initial weights.

    Parameters
    ----------
    log_dir : str, optional
        Path to a directory where the log files will be written.
        Defaults to None, in which case the logs will be written to a current directory.
    log_initial : bool, optional
        Whether to log the initial weights or not. Defaults to True.
    kwargs : dict
        Additional arguments to pass to the :code:`tf.keras.callbacks.TensorBoard` callback.
    """

    def __init__(self, log_dir=None, log_initial=True, **kwargs):
        # Create log directory if it does not exist
        self._log_dir = log_dir
        self._make_log_dir()

        # Get arguments
        self.log_initial = log_initial  # enable or disable initial weight logging
        self.initial_weights_logged = False  # log status

        super().__init__(log_dir=self._log_dir, **kwargs)

    def _make_log_dir(self):
        if self._log_dir is None:
            self._log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(self._log_dir, exist_ok=True)

    def on_train_begin(self, logs=None):
        # Call the parent method first
        super().on_train_begin(logs)

        # Log the initial weights once
        if self.log_initial and not self.initial_weights_logged:
            # Create a subdirectory for the initial weights
            init_log_dir = os.path.join(self._log_dir, "initial_weights")
            os.makedirs(init_log_dir, exist_ok=True)
            writer = tf.summary.create_file_writer(init_log_dir)

            # Log the initial weights
            with writer.as_default():
                for weight in self.model.weights:
                    tf.summary.histogram(weight.name, weight, step=0)
                writer.flush()  # ensure all buffered data are written to disk

            self.initial_weights_logged = True
            print(
                "Initial weights logged. You can launch TensorBoard to view the histograms."
            )


class GradientMonitoringCallback(tf.keras.callbacks.Callback):
    """Callback for logging gradients during the model training.

    Parameters
    ----------
    sample_dataset : tf.data.Dataset
        A dataset containing a representative batch of data used to compute gradients.
    log_dir : str, optional
        Path to a directory where gradient logs will be saved.
        Defaults to None, in which case the logs will be written to a current directory.
    print_stats : bool, optional
        Wheter to print the summary statistics (mean, std, min, max, L2 norm) for each variable.
        Defaults to True.
    """

    def __init__(
        self,
        sample_dataset,
        log_dir=None,
        print_stats=True,
    ):
        super().__init__()
        self.sample_dataset = sample_dataset
        self.print_stats = print_stats

        # Prepare a log directory
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs/gradients")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = tf.summary.create_file_writer(log_dir)

    @tf.function
    def compute_gradients(self, inputs):
        """Compute gradients for a given input batch.

        Parameters
        ----------
        inputs : tf.Tensor
            Input batch.
        """
        n_losses = 1
        if "HMM" not in self.model.name:
            n_losses += 1

        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)[:n_losses]
            loss = tf.reduce_sum(outputs)
        return tape.gradient(loss, self.model.trainable_variables)

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
        # Initialize accumulators for each trainable variable
        accumulated_gradients = [None] * len(self.model.trainable_variables)
        batch_count = 0

        # Compute the loss and gradients on the sample dataset
        for batch in self.sample_dataset:
            inputs = batch["data"]
            gradients = self.compute_gradients(inputs)

            # Accumulate gradients
            for i, grad in enumerate(gradients):
                if grad is not None:
                    if accumulated_gradients[i] is None:
                        accumulated_gradients[i] = grad
                    else:
                        accumulated_gradients[i] += grad
            batch_count += 1

        # Average gradients over the batches
        if batch_count > 0:
            averaged_gradients = [
                grad / batch_count if grad is not None else None
                for grad in accumulated_gradients
            ]
        else:
            averaged_gradients = accumulated_gradients

        # Group gradients by layer
        layer_gradients = {}
        for grad, var in zip(averaged_gradients, self.model.trainable_variables):
            var_name = var.name
            layer_name = var_name.split("/")[0]  # get the first part as the layer name.
            if grad is not None:
                layer_gradients.setdefault(layer_name, []).append((grad, var))

        # Log and print gradient summary statistics for each layer
        with self.writer.as_default():
            for layer, grad_var_pairs in layer_gradients.items():
                if self.print_stats:
                    print(f"\nLayer: {layer}")
                for grad, var in grad_var_pairs:
                    if grad is not None:
                        grad_mean = tf.reduce_mean(grad)
                        grad_std = tf.math.reduce_std(grad)
                        grad_min = tf.reduce_min(grad)
                        grad_max = tf.reduce_max(grad)
                        grad_norm = tf.norm(grad)

                        # Print summary statistics.
                        if self.print_stats:
                            print(f"  {var.name}:")
                            print(
                                f"    Mean: {grad_mean.numpy():.5f}, Std: {grad_std.numpy():.5f}"
                            )
                            print(
                                f"    Min: {grad_min.numpy():.5f}, Max: {grad_max.numpy():.5f}"
                            )
                            print(f"    L2 Norm: {grad_norm.numpy():.5f}")

                        # Log gradient histogram and scalar summaries.
                        tf.summary.histogram(f"gradients/{var.name}", grad, step=epoch)
                        tf.summary.scalar(
                            f"gradients/{var.name}_mean", grad_mean, step=epoch
                        )
                        tf.summary.scalar(
                            f"gradients/{var.name}_std", grad_std, step=epoch
                        )
                        tf.summary.scalar(
                            f"gradients/{var.name}_min", grad_min, step=epoch
                        )
                        tf.summary.scalar(
                            f"gradients/{var.name}_max", grad_max, step=epoch
                        )
                        tf.summary.scalar(
                            f"gradients/{var.name}_norm", grad_norm, step=epoch
                        )
                    else:
                        if self.print_stats:
                            print(f"  {var.name}: Gradient is None")
            self.writer.flush()
