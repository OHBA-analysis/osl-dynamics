"""Base class for a joint inference and observation model.

"""
import logging
from operator import lt
from typing import Tuple, Union

import numpy as np
from tensorflow import Variable
from vrad.inference import callbacks, initializers, losses
from vrad.models.go import GO
from vrad.models.maro import MARO
from vrad.utils.misc import replace_argument

_logger = logging.getLogger("VRAD")


class InferenceAndObservationModelBase(GO, MARO):
    """Base class for a joint inference and observation model.

    Parameters
    ----------
    config : vrad.models.Config
    """

    def __init__(self, config):

        # KL annealing
        self.kl_annealing_factor = (
            Variable(0.0) if config.do_kl_annealing else Variable(1.0)
        )

        # Initialise the observation model
        if config.observation_model == "multivariate_normal":
            GO.__init__(self, config)
        elif config.observation_model == "multivariate_autoregressive":
            MARO.__init__(self, config)

    def compile(self):
        """Wrapper for the standard keras compile method."""

        # Loss function
        ll_loss = losses.ModelOutputLoss()
        kl_loss = losses.ModelOutputLoss(self.kl_annealing_factor)
        loss = [ll_loss, kl_loss]

        # Compile
        self.model.compile(optimizer=self.config.optimizer, loss=loss)

    def fit(
        self,
        *args,
        kl_annealing_callback=None,
        alpha_temperature_annealing_callback=None,
        use_tqdm=False,
        tqdm_class=None,
        use_tensorboard=None,
        tensorboard_dir=None,
        save_best_after=None,
        save_filepath=None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Adds callbacks and then trains the model.

        Parameters
        ----------
        kl_annealing_callback : bool
            Should we update the KL annealing factor during training?
        alpha_temperature_annealing_callback : bool
            Should we update the alpha temperature annealing factor during training?
        use_tqdm : bool
            Should we use a tqdm progress bar instead of the usual output from
            tensorflow.
        tqdm_class : tqdm
            Class for the tqdm progress bar.
        use_tensorboard : bool
            Should we use TensorBoard?
        tensorboard_dir : str
            Path to the location to save the TensorBoard log files.
        save_best_after : int
            Epoch number after which we should save the best model. The best model is
            that which achieves the lowest loss.
        save_filepath : str
            Path to save the best model to.

        Returns
        -------
        history
            The training history.
        """
        if use_tqdm:
            args, kwargs = replace_argument(self.model.fit, "verbose", 0, args, kwargs)

        additional_callbacks = []

        if kl_annealing_callback is None:
            kl_annealing_callback = self.config.do_kl_annealing

        if kl_annealing_callback:
            kl_annealing_callback = callbacks.KLAnnealingCallback(
                kl_annealing_factor=self.kl_annealing_factor,
                annealing_sharpness=self.config.kl_annealing_sharpness,
                n_epochs_annealing=self.config.n_epochs_kl_annealing,
            )
            additional_callbacks.append(kl_annealing_callback)

        if alpha_temperature_annealing_callback is None:
            alpha_temperature_annealing_callback = self.config.learn_alpha_temperature

        if alpha_temperature_annealing_callback:
            alpha_temperature_annealing_callback = (
                callbacks.AlphaTemperatureAnnealingCallback(
                    initial_alpha_temperature=self.config.initial_alpha_temperature,
                    final_alpha_temperature=self.config.final_alpha_temperature,
                    n_epochs_annealing=self.config.n_epochs_alpha_temperature_annealing,
                )
            )
            additional_callbacks.append(alpha_temperature_annealing_callback)

        args, kwargs = replace_argument(
            func=self.model.fit,
            name="callbacks",
            item=self.create_callbacks(
                use_tqdm,
                tqdm_class,
                use_tensorboard,
                tensorboard_dir,
                save_best_after,
                save_filepath,
                additional_callbacks,
            ),
            args=args,
            kwargs=kwargs,
            append=True,
        )

        return self.model.fit(*args, **kwargs)

    def reset_weight(self):
        """Reset the model as if you've built a new model.

        Resets the model weights, optimizer and annealing factor.
        """
        self.compile()
        initializers.reinitialize_model_weights(self.model)
        if self.config.do_kl_annealing:
            self.kl_annealing_factor.assign(0.0)
        if self.config.do_alpha_temperature_annealing:
            alpha_layer = self.model.get_layer("alpha")
            alpha_layer.alpha_temperature = self.config.initial_alpha_temperature

    def predict(self, *args, **kwargs) -> dict:
        """Wrapper for the standard keras predict method.

        Returns
        -------
        dict
            Dictionary with labels for each prediction.
        """
        predictions = self.model.predict(*args, *kwargs)
        return_names = ["ll_loss", "kl_loss", "alpha"]
        predictions_dict = dict(zip(return_names, predictions))
        return predictions_dict

    def predict_states(
        self, inputs, *args, concatenate: bool = False, **kwargs
    ) -> Union[list, np.ndarray]:
        """State mixing factors, alpha.

        Parameters
        ----------
        inputs : tensorflow.data.Dataset
            Prediction dataset.
        concatenate : bool
            Should we concatenate alpha for each subject? Optional, default
            is False.

        Returns
        -------
        np.ndarray
            State mixing factors with shape (n_subjects, n_samples,
            n_states) or (n_samples, n_states).
        """
        inputs = self._make_dataset(inputs)
        outputs = []
        for dataset in inputs:
            alpha = self.predict(dataset, *args, **kwargs)["alpha"]
            alpha = np.concatenate(alpha)
            outputs.append(alpha)
        if len(outputs) == 1:
            outputs = outputs[0]
        elif concatenate:
            outputs = np.concatenate(outputs)
        return outputs

    def losses(self, dataset, return_mean: bool = False) -> Tuple[float, float]:
        """Calculates the log-likelihood and KL loss for a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate losses for.
        return_mean : bool
            Should we return the mean loss over batches? Otherwise we return
            the sum. Optional, default is False.

        Returns
        -------
        ll_loss : float
            Negative log-likelihood loss.
        kl_loss : float
            KL divergence loss.
        """
        if return_mean:
            mean_or_sum = np.mean
        else:
            mean_or_sum = np.sum
        if isinstance(dataset, list):
            predictions = [self.predict(subject) for subject in dataset]
            ll_loss = mean_or_sum([mean_or_sum(p["ll_loss"]) for p in predictions])
            kl_loss = mean_or_sum([mean_or_sum(p["kl_loss"]) for p in predictions])
        else:
            predictions = self.predict(dataset)
            ll_loss = mean_or_sum(predictions["ll_loss"])
            kl_loss = mean_or_sum(predictions["kl_loss"])
        return ll_loss, kl_loss

    def free_energy(self, dataset, return_mean: bool = False) -> float:
        """Calculates the variational free energy of a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate the variational free energy for.
        return_mean : bool
            Should we return the mean free energy over batches? Otherwise
            we return the sum. Optional, default is False.

        Returns
        -------
        float
            Variational free energy for the dataset.
        """
        ll_loss, kl_loss = self.losses(dataset, return_mean=return_mean)
        free_energy = ll_loss + kl_loss
        return free_energy

    def get_alpha_temperature(self) -> float:
        """Alpha temperature used in the model.

        Returns
        -------
        float
            Alpha temperature.
        """
        alpha_layer = self.model.get_layer("alpha")
        alpha_temperature = alpha_layer.alpha_temperature.numpy()
        return alpha_temperature
