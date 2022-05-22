"""Base class for an inference model.

"""

import time, os
import logging
from dataclasses import dataclass
from typing import Tuple, Union, Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, utils
from tqdm import trange
from osl_dynamics.models.mod_base import ModelBase
from osl_dynamics.inference import callbacks, initializers
from osl_dynamics.inference.layers import SAGELogLikelihoodLossLayer
from osl_dynamics.utils.misc import replace_argument

_logger = logging.getLogger("osl-dynamics")


@dataclass
class VariationalInferenceModelConfig:
    """Settings needed for the inference model."""

    # Alpha parameters
    theta_normalization: Literal[None, "batch", "layer"] = None
    learn_alpha_temperature: bool = None
    initial_alpha_temperature: float = None

    # KL annealing parameters
    do_kl_annealing: bool = False
    kl_annealing_curve: Literal["linear", "tanh"] = None
    kl_annealing_sharpness: float = None
    n_kl_annealing_epochs: int = None

    def validate_alpha_parameters(self):
        if self.initial_alpha_temperature is None:
            raise ValueError("initial_alpha_temperature must be passed.")

        if self.initial_alpha_temperature <= 0:
            raise ValueError("initial_alpha_temperature must be greater than zero.")

    def validate_kl_annealing_parameters(self):
        if self.do_kl_annealing:
            if self.kl_annealing_curve is None:
                raise ValueError(
                    "If we are performing KL annealing, "
                    "kl_annealing_curve must be passed."
                )

            if self.kl_annealing_curve not in ["linear", "tanh"]:
                raise ValueError("KL annealing curve must be 'linear' or 'tanh'.")

            if self.kl_annealing_curve == "tanh":
                if self.kl_annealing_sharpness is None:
                    raise ValueError(
                        "kl_annealing_sharpness must be passed if "
                        + "kl_annealing_curve='tanh'."
                    )

                if self.kl_annealing_sharpness < 0:
                    raise ValueError("KL annealing sharpness must be positive.")

            if self.n_kl_annealing_epochs is None:
                raise ValueError(
                    "If we are performing KL annealing, "
                    + "n_kl_annealing_epochs must be passed."
                )

            if self.n_kl_annealing_epochs < 1:
                raise ValueError(
                    "Number of KL annealing epochs must be greater than zero."
                )


class VariationalInferenceModelBase(ModelBase):
    """Base class for a variational inference model."""

    def fit(
        self,
        *args,
        kl_annealing_callback: bool = None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Parameters
        ----------
        *args
            Arguments for ModelBase.fit().
        kl_annealing_callback : bool
            Should we update the KL annealing factor during training?
        **kwargs
            Keyword arguments for ModelBase.fit()

        Returns
        -------
        history
            The training history.
        """
        if kl_annealing_callback is None:
            # Check config to see if we should do KL annealing
            kl_annealing_callback = self.config.do_kl_annealing

        if kl_annealing_callback:
            kl_annealing_callback = callbacks.KLAnnealingCallback(
                curve=self.config.kl_annealing_curve,
                annealing_sharpness=self.config.kl_annealing_sharpness,
                n_annealing_epochs=self.config.n_kl_annealing_epochs,
            )

            # Update arguments to pass to the fit method
            args, kwargs = replace_argument(
                self.model.fit,
                "callbacks",
                [kl_annealing_callback],
                args,
                kwargs,
                append=True,
            )

        return super().fit(*args, **kwargs)

    def initialize(
        self,
        training_dataset,
        epochs: int,
        n_init: int,
        **kwargs,
    ):
        """Multi-start training.

        The model is trained for a few epochs with different random initializations
        for weights and the model with the best free energy is kept.

        Parameters
        ----------
        training_dataset : tensorflow.data.Dataset
            Dataset to use for training.
        epochs : int
            Number of epochs to train the model.
        n_init : int
            Number of initializations.

        Returns
        -------
        history
            The training history of the best initialization.
        """
        if n_init is None or n_init == 0:
            _logger.warning(
                "Number of initializations was set to zero. "
                + "Skipping initialization."
            )
            return

        # Pick the initialization with the lowest free energy
        best_free_energy = np.Inf
        for n in range(n_init):
            print(f"Initialization {n}")
            self.reset_weights()
            self.compile()
            history = self.fit(
                training_dataset,
                epochs=epochs,
                **kwargs,
            )
            free_energy = history.history["loss"][-1]
            if free_energy < best_free_energy:
                best_initialization = n
                best_free_energy = free_energy
                best_weights = self.model.get_weights()
                best_optimizer = self.model.optimizer
                best_history = history

        print(f"Using initialization {best_initialization}")
        self.reset_weights()
        self.model.set_weights(best_weights)
        self.compile(optimizer=best_optimizer)

        return best_history

    def reset_weights(self, keep: list = None):
        """Reset the model as if you've built a new model.

        Parameters
        ----------
        keep : list of str
            Layer names to NOT reset.
        """
        initializers.reinitialize_model_weights(self.model, keep)
        if self.config.do_kl_annealing:
            kl_loss_layer = self.model.get_layer("kl_loss")
            kl_loss_layer.annealing_factor.assign(0.0)

    def predict(self, *args, **kwargs) -> dict:
        """Wrapper for the standard keras predict method.

        Returns
        -------
        dict
            Dictionary with labels for each prediction.
        """
        predictions = self.model.predict(*args, *kwargs)
        return_names = ["ll_loss", "kl_loss", "alpha"]
        if self.config.multiple_dynamics:
            return_names.append("gamma")
        predictions_dict = dict(zip(return_names, predictions))

        return predictions_dict

    def get_alpha(
        self, inputs, *args, concatenate: bool = False, **kwargs
    ) -> Union[list, np.ndarray]:
        """Mode mixing factors, alpha.

        Parameters
        ----------
        inputs : tensorflow.data.Dataset
            Prediction dataset.
        concatenate : bool
            Should we concatenate alpha for each subject?

        Returns
        -------
        list or np.ndarray
            Mode mixing factors with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        """
        if self.config.multiple_dynamics:
            return self.get_mode_time_courses(
                inputs, *args, concatenate=concatenate, **kwargs
            )

        inputs = self._make_dataset(inputs)
        outputs = []
        for dataset in inputs:
            alpha = self.predict(dataset, *args, **kwargs)["alpha"]
            alpha = np.concatenate(alpha)
            outputs.append(alpha)

        if concatenate or len(outputs) == 1:
            outputs = np.concatenate(outputs)

        return outputs

    def get_mode_time_courses(
        self, inputs, *args, concatenate: bool = False, **kwargs
    ) -> Tuple[Union[list, np.ndarray], Union[list, np.ndarray]]:
        """Get mode time courses.

        This method is used to get mode time courses for the multi-time-scale model.

        Parameters
        ----------
        inputs : tensorflow.data.Dataset
            Prediction dataset.
        concatenate : bool
            Should we concatenate alpha for each subject?

        Returns
        -------
        list or np.ndarray
            Alpha time course with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        list or np.ndarray
            Gamma time course with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        """
        if not self.config.multiple_dynamics:
            raise ValueError("Please use get_alpha for a single time scale model.")

        inputs = self._make_dataset(inputs)

        outputs_alpha = []
        outputs_gamma = []
        for dataset in inputs:
            predictions = self.predict(dataset, *args, **kwargs)

            alpha = predictions["alpha"]
            gamma = predictions["gamma"]

            alpha = np.concatenate(alpha)
            gamma = np.concatenate(gamma)

            outputs_alpha.append(alpha)
            outputs_gamma.append(gamma)

        if concatenate or len(outputs_alpha) == 1:
            outputs_alpha = np.concatenate(outputs_alpha)
            outputs_gamma = np.concatenate(outputs_gamma)

        return outputs_alpha, outputs_gamma

    def losses(self, dataset) -> Tuple[float, float]:
        """Calculates the log-likelihood and KL loss for a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate losses for.

        Returns
        -------
        ll_loss : float
            Negative log-likelihood loss.
        kl_loss : float
            KL divergence loss.
        """
        if isinstance(dataset, list):
            predictions = [self.predict(subject) for subject in dataset]
            ll_loss = np.mean([np.mean(p["ll_loss"]) for p in predictions])
            kl_loss = np.mean([np.mean(p["kl_loss"]) for p in predictions])

        else:
            predictions = self.predict(dataset)
            ll_loss = np.mean(predictions["ll_loss"])
            kl_loss = np.mean(predictions["kl_loss"])

        return ll_loss, kl_loss

    def free_energy(self, dataset) -> float:
        """Calculates the variational free energy of a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate the variational free energy for.

        Returns
        -------
        float
            Variational free energy for the dataset.
        """
        ll_loss, kl_loss = self.losses(dataset)
        free_energy = ll_loss + kl_loss
        return free_energy


class AdversarialInferenceModelBase(ModelBase):
    """Adversarial inference model base class.

    This class assumes the model has a inference_model, generator_model,
    discriminator_model and model attribute.
    """

    def compile(self):
        """Compile the model."""

        self.discriminator_model.compile(
            loss="binary_crossentropy",
            optimizer=self.config.optimizer.lower(),
            metrics=["accuracy"],
        )
        self.discriminator_model.trainable = False

        # Reconstruction (Likelihood) loss:
        # The first loss corresponds to the likelihood - this tells us how well we
        # are explaining our data according to the current estimate of the
        # generative model, and is given by:
        # L = \sum_{t=1}^{T} log p(Y_t | \theta_t^m = \mu^{m,\theta}_t,
        #                                \theta_t^c = \mu^{c,\theta}_t)
        ll_loss = SAGELogLikelihoodLossLayer(self.config.n_channels, name="ll_loss")

        # Regularization (Prior) Loss:
        # The second loss regularises the estimate of the latent, time-varying
        # parameters [$\theta^m$, $\theta^c$] using an adaptive prior - this penalises
        # when the posterior estimates of [$\theta^m$, $\theta^c$] deviate from the
        # prior:
        # R = \sum_{t=1}^{T} [
        #     CrossEntropy(\mu^{m,\theta}_t|| \hat{\mu}^{m,\theta}_{t})
        #     + CrossEntropy(\mu^{c,\theta}_t || \hat{\mu}^{c,\theta}_{t})
        # ]

        # Optimizer
        optimizer = optimizers.get(
            {
                "class_name": self.config.optimizer.lower(),
                "config": {"learning_rate": self.config.learning_rate},
            }
        )

        # Compile the full model
        self.model.compile(
            loss=[ll_loss, "binary_crossentropy"],
            loss_weights=[0.995, 0.005],
            optimizer=optimizer,
        )

    def fit(self, *args, **kwargs):
        print("Warning: model.train() should be used with adversarial training.")
        return self.train(*args, **kwargs)

    def train(self, train_data: tf.data.Dataset, epochs: int = None, verbose: int = 1):
        """Train the model.

        Parameters
        ----------
        train_data : tensorflow.data.Dataset
            Training dataset.
        epochs : int
            Number of epochs to train. Defaults to value in config if not passed.
        verbose : int
            Should we print a progress bar?

        Returns
        -------
        history
            History of discriminator_loss and generator_loss.
        """
        if epochs is None:
            epochs = self.config.n_epochs

        # Path to save the best model weights
        timestr = time.strftime("%Y%m%d-%H%M%S")  # current date-time
        filepath = "tmp/"
        save_filepath = filepath + str(timestr) + "_best_model.h5"

        # Generating real/fake input for the descriminator
        real = np.ones((self.config.batch_size, self.config.sequence_length, 1))
        fake = np.zeros((self.config.batch_size, self.config.sequence_length, 1))

        # Batch-wise training
        train_discriminator = self._discriminator_training(real, fake)
        history = []
        best_val_loss = np.Inf
        for epoch in range(epochs):
            if verbose:
                print("Epoch {}/{}".format(epoch + 1, epochs))
                pb_i = utils.Progbar(len(train_data), stateful_metrics=["D", "G"])

            for idx, batch in enumerate(train_data):
                C_m, alpha_posterior = self.inference_model.predict_on_batch(batch)
                alpha_prior = self.generator_model.predict_on_batch(alpha_posterior)

                discriminator_loss = train_discriminator(alpha_posterior, alpha_prior)
                generator_loss = self.model.train_on_batch(batch, [batch, real])

                if verbose:
                    pb_i.add(
                        1,
                        values=[("D", discriminator_loss[1]), ("G", generator_loss[0])],
                    )

            if generator_loss[0] < best_val_loss:
                self.model.save_weights(save_filepath)
                print(
                    "Best model w/ val loss (generator) {} saved to {}".format(
                        generator_loss[0], save_filepath
                    )
                )
                best_val_loss = generator_loss[0]
            val_loss = self.model.test_on_batch([batch], [batch, real])

            history.append({"D": discriminator_loss[1], "G": generator_loss[0]})

        return history

    def _discriminator_training(self, real, fake):
        def train(real_samples, fake_samples):
            self.discriminator_model.trainable = True
            loss_real = self.discriminator_model.train_on_batch(real_samples, real)
            loss_fake = self.discriminator_model.train_on_batch(fake_samples, fake)
            loss = np.add(loss_real, loss_fake) * 0.5
            self.discriminator_model.trainable = False
            return loss

        return train

    def get_alpha(self, inputs, concatenate: bool = True) -> Union[list, np.ndarray]:
        """Mode mixing factors, alpha.

        Parameters
        ----------
        inputs : tensorflow.data.Dataset
            Prediction dataset.

        Returns
        -------
        list or np.ndarray
            Mode mixing factors with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        """
        outputs = []
        for dataset in inputs:
            alpha = self.inference_model.predict(dataset)[1]
            alpha = np.concatenate(alpha)
            outputs.append(alpha)

        if concatenate or len(outputs) == 1:
            outputs = np.concatenate(outputs)

        return outputs

    def gen_alpha(self, alpha: np.ndarray = None) -> np.ndarray:
        """Uses the generator to predict the prior alphas.

        Parameters
        ----------
        alpha : np.ndarray
            Shape must be (n_samples, n_modes).

        Returns
        -------
        np.ndarray
            Predicted alpha.
        """
        n_samples = np.shape(alpha)[0]
        alpha_sampled = np.empty([n_samples, self.config.n_modes], dtype=np.float32)

        for i in trange(
            n_samples - self.config.sequence_length,
            desc="Predicting mode time course",
            ncols=98,
        ):
            # Extract the sequence
            alpha_input = alpha[i : i + self.config.sequence_length]
            alpha_input = alpha_input[np.newaxis, :, :]
            # Predict the point estimates for theta one time step in the future
            alpha_sampled[i] = self.generator_model.predict_on_batch(alpha_input)[0, 0]

        return alpha_sampled
