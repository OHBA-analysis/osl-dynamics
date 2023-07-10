"""Base classes inference models.

"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference import callbacks, initializers
from osl_dynamics.models.mod_base import ModelBase
from osl_dynamics.utils.misc import replace_argument

_logger = logging.getLogger("osl-dynamics")


@dataclass
class VariationalInferenceModelConfig:
    """Settings needed for the inference model."""

    # Alpha parameters
    theta_normalization: Literal[None, "batch", "layer"] = None
    learn_alpha_temperature: bool = None
    initial_alpha_temperature: float = None
    theta_std_epsilon: float = 1e-6

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
                        "kl_annealing_curve='tanh'."
                    )

                if self.kl_annealing_sharpness < 0:
                    raise ValueError("KL annealing sharpness must be positive.")

            if self.n_kl_annealing_epochs is None:
                raise ValueError(
                    "If we are performing KL annealing, "
                    "n_kl_annealing_epochs must be passed."
                )

            if self.n_kl_annealing_epochs < 1:
                raise ValueError(
                    "Number of KL annealing epochs must be greater than zero."
                )


class VariationalInferenceModelBase(ModelBase):
    """Base class for a variational inference model."""

    def fit(self, *args, kl_annealing_callback=None, **kwargs):
        """Wrapper for the standard keras fit method.

        Parameters
        ----------
        *args : arguments
            Arguments for :code:`ModelBase.fit()`.
        kl_annealing_callback : bool, optional
            Should we update the KL annealing factor during training?
        **kwargs : keyword arguments, optional
            Keyword arguments for :code:`ModelBase.fit()`.

        Returns
        -------
        history : history
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

    def random_subset_initialization(
        self,
        training_data,
        n_epochs,
        n_init,
        take,
        n_kl_annealing_epochs=None,
        **kwargs,
    ):
        """Random subset initialization.

        The model is trained for a few epochs with different random subsets
        of the training dataset. The model with the best free energy is kept.

        Parameters
        ----------
        training_data : tf.data.Dataset or osl_dynamics.data.Data
            Dataset to use for training.
        n_epochs : int
            Number of epochs to train the model.
        n_init : int
            Number of initializations.
        take : float
            Fraction of total batches to take.
        n_kl_annealing_epochs : int, optional
            Number of KL annealing epochs.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method.

        Returns
        -------
        history : history
            The training history of the best initialization.
        """
        if n_init is None or n_init == 0:
            _logger.warning(
                "Number of initializations was set to zero. "
                + "Skipping initialization."
            )
            return

        _logger.info("Random subset initialization")

        # Original number of KL annealing epochs
        original_n_kl_annealing_epochs = self.config.n_kl_annealing_epochs

        # Use n_kl_annealing_epochs if passed
        self.config.n_kl_annealing_epochs = (
            n_kl_annealing_epochs or original_n_kl_annealing_epochs
        )

        # Make a TensorFlow Dataset
        training_data = self.make_dataset(
            training_data,
            shuffle=True,
            concatenate=True,
        )

        # Calculate the number of batches to use
        n_total_batches = dtf.get_n_batches(training_data)
        n_batches = max(round(n_total_batches * take), 1)
        _logger.info(f"Using {n_batches} out of {n_total_batches} batches")

        # Pick the initialization with the lowest free energy
        best_loss = np.Inf
        for n in range(n_init):
            _logger.info(f"Initialization {n}")
            self.reset()
            training_data_subset = training_data.shuffle(100000).take(n_batches)
            try:
                history = self.fit(
                    training_data_subset,
                    epochs=n_epochs,
                    **kwargs,
                )
            except tf.errors.InvalidArgumentError as e:
                _logger.warning(e)
                _logger.warning(
                    "Training failed! Could be due to instability of the KL "
                    + "term. Skipping initialization."
                )
                continue

            loss = history["loss"][-1]
            if loss < best_loss:
                best_initialization = n
                best_loss = loss
                best_history = history
                best_weights = self.get_weights()

        if best_loss == np.Inf:
            raise ValueError("No valid initializations were found.")

        _logger.info(f"Using initialization {best_initialization}")
        self.set_weights(best_weights)
        self.reset_kl_annealing_factor()

        # Reset the number of KL annealing epochs
        self.config.n_kl_annealing_epochs = original_n_kl_annealing_epochs

        return best_history

    def single_subject_initialization(
        self,
        training_data,
        n_epochs,
        n_init,
        n_kl_annealing_epochs=None,
        **kwargs,
    ):
        """Initialization for the mode means/covariances.

        Pick a subject at random, train a model, repeat a few times. Use
        the means/covariances from the best model (judged using the final loss).

        Parameters
        ----------
        training_data : list of tf.data.Dataset or osl_dynamics.data.Data
            Datasets for each subject.
        n_epochs : int
            Number of epochs to train.
        n_init : int
            How many subjects should we train on?
        n_kl_annealing_epochs : int, optional
            Number of KL annealing epochs to use during initialization. If
            :code:`None` then the KL annealing epochs in the :code:`config`
            is used.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method.
        """
        if n_init is None or n_init == 0:
            _logger.warning(
                "Number of initializations was set to zero. Skipping initialization."
            )
            return

        _logger.info("Single subject initialization")

        # Original number of KL annealing epochs
        original_n_kl_annealing_epochs = self.config.n_kl_annealing_epochs

        # Use n_kl_annealing_epochs if passed
        self.config.n_kl_annealing_epochs = (
            n_kl_annealing_epochs or original_n_kl_annealing_epochs
        )

        # Make a list of tensorflow Datasets if the data
        training_data = self.make_dataset(training_data, shuffle=True)

        if not isinstance(training_data, list):
            raise ValueError(
                "training_data must be a list of Datasets or a Data object."
            )

        # Pick n_init subjects at random
        n_all_subjects = len(training_data)
        subjects_to_use = np.random.choice(
            range(n_all_subjects),
            n_init,
            replace=False,
        )

        # Train the model a few times and keep the best one
        best_loss = np.Inf
        losses = []
        for subject in subjects_to_use:
            _logger.info(f"Using subject {subject}")

            # Get the dataset for this subject
            subject_dataset = training_data[subject]

            # Reset the model weights and train
            self.reset()
            history = self.fit(subject_dataset, epochs=n_epochs, **kwargs)
            loss = history["loss"][-1]
            losses.append(loss)
            _logger.info(f"Subject {subject} loss: {loss}")

            # Record the loss of this subject's data
            if loss < best_loss:
                best_loss = loss
                subject_chosen = subject
                best_weights = self.get_weights()

        _logger.info(f"Using means and covariances from subject {subject_chosen}")

        # Use the weights from the best initialisation for the full training
        self.set_weights(best_weights)
        self.reset_kl_annealing_factor()

        # Reset the number of KL annealing epochs
        self.config.n_kl_annealing_epochs = original_n_kl_annealing_epochs

    def multistart_initialization(
        self,
        training_data,
        n_epochs,
        n_init,
        n_kl_annealing_epochs=None,
        **kwargs,
    ):
        """Multi-start initialization.

        Wrapper for :code:`random_subset_initialization` with :code:`take=1`.

        Returns
        -------
        history : history
            The training history of the best initialization.
        """

        return self.random_subset_initialization(
            training_data,
            n_epochs,
            n_init,
            take=1,
            n_kl_annealing_epochs=n_kl_annealing_epochs,
            **kwargs,
        )

    def reset_kl_annealing_factor(self):
        """Sets the KL annealing factor to zero.

        This method assumes there is a keras layer named :code:`'kl_loss'`
        in the model.
        """
        if self.config.do_kl_annealing:
            kl_loss_layer = self.model.get_layer("kl_loss")
            kl_loss_layer.annealing_factor.assign(0.0)

    def reset_weights(self, keep=None):
        """Reset the model as if you've built a new model.

        Parameters
        ----------
        keep : list of str, optional
            Layer names to NOT reset.
        """
        initializers.reinitialize_model_weights(self.model, keep)
        self.reset_kl_annealing_factor()

    def predict(self, *args, **kwargs):
        """Wrapper for the standard keras predict method.

        Returns
        -------
        predictions : dict
            Dictionary with labels for each prediction.
        """
        predictions = self.model.predict(*args, **kwargs)
        if not self.config.multiple_dynamics:
            return_names = ["ll_loss", "kl_loss", "theta"]
        else:
            return_names = ["ll_loss", "kl_loss", "mean_theta", "fc_theta"]
        predictions_dict = dict(zip(return_names, predictions))

        return predictions_dict

    def get_theta(
        self, dataset, concatenate=False, remove_edge_effects=False, **kwargs
    ):
        """Mode mixing logits, :code:`theta`.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Prediction dataset. This can be a list of datasets, one for each
            subject.
        concatenate : bool, optional
            Should we concatenate theta for each subject?
        remove_edge_effects : bool, optional
            Edge effects can arise due to separating the data into sequences.
            We can remove these by predicting overlapping :code:`theta` and
            disregarding the :code:`theta` near the ends. Passing :code:`True`
            does this by using sequences with 50% overlap and throwing away the
            first and last 25% of predictions.

        Returns
        -------
        theta : list or np.ndarray
            Mode mixing logits with shape (n_subjects, n_samples, n_modes)
            or (n_samples, n_modes).
        fc_theta : list or np.ndarray
            Mode mixing logits for FC.
            Only returned if :code:`self.config.multiple_dynamics=True`.
        """
        if self.config.multiple_dynamics:
            return self.get_mode_logits(
                dataset,
                concatenate,
                remove_edge_effects,
            )

        if remove_edge_effects:
            step_size = self.config.sequence_length // 2  # 50% overlap
        else:
            step_size = None

        dataset = self.make_dataset(dataset, step_size=step_size)

        _logger.info("Getting theta")
        theta = []
        for ds in dataset:
            predictions = self.predict(ds, **kwargs)
            theta_ = predictions["theta"]
            if remove_edge_effects:
                trim = step_size // 2  # throw away 25%
                theta_ = (
                    [theta_[0, :-trim]]
                    + list(theta_[1:-1, trim:-trim])
                    + [theta_[-1, trim:]]
                )
            theta.append(np.concatenate(theta_))

        if concatenate or len(theta) == 1:
            theta = np.concatenate(theta)

        return theta

    def get_mode_logits(
        self, dataset, concatenate=False, remove_edge_effects=False, **kwargs
    ):
        """Get logits (:code:`theta`) for a multi-time-scale model.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Prediction dataset. This can be a list of datasets, one for each
            subject.
        concatenate : bool, optional
            Should we concatenate theta for each subject?
        remove_edge_effects : bool, optional
            Edge effects can arise due to separating the data into sequences.
            We can remove these by predicting overlapping :code:`theta` and
            disregarding the :code:`theta` near the ends. Passing :code:`True`
            does this by using sequences with 50% overlap and throwing away the
            first and last 25% of predictions.

        Returns
        -------
        mean_theta : list or np.ndarray
            Mode mixing logits for mean with shape (n_subjects, n_samples,
            n_modes) or (n_samples, n_modes).
        fc_theta : list or np.ndarray
            Mode mixing logits for FC with shape (n_subjects, n_samples,
            n_modes) or (n_samples, n_modes).
        """
        if not self.config.multiple_dynamics:
            raise ValueError("Please use get_theta for a single time scale model.")

        if remove_edge_effects:
            step_size = self.config.sequence_length // 2  # 50% overlap
        else:
            step_size = None

        dataset = self.make_dataset(dataset, step_size=step_size)

        _logger.info("Getting mode logits")
        mean_theta = []
        fc_theta = []
        for ds in dataset:
            predictions = self.predict(ds, **kwargs)
            mean_theta_ = predictions["mean_theta"]
            fc_theta_ = predictions["fc_theta"]
            if remove_edge_effects:
                trim = step_size // 2  # throw away 25%
                mean_theta_ = (
                    [mean_theta_[0, :-trim]]
                    + list(mean_theta_[1:-1, trim:-trim])
                    + [mean_theta_[-1, trim:]]
                )
                fc_theta_ = (
                    [fc_theta_[0, :-trim]]
                    + list(fc_theta_[1:-1, trim:-trim])
                    + [fc_theta_[-1, trim:]]
                )
            mean_theta.append(np.concatenate(mean_theta_))
            fc_theta.append(np.concatenate(fc_theta_))

        if concatenate or len(mean_theta) == 1:
            mean_theta = np.concatenate(mean_theta)
            fc_theta = np.concatenate(fc_theta)

        return mean_theta, fc_theta

    def get_alpha(
        self, dataset, concatenate=False, remove_edge_effects=False, **kwargs
    ):
        """Get mode mixing coefficients, :code:`alpha`.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Prediction dataset. This can be a list of datasets, one for each
            subject.
        concatenate : bool, optional
            Should we concatenate alpha for each subject?
        remove_edge_effects : bool, optional
            Edge effects can arise due to separating the data into sequences.
            We can remove these by predicting overlapping :code:`alpha` and
            disregarding the :code:`alpha` near the ends. Passing :code:`True`
            does this by using sequences with 50% overlap and throwing away the
            first and last 25% of predictions.

        Returns
        -------
        alpha : list or np.ndarray
            Mode mixing coefficients with shape (n_subjects, n_samples,
            n_modes) or (n_samples, n_modes).
        """
        if self.config.multiple_dynamics:
            return self.get_mode_time_courses(
                dataset,
                concatenate,
                remove_edge_effects,
            )

        if remove_edge_effects:
            step_size = self.config.sequence_length // 2  # 50% overlap
        else:
            step_size = None

        dataset = self.make_dataset(dataset, step_size=step_size)
        alpha_layer = self.model.get_layer("alpha")

        _logger.info("Getting alpha")
        alpha = []
        for ds in dataset:
            predictions = self.predict(ds, **kwargs)
            theta = predictions["theta"]
            alpha_ = alpha_layer(theta)
            if remove_edge_effects:
                trim = step_size // 2  # throw away 25%
                alpha_ = (
                    [alpha_[0, :-trim]]
                    + list(alpha_[1:-1, trim:-trim])
                    + [alpha_[-1, trim:]]
                )
            alpha.append(np.concatenate(alpha_))

        if concatenate or len(alpha) == 1:
            alpha = np.concatenate(alpha)

        return alpha

    def get_mode_time_courses(
        self, dataset, concatenate=False, remove_edge_effects=False, **kwargs
    ):
        """Get mode time courses (:code:`alpha`) for a multi-time-scale model.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Prediction data. This can be a list of datasets, one for each
            subject.
        concatenate : bool, optional
            Should we concatenate alpha/gamma for each subject?
        remove_edge_effects : bool, optional
            Edge effects can arise due to separating the data into sequences.
            We can remove these by predicting overlapping :code:`alpha`/
            :code:`gamma` and disregarding the :code:`alpha`/:code:`gamma` near
            the ends. Passing :code:`True` does this by using sequences with 50%
            overlap and throwing away the first and last 25% of predictions.

        Returns
        -------
        alpha : list or np.ndarray
            Alpha time course with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        gamma : list or np.ndarray
            Gamma time course with shape (n_subjects, n_samples, n_modes) or
            (n_samples, n_modes).
        """
        if not self.config.multiple_dynamics:
            raise ValueError("Please use get_alpha for a single time scale model.")

        if remove_edge_effects:
            step_size = self.config.sequence_length // 2  # 50% overlap
        else:
            step_size = None

        dataset = self.make_dataset(dataset, step_size=step_size)
        alpha_layer = self.model.get_layer("alpha")
        gamma_layer = self.model.get_layer("gamma")

        _logger.info("Getting mode time courses")
        alpha = []
        gamma = []
        for ds in dataset:
            predictions = self.predict(ds, **kwargs)
            mean_theta = predictions["mean_theta"]
            fc_theta = predictions["fc_theta"]
            alpha_ = alpha_layer(mean_theta)
            gamma_ = gamma_layer(fc_theta)
            if remove_edge_effects:
                trim = step_size // 2  # throw away 25%
                alpha_ = (
                    [alpha_[0, :-trim]]
                    + list(alpha_[1:-1, trim:-trim])
                    + [alpha_[-1, trim:]]
                )
                gamma_ = (
                    [gamma_[0, :-trim]]
                    + list(gamma_[1:-1, trim:-trim])
                    + [gamma_[-1, trim:]]
                )
            alpha.append(np.concatenate(alpha_))
            gamma.append(np.concatenate(gamma_))

        if concatenate or len(alpha) == 1:
            alpha = np.concatenate(alpha)
            gamma = np.concatenate(gamma)

        return alpha, gamma

    def losses(self, dataset, **kwargs):
        """Calculates the log-likelihood and KL loss for a dataset.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Dataset to calculate losses for.

        Returns
        -------
        ll_loss : float
            Negative log-likelihood loss.
        kl_loss : float
            KL divergence loss.
        """
        dataset = self.make_dataset(dataset, concatenate=True)
        _logger.info("Getting losses")
        predictions = self.predict(dataset, **kwargs)
        ll_loss = np.mean(predictions["ll_loss"])
        kl_loss = np.mean(predictions["kl_loss"])
        return ll_loss, kl_loss

    def free_energy(self, dataset, **kwargs):
        """Calculates the variational free energy of a dataset.

        Note, this method returns a free energy which may have a significantly
        smaller KL loss. This is because during training we sample from the
        posterior, however, when we're evaluating the model, we take the maximum
        a posteriori estimate (posterior mean). This has the effect of giving a
        lower KL loss for a given dataset.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data.
            Dataset to calculate the variational free energy for.

        Returns
        -------
        free_energy : float
            Variational free energy for the dataset.
        """
        dataset = self.make_dataset(dataset, concatenate=True)
        ll_loss, kl_loss = self.losses(dataset, **kwargs)
        free_energy = ll_loss + kl_loss
        return free_energy

    def bayesian_information_criterion(self, dataset):
        """Calculate the Bayesian Information Criterion (BIC) of the model
        for a given dataset.

        Note this method uses free energy as an approximate to the negative
        log-likelihood.

        Parameters
        ----------
        dataset : osl_dynamics.data.Data
            Dataset to calculate the BIC for.

        Returns
        -------
        bic : float
            Bayesian Information Criterion for the model (for each sequence).
        """
        loss = self.free_energy(dataset)
        n_params = self.get_n_params_generative_model()
        n_sequences = dtf.n_batches(
            dataset.time_series(concatenate=True), self.config.sequence_length
        )

        bic = (
            2 * loss
            + (np.log(self.config.sequence_length) + np.log(n_sequences))
            * n_params
            / n_sequences
        )
        return bic
