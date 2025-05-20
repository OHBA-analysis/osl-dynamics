"""Base classes inference models."""

import sys
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf
from scipy.special import xlogy, logsumexp
from tqdm.auto import tqdm, trange

import osl_dynamics.data.tf as dtf
from osl_dynamics import array_ops
from osl_dynamics.simulation import HMM
from osl_dynamics.inference import callbacks, optimizers
from osl_dynamics.inference.initializers import WeightInitializer
from osl_dynamics.models.mod_base import ModelBase
from osl_dynamics.utils.misc import replace_argument

_logger = logging.getLogger("osl-dynamics")


@dataclass
class VariationalInferenceModelConfig:
    """Settings needed for the inference model."""

    # Alpha parameters
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
            self.initial_alpha_temperature = 1.0

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

    def fit(self, *args, kl_annealing_callback=None, lr_decay=None, **kwargs):
        """Wrapper for the standard keras fit method.

        Parameters
        ----------
        *args : arguments
            Arguments for :code:`ModelBase.fit()`.
        kl_annealing_callback : bool, optional
            Should we update the KL annealing factor during training?
        lr_decay : float, optional
            Learning rate decay after KL annealing period.
        **kwargs : keyword arguments, optional
            Keyword arguments for :code:`ModelBase.fit()`.

        Returns
        -------
        history : history
            The training history.
        """
        # Validation
        if lr_decay is None:
            lr_decay = self.config.lr_decay

        if kl_annealing_callback is None:
            kl_annealing_callback = self.config.do_kl_annealing

        # Learning rate decay
        if kl_annealing_callback:
            decay_start_epoch = self.config.n_kl_annealing_epochs
        else:
            decay_start_epoch = 0
        learning_rate = self.model.optimizer.learning_rate.numpy()

        def lr_scheduler(epoch, lr):
            if epoch < decay_start_epoch:
                return learning_rate
            else:
                return learning_rate * np.exp(
                    -lr_decay * (epoch - decay_start_epoch + 1)
                )

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
        args, kwargs = replace_argument(
            self.model.fit,
            "callbacks",
            [lr_callback],
            args,
            kwargs,
            append=True,
        )

        # KL annealing
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
        n_epochs=None,
        n_init=None,
        take=None,
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
        n_epochs : int, optional
            Number of epochs to train the model.
            By default we use the value passed in the config.
        n_init : int, optional
            Number of initializations.
            By default we use the value passed in the config.
        take : float, optional
            Fraction of total batches to take.
            By default we use the value passed in the config.
        n_kl_annealing_epochs : int, optional
            Number of KL annealing epochs.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method.

        Returns
        -------
        history : history
            The training history of the best initialization.
        """
        n_epochs = n_epochs or self.config.n_init_epochs
        n_init = n_init or self.config.n_init
        take = take or self.config.init_take

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
        training_dataset = self.make_dataset(
            training_data, shuffle=True, concatenate=True
        )

        # Calculate the number of batches to use
        if take < 1:
            n_total_batches = dtf.get_n_batches(training_dataset)
            n_batches = max(round(n_total_batches * take), 1)
            _logger.info(f"Using {n_batches} out of {n_total_batches} batches")

        # Pick the initialization with the lowest free energy
        best_loss = np.inf
        for n in range(n_init):
            _logger.info(f"Initialization {n}")
            self.reset()
            if take < 1:
                training_data_subset = training_dataset.take(n_batches)
            else:
                training_data_subset = training_dataset

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

        if best_loss == np.inf:
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
        n_epochs=None,
        n_init=None,
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
        n_epochs : int, optional
            Number of epochs to train.
            By default we use the value passed in the config.
        n_init : int, optional
            How many subjects should we train on?
            By default we use the value passed in the config.
        n_kl_annealing_epochs : int, optional
            Number of KL annealing epochs to use during initialization. If
            :code:`None` then the KL annealing epochs in the :code:`config`
            is used.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method.
        """
        n_epochs = n_epochs or self.config.n_init_epochs
        n_init = n_init or self.config.n_init

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

        # Make a list of TensorFlow Datasets
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
        best_loss = np.inf
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
        n_epochs=None,
        n_init=None,
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

    def random_state_time_course_initialization(
        self,
        training_data,
        n_epochs=None,
        n_init=None,
        take=None,
        stay_prob=0.9,
        **kwargs,
    ):
        """Random state time course initialization.

        The model is trained for a few epochs with a sampled state time course
        initialization. The model with the best free energy is kept.

        Parameters
        ----------
        training_data : tf.data.Dataset or osl_dynamics.data.Data
            Dataset to use for training.
        n_epochs : int, optional
            Number of epochs to train the model.
            By default we use the value passed in the config.
        n_init : int, optional
            Number of initializations.
            By default we use the value passed in the config.
        take : float, optional
            Fraction of total batches to take.
            By default we use the value passed in the config.
        stay_prob : float, optional
            Stay probability (diagonal for the transition probability
            matrix). Other states have uniform probability.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method.

        Returns
        -------
        history : history
            The training history of the best initialization.
        """
        n_epochs = n_epochs or self.config.n_init_epochs
        n_init = n_init or self.config.n_init
        take = take or self.config.init_take

        if n_init is None or n_init == 0:
            _logger.info(
                "Number of initializations was set to zero. "
                + "Skipping initialization."
            )
            return

        _logger.info("Random state time course initialization")

        # Make a TensorFlow Dataset
        training_dataset = self.make_dataset(
            training_data, shuffle=True, concatenate=True
        )

        # Calculate the number of batches to use
        if take < 1:
            n_total_batches = dtf.get_n_batches(training_dataset)
            n_batches = max(round(n_total_batches * take), 1)
            _logger.info(f"Using {n_batches} out of {n_total_batches} batches")

        # Pick the initialization with the lowest free energy
        best_loss = np.inf
        for n in range(n_init):
            _logger.info(f"Initialization {n}")
            self.reset()
            if take < 1:
                training_data_subset = training_dataset.take(n_batches)
            else:
                training_data_subset = training_dataset

            self.set_random_state_time_course_initialization(
                training_data_subset, stay_prob
            )

            try:
                history = self.fit(training_data_subset, epochs=n_epochs, **kwargs)
            except tf.errors.InvalidArgumentError as e:
                _logger.warning(e)
                _logger.warning("Training failed! Skipping initialization.")
                continue

            loss = history["loss"][-1]
            if loss < best_loss:
                best_initialization = n
                best_loss = loss
                best_history = history
                best_weights = self.get_weights()

        if best_loss == np.inf:
            raise ValueError("No valid initializations were found.")

        _logger.info(f"Using initialization {best_initialization}")
        self.set_weights(best_weights)
        self.reset_kl_annealing_factor()

        return best_history

    def set_random_state_time_course_initialization(
        self, training_dataset, stay_prob=0.9
    ):
        """Sets the initial means/covariances based on a random state time course.

        Parameters
        ----------
        training_dataset : tf.data.Dataset
            Training data.
        stay_prob : float, optional
            Stay probability (diagonal for the transition probability
            matrix). Other states have uniform probability.
        """
        _logger.info("Setting random means and covariances")

        # HMM simulation to sample from
        sim = HMM(
            trans_prob="uniform",
            stay_prob=stay_prob,
            n_states=self.config.n_states or self.config.n_modes,
        )

        # Mean and covariance for each state
        means = np.zeros(
            [self.config.n_states or self.config.n_modes, self.config.n_channels],
            dtype=np.float32,
        )
        covariances = np.zeros(
            [
                self.config.n_states or self.config.n_modes,
                self.config.n_channels,
                self.config.n_channels,
            ],
            dtype=np.float32,
        )

        n_batches = 0
        for batch in training_dataset:
            # Concatenate all the sequences in this batch
            data = np.concatenate(batch["data"])

            if data.shape[0] < 2 * self.config.n_channels:
                raise ValueError(
                    "Not enough time points in batch, "
                    "increase batch_size or sequence_length"
                )

            # Sample a state time course using the initial transition
            # probability matrix
            stc = sim.generate_states(data.shape[0])

            # Make sure each state activates
            non_active_states = np.sum(stc, axis=0) < 2 * self.config.n_channels
            while np.any(non_active_states):
                new_stc = sim.generate_states(data.shape[0])
                new_active_states = np.sum(new_stc, axis=0) != 0
                for j in range(self.config.n_states or self.config.n_modes):
                    if non_active_states[j] and new_active_states[j]:
                        stc[:, j] = new_stc[:, j]
                non_active_states = np.sum(stc, axis=0) < 2 * self.config.n_channels

            # Calculate the mean/covariance for each state for this batch
            m = []
            C = []
            for j in range(self.config.n_states or self.config.n_modes):
                x = data[stc[:, j] == 1]
                mu = np.mean(x, axis=0)
                sigma = np.cov(x, rowvar=False)
                m.append(mu)
                C.append(sigma)
            means += m
            covariances += C
            n_batches += 1

        # Calculate the average from the running total
        means /= n_batches
        covariances /= n_batches

        if self.config.learn_means:
            # Set initial means
            self.set_means(means, update_initializer=True)

        if self.config.learn_covariances:
            # Set initial covariances
            self.set_covariances(covariances, update_initializer=True)

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
        super().reset_weights(keep=keep)
        self.reset_kl_annealing_factor()

    def get_theta(
        self, dataset, concatenate=False, remove_edge_effects=False, **kwargs
    ):
        """Mode mixing logits, :code:`theta`.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Prediction dataset. This can be a list of datasets, one for each
            session.
        concatenate : bool, optional
            Should we concatenate theta for each session?
        remove_edge_effects : bool, optional
            Edge effects can arise due to separating the data into sequences.
            We can remove these by predicting overlapping :code:`theta` and
            disregarding the :code:`theta` near the ends. Passing :code:`True`
            does this by using sequences with 50% overlap and throwing away the
            first and last 25% of predictions.

        Returns
        -------
        theta : list or np.ndarray
            Mode mixing logits with shape (n_sessions, n_samples, n_modes)
            or (n_samples, n_modes).
        fc_theta : list or np.ndarray
            Mode mixing logits for FC.
            Only returned if :code:`self.config.multiple_dynamics=True`.
        """
        if self.is_multi_gpu:
            raise ValueError(
                "MirroredStrategy is not supported for this method. "
                + "Please load a new model with "
                + "osl_dynamics.models.load(..., single_gpu=True)."
            )

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

        n_datasets = len(dataset)
        if len(dataset) > 1:
            iterator = trange(n_datasets, desc="Getting theta")
            kwargs["verbose"] = 0
        else:
            iterator = range(n_datasets)
            _logger.info("Getting theta")

        theta = []
        for i in iterator:
            predictions = self.predict(dataset[i], **kwargs)
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
            session.
        concatenate : bool, optional
            Should we concatenate theta for each session?
        remove_edge_effects : bool, optional
            Edge effects can arise due to separating the data into sequences.
            We can remove these by predicting overlapping :code:`theta` and
            disregarding the :code:`theta` near the ends. Passing :code:`True`
            does this by using sequences with 50% overlap and throwing away the
            first and last 25% of predictions.

        Returns
        -------
        power_theta : list or np.ndarray
            Mode mixing logits for power with shape (n_sessions, n_samples,
            n_modes) or (n_samples, n_modes).
        fc_theta : list or np.ndarray
            Mode mixing logits for FC with shape (n_sessions, n_samples,
            n_modes) or (n_samples, n_modes).
        """
        if self.is_multi_gpu:
            raise ValueError(
                "MirroredStrategy is not supported for this method. "
                + "Please load a new model with "
                + "osl_dynamics.models.load(..., single_gpu=True)."
            )

        if not self.config.multiple_dynamics:
            raise ValueError("Please use get_theta for a single time scale model.")

        if remove_edge_effects:
            step_size = self.config.sequence_length // 2  # 50% overlap
        else:
            step_size = None

        dataset = self.make_dataset(dataset, step_size=step_size)

        n_datasets = len(dataset)
        if len(dataset) > 1:
            iterator = trange(n_datasets, desc="Getting mode logits")
            kwargs["verbose"] = 0
        else:
            iterator = range(n_datasets)
            _logger.info("Getting mode logits")

        power_theta = []
        fc_theta = []
        for i in iterator:
            predictions = self.predict(dataset[i], **kwargs)
            power_theta_ = predictions["power_theta"]
            fc_theta_ = predictions["fc_theta"]
            if remove_edge_effects:
                trim = step_size // 2  # throw away 25%
                power_theta_ = (
                    [power_theta_[0, :-trim]]
                    + list(power_theta_[1:-1, trim:-trim])
                    + [power_theta_[-1, trim:]]
                )
                fc_theta_ = (
                    [fc_theta_[0, :-trim]]
                    + list(fc_theta_[1:-1, trim:-trim])
                    + [fc_theta_[-1, trim:]]
                )
            power_theta.append(np.concatenate(power_theta_))
            fc_theta.append(np.concatenate(fc_theta_))

        if concatenate or len(power_theta) == 1:
            power_theta = np.concatenate(power_theta)
            fc_theta = np.concatenate(fc_theta)

        return power_theta, fc_theta

    def get_alpha(
        self, dataset, concatenate=False, remove_edge_effects=False, **kwargs
    ):
        """Get mode mixing coefficients, :code:`alpha`.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Prediction dataset. This can be a list of datasets, one for each
            session.
        concatenate : bool, optional
            Should we concatenate alpha for each session?
        remove_edge_effects : bool, optional
            Edge effects can arise due to separating the data into sequences.
            We can remove these by predicting overlapping :code:`alpha` and
            disregarding the :code:`alpha` near the ends. Passing :code:`True`
            does this by using sequences with 50% overlap and throwing away the
            first and last 25% of predictions.

        Returns
        -------
        alpha : list or np.ndarray
            Mode mixing coefficients with shape (n_sessions, n_samples,
            n_modes) or (n_samples, n_modes).
        """
        if self.is_multi_gpu:
            raise ValueError(
                "MirroredStrategy is not supported for this method. "
                + "Please load a new model with "
                + "osl_dynamics.models.load(..., single_gpu=True)."
            )

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

        n_datasets = len(dataset)
        if len(dataset) > 1:
            iterator = trange(n_datasets, desc="Getting alpha")
            kwargs["verbose"] = 0
        else:
            iterator = range(n_datasets)
            _logger.info("Getting alpha")

        alpha = []
        for i in iterator:
            predictions = self.predict(dataset[i], **kwargs)
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
            session.
        concatenate : bool, optional
            Should we concatenate alpha/beta for each session?
        remove_edge_effects : bool, optional
            Edge effects can arise due to separating the data into sequences.
            We can remove these by predicting overlapping :code:`alpha`/
            :code:`beta` and disregarding the :code:`alpha`/:code:`beta` near
            the ends. Passing :code:`True` does this by using sequences with 50%
            overlap and throwing away the first and last 25% of predictions.

        Returns
        -------
        alpha : list or np.ndarray
            Alpha time course with shape (n_sessions, n_samples, n_modes) or
            (n_samples, n_modes).
        beta : list or np.ndarray
            Beta time course with shape (n_sessions, n_samples, n_modes) or
            (n_samples, n_modes).
        """
        if self.is_multi_gpu:
            raise ValueError(
                "MirroredStrategy is not supported for this method. "
                + "Please load a new model with "
                + "osl_dynamics.models.load(..., single_gpu=True)."
            )

        if not self.config.multiple_dynamics:
            raise ValueError("Please use get_alpha for a single time scale model.")

        if remove_edge_effects:
            step_size = self.config.sequence_length // 2  # 50% overlap
        else:
            step_size = None

        dataset = self.make_dataset(dataset, step_size=step_size)
        alpha_layer = self.model.get_layer("alpha")
        beta_layer = self.model.get_layer("beta")

        n_datasets = len(dataset)
        if len(dataset) > 1:
            iterator = trange(n_datasets, desc="Getting mode time courses")
            kwargs["verbose"] = 0
        else:
            iterator = range(n_datasets)
            _logger.info("Getting mode time courses")

        alpha = []
        beta = []
        for i in iterator:
            predictions = self.predict(dataset[i], **kwargs)
            power_theta = predictions["power_theta"]
            fc_theta = predictions["fc_theta"]
            alpha_ = alpha_layer(power_theta)
            beta_ = beta_layer(fc_theta)
            if remove_edge_effects:
                trim = step_size // 2  # throw away 25%
                alpha_ = (
                    [alpha_[0, :-trim]]
                    + list(alpha_[1:-1, trim:-trim])
                    + [alpha_[-1, trim:]]
                )
                beta_ = (
                    [beta_[0, :-trim]]
                    + list(beta_[1:-1, trim:-trim])
                    + [beta_[-1, trim:]]
                )
            alpha.append(np.concatenate(alpha_))
            beta.append(np.concatenate(beta_))

        if concatenate or len(alpha) == 1:
            alpha = np.concatenate(alpha)
            beta = np.concatenate(beta)

        return alpha, beta

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
        if self.is_multi_gpu:
            raise ValueError(
                "MirroredStrategy is not supported for this method. "
                + "Please load a new model with "
                + "osl_dynamics.models.load(..., single_gpu=True)."
            )

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


@dataclass
class MarkovStateInferenceModelConfig:
    """Settings needed for inferring a Markov chain for hidden states."""

    initial_trans_prob: np.ndarray = None
    learn_trans_prob: bool = True
    trans_prob_update_delay: float = 5  # alpha
    trans_prob_update_forget: float = 0.7  # beta

    initial_state_probs: np.ndarray = None
    learn_initial_state_probs: bool = True

    baum_welch_implementation: str = "log"

    def validate_hmm_parameters(self):
        if self.initial_trans_prob is not None:
            if (
                not isinstance(self.initial_trans_prob, np.ndarray)
                or self.initial_trans_prob.ndim != 2
            ):
                raise ValueError("initial_trans_prob must be a 2D numpy array.")

            if not all(np.isclose(np.sum(self.initial_trans_prob, axis=1), 1)):
                raise ValueError("rows of initial_trans_prob must sum to one.")

        if self.initial_state_probs is not None:
            if (
                not isinstance(self.initial_state_probs, np.ndarray)
                or self.initial_trans_prob.ndim != 1
            ):
                raise ValueError("initial_state_probs must be a 1D numpy array.")

            if not all(np.isclose(np.sum(self.initial_state_probs), 1)):
                raise ValueError("rows of initial_state_probs must sum to one.")

        if self.baum_welch_implementation not in ["log", "rescale"]:
            raise ValueError("baum_welch_implementation must be 'log' or 'rescale'.")


class MarkovStateInferenceModelBase(ModelBase):
    """Base class for a Markov chain hidden state inference model."""

    def fit(self, *args, lr_decay=None, **kwargs):
        """Wrapper for the standard keras fit method.

        Parameters
        ----------
        *args : arguments
            Arguments for :code:`ModelBase.fit()`.
        lr_decay : float, optional
            Learning rate decay.
        **kwargs : keyword arguments, optional
            Keyword arguments for :code:`ModelBase.fit()`.

        Returns
        -------
        history : history
            The training history.
        """
        # Callback for a learning rate decay
        if lr_decay is None:
            lr_decay = self.config.lr_decay

        def lr_scheduler(epoch, lr):
            return self.config.learning_rate * np.exp(-lr_decay * epoch)

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

        # Callback for updating the the decay rate used in the
        # EMA update of the HMM parameters
        ema_prob_decay_callback = callbacks.EMADecayCallback(
            delay=self.config.trans_prob_update_delay,
            forget=self.config.trans_prob_update_forget,
            n_epochs=self.config.n_epochs,
        )

        # Update arguments to pass to the fit method
        args, kwargs = replace_argument(
            self.model.fit,
            "callbacks",
            [lr_callback, ema_prob_decay_callback],
            args,
            kwargs,
            append=True,
        )

        return super().fit(*args, **kwargs)

    def compile(self, optimizer=None):
        """Compile the model.

        Parameters
        ----------
        optimizer : str or tf.keras.optimizers.Optimizer
            Optimizer to use when compiling.
        """

        # EMA optimizer for HMM state parameters
        decay = (
            1 + self.config.trans_prob_update_delay
        ) ** -self.config.trans_prob_update_forget
        ema_optimizer = optimizers.ExponentialMovingAverage(
            self.config.learning_rate, decay
        )
        ema_variables = self.model.get_layer("hid_state_inf").trainable_variables

        # Optimizer for all other trainable parameters
        base_optimizer = tf.keras.optimizers.get(
            {
                "class_name": self.config.optimizer.lower(),
                "config": {
                    "learning_rate": self.config.learning_rate,
                },
            }
        )

        # Combine into a single optimizer for the model
        optimizer = optimizers.MarkovStateModelOptimizer(
            base_optimizer,
            ema_optimizer,
            ema_variables,
            learning_rate=self.config.learning_rate,
        )

        # Compile
        super().compile(optimizer)

    def get_alpha(
        self, dataset, concatenate=False, remove_edge_effects=False, **kwargs
    ):
        """Get state probabilities.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Prediction dataset. This can be a list of datasets, one for
            each session.
        concatenate : bool, optional
            Should we concatenate alpha for each session?
        remove_edge_effects : bool, optional
            Edge effects can arise due to separating the data into sequences.
            We can remove these by predicting overlapping :code:`alpha` and
            disregarding the :code:`alpha` near the ends. Passing :code:`True`
            does this by using sequences with 50% overlap and throwing away the
            first and last 25% of predictions.

        Returns
        -------
        alpha : list or np.ndarray
            State probabilities with shape (n_sessions, n_samples, n_states)
            or (n_samples, n_states).
        """
        if self.is_multi_gpu:
            raise ValueError(
                "MirroredStrategy is not supported for this method. "
                + "Please load a new model with "
                + "osl_dynamics.models.load(..., single_gpu=True)."
            )

        if remove_edge_effects:
            step_size = self.config.sequence_length // 2  # 50% overlap
        else:
            step_size = None

        dataset = self.make_dataset(dataset, step_size=step_size)

        n_datasets = len(dataset)
        if len(dataset) > 1:
            iterator = trange(n_datasets, desc="Getting alpha")
            kwargs["verbose"] = 0
        else:
            iterator = range(n_datasets)
            _logger.info("Getting alpha")

        alpha = []
        for i in iterator:
            predictions = self.predict(dataset[i], **kwargs)
            alpha_ = predictions["gamma"]
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

    def get_viterbi_path(self, dataset, concatenate=False):
        """Get the Viterbi path with the Viterbi algorithm.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Prediction dataset. This can be a list of datasets, one for
            each session.
        concatenate : bool, optional
            Should we concatenate the Viterbi path for each session?

        Returns
        -------
        viterbi_path : list or np.ndarray
            Viterbi path with shape (n_sessions, n_samples) or (n_samples,).
        """
        Pi_0 = self.get_initial_state_probs()
        P = self.get_trans_prob()
        n_states = P.shape[0]

        eps = sys.float_info.epsilon
        log_Pi_0 = np.log(Pi_0 + eps)
        log_P = np.log(P + eps)

        sequence_length = self.config.sequence_length
        n_states = self.config.n_states

        def _viterbi_path(x):
            log_B = self.get_log_likelihood(x)
            batch_size = log_B.shape[0]
            log_prob = np.empty([batch_size, sequence_length, n_states], dtype=float)
            prev = np.empty([batch_size, sequence_length, n_states], dtype=int)

            # Recursion
            log_prob[:, 0] = log_Pi_0[np.newaxis, :] + log_B[:, 0]
            for t in range(1, sequence_length):
                p = (
                    log_prob[:, t - 1][..., np.newaxis]
                    + log_P[np.newaxis, ...]
                    + log_B[:, t][..., np.newaxis]
                )
                log_prob[:, t] = np.max(p, axis=-2)
                prev[:, t] = np.argmax(p, axis=-2)

            # Backtrace
            path = np.empty([batch_size, sequence_length], dtype=int)
            path[:, -1] = np.argmax(log_prob[:, -1], axis=-1)
            for t in range(sequence_length - 2, -1, -1):
                path[:, t] = prev[np.arange(batch_size), t + 1, path[:, t + 1]]

            return path

        dataset = self.make_dataset(dataset)
        n_datasets = len(dataset)
        if len(dataset) > 1:
            iterator = trange(n_datasets, desc="Getting Viterbi path")
        else:
            iterator = range(n_datasets)
            _logger.info("Getting Viterbi path")

        viterbi_path = []
        for i in iterator:
            path = []
            for data in dataset[i]:
                x = data["data"]  # (batch_size, sequence_length, n_states)
                vp = np.concatenate(_viterbi_path(x))  # concat over sequences
                path.append(vp)
            path = np.concatenate(path)  # concat over batches
            path = array_ops.get_one_hot(path, n_states)
            viterbi_path.append(path)

        if concatenate or len(viterbi_path) == 1:
            viterbi_path = np.concatenate(viterbi_path)

        return viterbi_path

    def get_trans_prob(self):
        """Get the transition probability matrix.

        Returns
        -------
        trans_prob : np.ndarray
            Transition probability matrix. Shape is (n_states, n_states).
        """
        layer = self.model.get_layer("hid_state_inf")
        return layer.get_trans_prob().numpy()

    def get_initial_state_probs(self):
        """Get the initial state probability distribution.

        Returns
        -------
        initial_distribution : np.ndarray
            Initial distribution. Shape is (n_states,).
        """
        layer = self.model.get_layer("hid_state_inf")
        return layer.get_initial_state_probs().numpy()

    def set_trans_prob(self, trans_prob, update_initializer=True):
        """Set the transition probability matrix.

        Parameters
        ----------
        trans_prob : np.ndarray
            Transition probability matrix. Shape must be (n_states, n_states).
            Rows (axis=1) must sum to one.
        """
        # Validation
        if not isinstance(trans_prob, np.ndarray) or trans_prob.ndim != 2:
            raise ValueError("trans_prob must be a 2D numpy array.")

        if not all(np.isclose(np.sum(trans_prob, axis=1), 1)):
            raise ValueError("rows of trans_prob must sum to one.")

        hidden_state_inference_layer = self.model.get_layer("hid_state_inf")
        learnable_tensor_layer = hidden_state_inference_layer.layers[1]
        learnable_tensor_layer.tensor.assign(trans_prob.astype(np.float32))

        if update_initializer:
            learnable_tensor_layer.tensor_initializer = WeightInitializer(
                trans_prob.astype(np.float32)
            )

    def random_subset_initialization(
        self,
        training_data,
        n_epochs=None,
        n_init=None,
        take=None,
        **kwargs,
    ):
        """Random subset initialization.

        The model is trained for a few epochs with different random subsets
        of the training dataset. The model with the best free energy is kept.

        Parameters
        ----------
        training_data : tf.data.Dataset or osl_dynamics.data.Data
            Dataset to use for training.
        n_epochs : int, optional
            Number of epochs to train the model.
            By default we use the value passed in the config.
        n_init : int, optional
            Number of initializations.
            By default we use the value passed in the config.
        take : float, optional
            Fraction of total batches to take.
            By default we use the value passed in the config.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method.

        Returns
        -------
        history : history
            The training history of the best initialization.
        """
        n_epochs = n_epochs or self.config.n_init_epochs
        n_init = n_init or self.config.n_init
        take = take or self.config.init_take

        if n_init is None or n_init == 0:
            _logger.info(
                "Number of initializations was set to zero. "
                + "Skipping initialization."
            )
            return

        _logger.info("Random subset initialization")

        # Make a TensorFlow Dataset
        training_dataset = self.make_dataset(
            training_data, shuffle=True, concatenate=True
        )

        # Calculate the number of batches to use
        if take < 1:
            n_total_batches = dtf.get_n_batches(training_dataset)
            n_batches = max(round(n_total_batches * take), 1)
            _logger.info(f"Using {n_batches} out of {n_total_batches} batches")

        # Pick the initialization with the lowest free energy
        best_loss = np.inf
        for n in range(n_init):
            _logger.info(f"Initialization {n}")
            self.reset()
            if take < 1:
                training_data_subset = training_dataset.take(n_batches)
            else:
                training_data_subset = training_dataset

            try:
                history = self.fit(
                    training_data_subset,
                    epochs=n_epochs,
                    **kwargs,
                )
            except tf.errors.InvalidArgumentError as e:
                _logger.warning(e)
                _logger.warning("Training failed! Skipping initialization.")
                continue

            loss = history["loss"][-1]
            if loss < best_loss:
                best_initialization = n
                best_loss = loss
                best_history = history
                best_weights = self.get_weights()

        if best_loss == np.inf:
            raise ValueError("No valid initializations were found.")

        _logger.info(f"Using initialization {best_initialization}")
        self.reset()
        self.set_weights(best_weights)

        return best_history

    def random_state_time_course_initialization(
        self,
        training_data,
        n_epochs=None,
        n_init=None,
        take=None,
        **kwargs,
    ):
        """Random state time course initialization.

        The model is trained for a few epochs with a sampled state time course
        initialization. The model with the best free energy is kept.

        Parameters
        ----------
        training_data : tf.data.Dataset or osl_dynamics.data.Data
            Dataset to use for training.
        n_epochs : int, optional
            Number of epochs to train the model.
            By default we use the value passed in the config.
        n_init : int, optional
            Number of initializations.
            By default we use the value passed in the config.
        take : float, optional
            Fraction of total batches to take.
            By default we use the value passed in the config.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method.

        Returns
        -------
        history : history
            The training history of the best initialization.
        """
        n_epochs = n_epochs or self.config.n_init_epochs
        n_init = n_init or self.config.n_init
        take = take or self.config.init_take

        if n_init is None or n_init == 0:
            _logger.info(
                "Number of initializations was set to zero. "
                + "Skipping initialization."
            )
            return

        _logger.info("Random state time course initialization")

        # Make a TensorFlow Dataset
        training_dataset = self.make_dataset(
            training_data, shuffle=True, concatenate=True
        )

        # Calculate the number of batches to use
        if take < 1:
            n_total_batches = dtf.get_n_batches(training_dataset)
            n_batches = max(round(n_total_batches * take), 1)
            _logger.info(f"Using {n_batches} out of {n_total_batches} batches")

        # Pick the initialization with the lowest free energy
        best_loss = np.inf
        for n in range(n_init):
            _logger.info(f"Initialization {n}")
            self.reset()
            if take < 1:
                training_data_subset = training_dataset.take(n_batches)
            else:
                training_data_subset = training_dataset

            self.set_random_state_time_course_initialization(training_data_subset)

            try:
                history = self.fit(training_data_subset, epochs=n_epochs, **kwargs)
            except tf.errors.InvalidArgumentError as e:
                _logger.warning(e)
                _logger.warning("Training failed! Skipping initialization.")
                continue

            loss = history["loss"][-1]
            if loss < best_loss:
                best_initialization = n
                best_loss = loss
                best_history = history
                best_weights = self.get_weights()

        if best_loss == np.inf:
            raise ValueError("No valid initializations were found.")

        _logger.info(f"Using initialization {best_initialization}")
        self.reset()
        self.set_weights(best_weights)

        return best_history

    def set_random_state_time_course_initialization(self, training_dataset):
        """Sets the initial means/covariances based on a random state time course.

        Parameters
        ----------
        training_dataset : tf.data.Dataset
            Training data.
        """
        _logger.info("Setting random means and covariances")

        # Mean and covariance for each state
        means = np.zeros(
            [self.config.n_states, self.config.n_channels], dtype=np.float32
        )
        covariances = np.zeros(
            [self.config.n_states, self.config.n_channels, self.config.n_channels],
            dtype=np.float32,
        )

        n_batches = 0
        for batch in training_dataset:
            # Concatenate all the sequences in this batch
            data = np.concatenate(batch["data"])

            if data.shape[0] < 2 * self.config.n_channels:
                raise ValueError(
                    "Not enough time points in batch, "
                    "increase batch_size or sequence_length"
                )

            # Sample a state time course using the initial transition
            # probability matrix
            stc = self.sample_state_time_course(data.shape[0])

            # Make sure each state activates
            non_active_states = np.sum(stc, axis=0) < 2 * self.config.n_channels
            while np.any(non_active_states):
                new_stc = self.sample_state_time_course(data.shape[0])
                new_active_states = np.sum(new_stc, axis=0) != 0
                for j in range(self.config.n_states):
                    if non_active_states[j] and new_active_states[j]:
                        stc[:, j] = new_stc[:, j]
                non_active_states = np.sum(stc, axis=0) < 2 * self.config.n_channels

            # Calculate the mean/covariance for each state for this batch
            m = []
            C = []
            for j in range(self.config.n_states):
                x = data[stc[:, j] == 1]
                mu = np.mean(x, axis=0)
                sigma = np.cov(x, rowvar=False)
                m.append(mu)
                C.append(sigma)
            means += m
            covariances += C
            n_batches += 1

        # Calculate the average from the running total
        means /= n_batches
        covariances /= n_batches

        if self.config.learn_means:
            # Set initial means
            self.set_means(means, update_initializer=True)

        if self.config.learn_covariances:
            # Set initial covariances
            self.set_covariances(covariances, update_initializer=True)

    def sample_state_time_course(self, n_samples):
        """Sample a state time course.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        stc : np.ndarray
            State time course with shape (n_samples, n_states).
        """
        trans_prob = self.get_trans_prob()
        sim = HMM(trans_prob)
        return sim.generate_states(n_samples)

    def get_log_likelihood(self, x):
        """Log-likelihood.

        Parameters
        ----------
        x : np.ndarray
            Data. Shape is (batch_size, sequence_length, n_channels).

        Returns
        -------
        log_likelihood : np.ndarray
            Log-likelihood. Shape is (batch_size, sequence_length, n_states).
        """
        if not isinstance(x, np.ndarray) and not isinstance(x, tf.Tensor):
            raise ValueError("A numpy array or Tensor should be passed for the x.")

        if self.is_multi_gpu:
            raise ValueError(
                "MirroredStrategy is not supported for this method. "
                + "Please load a new model with "
                + "osl_dynamics.models.load(..., single_gpu=True)."
            )

        obs_mod_params = self.get_observation_model_parameters()
        args = [x] + list(obs_mod_params)
        ll_layer = self.model.get_layer("ll")
        return ll_layer(args).numpy()

    def get_posterior_entropy(self, gamma, xi):
        r"""Posterior entropy.

        Calculate the entropy of the posterior distribution:

        .. math::
            E &= \int q(s_{1:T}) \log q(s_{1:T}) ds_{1:T}

              &= \displaystyle\sum_{t=1}^{T-1} \int q(s_t, s_{t+1}) \
                 \log q(s_t, s_{t+1}) ds_t ds_{t+1} - \
                 \displaystyle\sum_{t=2}^{T-1} \
                 \int q(s_t) \log q(s_t) ds_t

        Parameters
        ----------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data,
            :math:`q(s_t)`. Shape is (batch_size, sequence_length, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive
            time points, :math:`q(s_t, s_{t+1})`. Shape is
            (batch_size, sequence_length-1, n_states, n_states).

        Returns
        -------
        entropy : float
            Posterior entropy.
        """

        # first_term = sum^{T-1}_t=1 int q(s_t, s_t+1)
        # log(q(s_t, s_t+1)) ds_t ds_t+1
        first_term = xlogy(xi, xi)
        first_term = np.sum(first_term, axis=(1, 2, 3))

        # second_term = sum^{T-1}_t=2 int q(s_t) log q(s_t) ds_t
        second_term = xlogy(gamma, gamma)[:, 1:-1, :]
        second_term = np.sum(second_term, axis=(1, 2))

        # Average over sequences in a batch
        entropy = np.mean(first_term - second_term)

        if self.config.loss_calc == "mean":
            # Correct sum over time into an average
            entropy /= self.config.sequence_length

        return entropy

    def get_posterior_expected_log_likelihood(self, x, gamma):
        r"""Posterior expected log-likelihood.

        Calculates the expected log-likelihood with respect to the posterior
        distribution of the states:

        .. math::
            LL &= \int q(s_{1:T}) \log \prod_{t=1}^T p(x_t | s_t) ds_{1:T}

               &= \sum_{t=1}^T \int q(s_t) \log p(x_t | s_t) ds_t

        Parameters
        ----------
        x : np.ndarray
            Data. Shape is (batch_size, sequence_length, n_channels).
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data,
            :math:`q(s_t)`. Shape is (batch_size, sequence_length, n_states).

        Returns
        -------
        log_likelihood : float
            Posterior expected log-likelihood.
        """
        log_likelihood = self.get_log_likelihood(x)
        expected_log_likelihood = log_likelihood * gamma

        # Sum over time points and states
        expected_log_likelihood = np.sum(expected_log_likelihood, axis=(1, 2))

        # Average over sequences in a batch
        expected_log_likelihood = np.mean(expected_log_likelihood, axis=0)

        if self.config.loss_calc == "mean":
            # Correct sum over time into an average
            expected_log_likelihood /= self.config.sequence_length

        return expected_log_likelihood

    def get_posterior_expected_prior(self, gamma, xi):
        r"""Posterior expected prior.

        Calculates the expected prior probability of states with respect to the
        posterior distribution of the states:

        .. math::
            P &= \int q(s_{1:T}) \log p(s_{1:T}) ds

              &= \int q(s_1) \log p(s_1) ds_1 + \displaystyle\sum_{t=1}^{T-1} \
                 \int q(s_t, s_{t+1}) \log p(s_{t+1} | s_t) ds_t ds_{t+1}

        Parameters
        ----------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data,
            :math:`q(s_t)`. Shape is (batch_size, sequence_length, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive
            time points, :math:`q(s_t, s_{t+1})`. Shape is
            (batch_size, sequence_length-1, n_states, n_states).

        Returns
        -------
        prior : float
            Posterior expected prior probability.
        """
        initial_distribution = self.get_initial_state_probs()
        trans_prob = self.get_trans_prob()

        # first_term = int q(s_1) log p(s_1) ds_1
        first_term = xlogy(gamma[:, 0, :], initial_distribution[None, ...])
        first_term = np.sum(first_term, axis=1)

        # remaining_terms =
        # sum^{T-1}_t=1 int q(s_t, s_t+1) log p(s_t+1 | s_t}) ds_t ds_t+1
        remaining_terms = xlogy(xi, trans_prob[None, None, ...])
        remaining_terms = np.sum(remaining_terms, axis=(1, 2, 3))

        # Average over sequences in a batch
        prior = np.mean(first_term + remaining_terms)

        if self.config.loss_calc == "mean":
            # Correct sum over time into an average
            prior /= self.config.sequence_length

        return prior

    def free_energy(self, dataset):
        r"""Get the variational free energy of HMM-based models.

        This calculates:

        .. math::
            \mathcal{F} = \int q(s_{1:T}) \log \left[ \
                          \frac{q(s_{1:T})}{p(x_{1:T}, s_{1:T})} \right] \
                          ds_{1:T}

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Dataset to evaluate the free energy for.

        Returns
        -------
        free_energy : float
            Variational free energy.
        """
        if self.is_multi_gpu:
            raise ValueError(
                "MirroredStrategy is not supported for this method. "
                + "Please load a new model with "
                + "osl_dynamics.models.load(..., single_gpu=True)."
            )

        dataset = self.make_dataset(dataset, concatenate=True)

        free_energy = []
        weights = []
        for batch in tqdm(dataset, desc="Getting free energy"):
            predictions = self.predict(batch, verbose=0)
            nll = predictions["ll_loss"][0]
            entropy = self.get_posterior_entropy(
                predictions["gamma"], predictions["xi"]
            )
            prior = self.get_posterior_expected_prior(
                predictions["gamma"], predictions["xi"]
            )
            fe = nll + entropy - prior
            if self.config.model_name == "HIVE":
                kl_loss = predictions["kl_loss"][0]
                fe += kl_loss
            free_energy.append(fe)
            weights.append(batch["data"].shape[0])

        return np.average(free_energy, weights=weights)

    def evidence(self, dataset):
        """Calculate the model evidence, :math:`p(x)`, of HMM on a dataset.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Dataset to evaluate the model evidence on.

        Returns
        -------
        evidence : float
            Model evidence.
        """

        def _evidence_predict_step(log_smoothing_distribution=None):
            # Predict step for calculating the evidence
            # p(s_t=j|x_{1:t-1}) = sum_i p(s_t=j|s_{t-1}=i) p(s_{t-1}=i|x_{1:t-1})
            # log_smoothing_distribution.shape = (batch_size, n_states)
            if log_smoothing_distribution is None:
                initial_distribution = self.get_initial_state_probs()
                log_prediction_distribution = np.broadcast_to(
                    np.expand_dims(initial_distribution, axis=0),
                    (batch_size, self.config.n_states),
                )
            else:
                log_trans_prob = np.expand_dims(np.log(self.get_trans_prob()), axis=0)
                log_smoothing_distribution = np.expand_dims(
                    log_smoothing_distribution,
                    axis=-1,
                )
                log_prediction_distribution = logsumexp(
                    log_trans_prob + log_smoothing_distribution, axis=-2
                )
            return log_prediction_distribution

        def _evidence_update_step(data, log_prediction_distribution):
            # Update step for calculating the evidence
            # p(s_t=j|x_{1:t}) = p(x_t|s_t=j) p(s_t=j|x_{1:t-1}) / p(x_t|x_{1:t-1})
            # p(x_t|x_{1:t-1}) = sum_i p(x_t|s_t=i) p(s_t=i|x_{1:t-1})
            # data.shape = (batch_size, n_channels)
            # log_prediction_distribution.shape = (batch_size, n_states)

            log_likelihood = self.get_log_likelihood(data[:, np.newaxis])[:, 0]
            log_smoothing_distribution = log_likelihood + log_prediction_distribution
            predictive_log_likelihood = logsumexp(log_smoothing_distribution, axis=-1)

            # Normalise the log smoothing distribution
            log_smoothing_distribution -= np.expand_dims(
                predictive_log_likelihood,
                axis=-1,
            )

            return log_smoothing_distribution, predictive_log_likelihood

        dataset = self.make_dataset(dataset, concatenate=True)

        evidence = []
        for batch in tqdm(dataset, desc="Getting evidence"):
            data = batch["data"]
            batch_size = tf.shape(data)[0]
            batch_evidence = np.zeros(batch_size, dtype=np.float32)
            log_smoothing_distribution = None
            for t in range(self.config.sequence_length):
                log_prediction_distribution = _evidence_predict_step(
                    log_smoothing_distribution
                )
                (
                    log_smoothing_distribution,
                    predictive_log_likelihood,
                ) = _evidence_update_step(data[:, t, :], log_prediction_distribution)
                batch_evidence += predictive_log_likelihood
            evidence.append(np.mean(batch_evidence))
        evidence = np.mean(evidence)

        if self.config.loss_calc == "mean":
            evidence /= self.config.sequence_length

        return evidence
