"""Initialization of model parameters before the full training.

Every initialization method trains a few candidate initializations for a
short time and keeps the best. The multi-start loop
(:meth:`InitializationMixin._run_multistart`) is shared by all models and
a candidate is just a function that sets the initial parameters.

A candidate's parameters are set from windows of data. A window is
summarised by a feature vector (:meth:`InitializationMixin._window_features`)
and a model knows how to set its parameters from one feature vector per
state (:meth:`InitializationMixin._set_parameters_from_features`). For
Gaussian observation models the features are the mean and the covariance.
The candidate generators are:

- :code:`random_subset`: the parameters from :code:`reset()`.
- :code:`random_state_time_course`: each state's window is the data
  assigned to it by a sampled state time course.

Models customise the loop through hooks:

- :meth:`InitializationMixin._initialization_context`: settings that only
  apply while initializing (e.g. the number of KL annealing epochs).
- :meth:`InitializationMixin._restore_initialization`: how to load the
  weights of the best candidate.
- :meth:`InitializationMixin._sample_labels`: how to sample a state
  assignment for the random state time course initialization.
"""

import logging
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference.callbacks import ProgressBarCallback
from osl_dynamics.simulation import HMM
from osl_dynamics.utils.misc import listify

_logger = logging.getLogger("osl-dynamics")

# A candidate initialization: a name, the dataset to train on and a
# function that sets the initial parameters (or None to keep the
# parameters from reset())
Candidate = Tuple[str, tf.data.Dataset, Optional[Callable]]

# Default number of batches the random state time course initialization
# sets each candidate's parameters from (n_batches). A batch gives every
# state thousands of samples, so a few batches are enough. Using more makes
# every state closer to the average of the dataset, so the candidates
# differ less, and using the whole dataset is slow.
N_FEATURE_BATCHES = 20


class InitializationMixin:
    """Initialization methods shared by all models.

    Expects the class to provide :code:`config`, :code:`reset`, :code:`fit`,
    :code:`get_weights`, :code:`set_weights` and :code:`make_dataset`.
    """

    # Keyword arguments of the public methods that configure
    # _initialization_context rather than fit
    _initialization_context_keys = ()

    # ------------------------------------------------------------------
    # Hooks

    @contextmanager
    def _initialization_context(self, **kwargs):
        """Settings that only apply while initializing. No-op by default."""
        yield

    def _restore_initialization(self, weights: list, setup) -> None:
        """Load the weights of the best initialization.

        The candidate's setup is run again so the weight initializers hold
        the best candidate's parameters (a later reset() returns to them).
        """
        self.reset()
        if setup is not None:
            setup()
        self.set_weights(weights)

    @property
    def _n_states(self) -> int:
        return getattr(self.config, "n_states", None) or self.config.n_modes

    def _sample_state_time_course(self, n_samples: int, stay_prob: float):
        """Sample a state time course to initialize from.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        stay_prob : float
            Stay probability (diagonal for the transition probability
            matrix). Other states have uniform probability.

        Returns
        -------
        stc : np.ndarray
            State time course with shape (n_samples, n_states).
        """
        sim = HMM(trans_prob="uniform", stay_prob=stay_prob, n_states=self._n_states)
        return sim.generate_states(n_samples)

    @property
    def _supports_feature_initialization(self) -> bool:
        """Can this model set its parameters from window features?"""
        return hasattr(self, "set_covariances")

    @property
    def _min_window_samples(self) -> int:
        """Fewest samples a window needs for its features to be meaningful.

        For a Gaussian observation model the covariance needs more samples
        than channels.
        """
        return 2 * self.config.n_channels

    def _sample_labels(self, n_samples: int, stay_prob: float) -> np.ndarray:
        """Sample a state label for each sample."""
        return np.argmax(self._sample_state_time_course(n_samples, stay_prob), axis=1)

    def _window_features(self, windows: List[np.ndarray]) -> np.ndarray:
        """Summarise windows of data as feature vectors.

        For Gaussian observation models: the mean (if means are learnt)
        and the upper triangle of the covariance.

        Parameters
        ----------
        windows : list of np.ndarray
            Each with shape (n_samples, n_channels).

        Returns
        -------
        features : np.ndarray
            Shape is (n_windows, n_features).
        """
        n_channels = self.config.n_channels
        for window in windows:
            if window.shape[0] < 2:
                raise ValueError("A window must have at least two samples.")
        means = np.stack([np.mean(w, axis=0) for w in windows])
        covariances = np.stack(
            [
                np.atleast_2d(np.cov(w, rowvar=False)).reshape(n_channels, n_channels)
                for w in windows
            ]
        )
        i, j = np.triu_indices(n_channels)
        features = covariances[:, i, j]
        if getattr(self.config, "learn_means", False):
            features = np.concatenate([means, features], axis=1)
        return features

    def _set_parameters_from_features(self, features: np.ndarray) -> None:
        """Set the observation model from one feature vector per state.

        Parameters
        ----------
        features : np.ndarray
            Shape is (n_states, n_features), see :meth:`_window_features`.
        """
        n_channels = self.config.n_channels
        if not hasattr(self, "set_covariances"):
            raise NotImplementedError(
                f"{self.config.model_name} does not support this initialization "
                "method."
            )
        if getattr(self.config, "learn_means", False):
            means = features[:, :n_channels].astype(np.float32)
            features = features[:, n_channels:]
            self.set_means(means, update_initializer=True)
        if self.config.learn_covariances:
            i, j = np.triu_indices(n_channels)
            covariances = np.zeros(
                (features.shape[0], n_channels, n_channels), dtype=np.float32
            )
            covariances[:, i, j] = features
            covariances[:, j, i] = features
            self.set_covariances(covariances, update_initializer=True)

    # ------------------------------------------------------------------
    # Multi-start loop

    def _initialization_settings(self, n_epochs, n_init, take):
        """Fill in settings from the config."""
        return (
            n_epochs or self.config.n_init_epochs,
            n_init or self.config.n_init,
            take or self.config.init_take,
        )

    def _split_initialization_kwargs(self, kwargs: dict) -> Tuple[dict, dict]:
        """Separate context keyword arguments from fit keyword arguments."""
        context_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs)
            if key in self._initialization_context_keys
        }
        return context_kwargs, kwargs

    def _subset(self, training_dataset, take: Optional[float]):
        """A random fraction of the batches in a dataset."""
        if take is None or take >= 1:
            return training_dataset
        n_total_batches = dtf.get_n_batches(training_dataset)
        n_take = max(round(n_total_batches * take), 1)
        _logger.info(f"Using {n_take} out of {n_total_batches} batches")
        return training_dataset.take(n_take)

    def _run_multistart(
        self,
        candidates: List[Candidate],
        n_epochs: int,
        **kwargs,
    ) -> dict:
        """Train each candidate initialization and keep the best (lowest
        training loss in the last epoch).

        Parameters
        ----------
        candidates : list of (str, tf.data.Dataset, callable or None)
            Name, dataset to train on and a function that sets the initial
            parameters. The model is reset before each candidate.
        n_epochs : int
            Number of epochs to train each candidate.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method.

        Returns
        -------
        history : dict
            The training history of the best initialization.
        """
        # One progress bar for each candidate
        verbose = kwargs.pop("verbose", 1)
        user_callbacks = kwargs.pop("callbacks", None)

        best_score = np.inf
        for name, dataset, setup in candidates:
            self.reset()
            if setup is not None:
                setup()

            fit_callbacks = listify(user_callbacks)
            if verbose:
                mode = "steps" if n_epochs == 1 else "epochs"
                bar = ProgressBarCallback(desc=f"Initialization {name}", mode=mode)
                fit_callbacks = [bar] + fit_callbacks

            try:
                history = self.fit(
                    dataset,
                    epochs=n_epochs,
                    verbose=0,
                    callbacks=fit_callbacks,
                    **kwargs,
                )
            except tf.errors.InvalidArgumentError as e:
                _logger.warning(e)
                _logger.warning("Training failed! Skipping initialization.")
                continue

            score = history["loss"][-1]
            if score < best_score:
                best_name = name
                best_score = score
                best_history = history
                best_weights = self.get_weights()
                best_setup = setup

        if best_score == np.inf:
            raise ValueError("No valid initializations were found.")

        _logger.info(f"Using initialization {best_name} (loss={best_score:.4g})")
        self._restore_initialization(best_weights, best_setup)

        return best_history

    def _multistart(
        self,
        name: str,
        training_data,
        make_candidates: Callable,
        n_epochs: Optional[int],
        n_init: Optional[int],
        take: Optional[float],
        kwargs: dict,
        uses_features: bool = False,
    ) -> Optional[dict]:
        """Common wrapper for the public initialization methods.

        Parameters
        ----------
        name : str
            Name of the method for logging.
        training_data : tf.data.Dataset or osl_dynamics.data.Data
            Dataset to use for training.
        make_candidates : callable
            Called with (training_dataset, n_init) and returns the list of
            candidates to train.
        n_epochs, n_init, take : optional
            Settings, filled in from the config if :code:`None`.
        kwargs : dict
            Keyword arguments for the fit method, possibly including
            context keyword arguments.
        uses_features : bool, optional
            Does the method set parameters from window features? If so
            the model must support that.

        Returns
        -------
        history : dict
            The training history of the best initialization.
        """
        if uses_features and not self._supports_feature_initialization:
            raise NotImplementedError(
                f"{self.config.model_name} does not support {name.lower()} "
                "initialization."
            )
        n_epochs, n_init, take = self._initialization_settings(n_epochs, n_init, take)
        if not n_init:
            _logger.info(
                "Number of initializations was set to zero. Skipping initialization."
            )
            return
        context_kwargs, kwargs = self._split_initialization_kwargs(kwargs)

        _logger.info(f"{name} initialization")
        training_dataset = self.make_dataset(
            training_data, shuffle=True, concatenate=True
        )
        subset = self._subset(training_dataset, take)

        candidates = make_candidates(subset, n_init)
        candidates = [(name_, subset, setup) for name_, setup in candidates]

        with self._initialization_context(**context_kwargs):
            return self._run_multistart(candidates, n_epochs, **kwargs)

    # ------------------------------------------------------------------
    # Building candidates from windows

    def _state_time_course_features(
        self, dataset, stay_prob: float, n_batches: Optional[int]
    ) -> np.ndarray:
        """Features of the data a sampled state time course assigns to
        each state.

        A state time course is sampled for each batch and the samples
        assigned to each state are pooled over batches before the features
        are calculated.

        Parameters
        ----------
        dataset : tf.data.Dataset
            Shuffled dataset to take batches from.
        stay_prob : float
            Stay probability of the sampled state time course.
        n_batches : int
            Number of batches to use, :code:`None` for the whole dataset.
        """
        n_states = self._n_states
        if n_batches is not None:
            dataset = dataset.take(n_batches)
        samples = [[] for _ in range(n_states)]
        for batch in dataset:
            data = np.concatenate(batch["data"])
            labels = self._sample_labels(data.shape[0], stay_prob)
            for k in range(n_states):
                samples[k].append(data[labels == k])
        windows = [np.concatenate(s) if s else np.empty((0, 1)) for s in samples]
        n_samples = min(w.shape[0] for w in windows)
        if n_samples < self._min_window_samples:
            raise ValueError(
                f"A state was assigned {n_samples} samples, fewer than the "
                f"{self._min_window_samples} needed to estimate its parameters "
                f"with {self.config.n_channels} channels. Increase n_batches, "
                "batch_size or sequence_length."
            )
        return self._window_features(windows)

    # ------------------------------------------------------------------
    # Public methods

    def random_subset_initialization(
        self,
        training_data,
        n_epochs: Optional[int] = None,
        n_init: Optional[int] = None,
        take: Optional[float] = None,
        **kwargs,
    ) -> Optional[dict]:
        """Random subset initialization.

        The model is trained for a few epochs with different random subsets
        of the training dataset, starting from the parameters given by
        :code:`reset()`. The best model is kept.

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
            Keyword arguments for the fit method. Variational models also
            accept :code:`n_kl_annealing_epochs`.

        Returns
        -------
        history : dict
            The training history of the best initialization.
        """

        def make_candidates(dataset, n_init):
            return [(str(n), None) for n in range(n_init)]

        return self._multistart(
            "Random subset",
            training_data,
            make_candidates,
            n_epochs,
            n_init,
            take,
            kwargs,
        )

    def multistart_initialization(
        self,
        training_data,
        n_epochs: Optional[int] = None,
        n_init: Optional[int] = None,
        **kwargs,
    ) -> Optional[dict]:
        """Multi-start initialization.

        Wrapper for :code:`random_subset_initialization` with :code:`take=1`.

        Returns
        -------
        history : dict
            The training history of the best initialization.
        """
        return self.random_subset_initialization(
            training_data, n_epochs, n_init, take=1, **kwargs
        )

    def random_state_time_course_initialization(
        self,
        training_data,
        n_epochs: Optional[int] = None,
        n_init: Optional[int] = None,
        take: Optional[float] = None,
        stay_prob: float = 0.9,
        n_batches: Optional[int] = N_FEATURE_BATCHES,
        **kwargs,
    ) -> Optional[dict]:
        """Random state time course initialization.

        A state time course is sampled and each state's parameters are set
        from the data assigned to it in :code:`n_batches` batches. The
        model is then trained for a few epochs and the best candidate is
        kept.

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
            matrix). Other states have uniform probability. Ignored by
            models that learn a transition probability matrix, which
            sample from their initial matrix.
        n_batches : int, optional
            Number of batches used to set each candidate's initial
            parameters, :code:`None` for the whole dataset. A batch gives
            every state thousands of samples, so a few batches are enough.
            Using more makes every state closer to the average of the
            dataset, so the candidates differ less, and the samples are
            held in memory, so :code:`None` loads the whole dataset. This
            is separate from :code:`take`, which sets how much data each
            candidate is trained on.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method. Variational models also
            accept :code:`n_kl_annealing_epochs`.

        Returns
        -------
        history : dict
            The training history of the best initialization.
        """

        def make_candidates(dataset, n_init):
            candidates = []
            for n in range(n_init):
                features = self._state_time_course_features(
                    dataset, stay_prob, n_batches
                )
                candidates.append(
                    (str(n), lambda f=features: self._set_parameters_from_features(f))
                )
            return candidates

        return self._multistart(
            "Random state time course",
            training_data,
            make_candidates,
            n_epochs,
            n_init,
            take,
            kwargs,
            uses_features=True,
        )

    def set_random_state_time_course_initialization(
        self,
        training_dataset: tf.data.Dataset,
        stay_prob: float = 0.9,
        n_batches: Optional[int] = N_FEATURE_BATCHES,
    ) -> None:
        """Sets the observation model based on a random state time course.

        Parameters
        ----------
        training_dataset : tf.data.Dataset
            Training data, shuffled.
        stay_prob : float, optional
            Stay probability (diagonal for the transition probability
            matrix). Other states have uniform probability.
        n_batches : int, optional
            Number of batches to use, :code:`None` for the whole dataset.
        """
        _logger.info("Setting parameters from a random state time course")
        features = self._state_time_course_features(
            training_dataset, stay_prob, n_batches
        )
        self._set_parameters_from_features(features)

    def single_subject_initialization(
        self,
        training_data,
        n_epochs: Optional[int] = None,
        n_init: Optional[int] = None,
        **kwargs,
    ) -> Optional[dict]:
        """Single subject initialization.

        Pick a subject at random, train a model, repeat a few times. Use
        the weights from the best model.

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
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method. Variational models also
            accept :code:`n_kl_annealing_epochs`.

        Returns
        -------
        history : dict
            The training history of the best initialization.
        """
        n_epochs, n_init, _ = self._initialization_settings(n_epochs, n_init, None)
        if not n_init:
            _logger.info(
                "Number of initializations was set to zero. Skipping initialization."
            )
            return
        context_kwargs, kwargs = self._split_initialization_kwargs(kwargs)

        _logger.info("Single subject initialization")
        datasets = self.make_dataset(training_data, shuffle=True)
        if not isinstance(datasets, list):
            raise ValueError(
                "training_data must be a list of Datasets or a Data object."
            )
        # Pick n_init subjects at random
        subjects = np.random.choice(len(datasets), n_init, replace=False)
        candidates = [(f"subject {s}", datasets[s], None) for s in subjects]

        with self._initialization_context(**context_kwargs):
            return self._run_multistart(candidates, n_epochs, **kwargs)
