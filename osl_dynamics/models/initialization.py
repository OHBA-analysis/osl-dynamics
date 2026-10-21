"""Initialization of model parameters before the full training.

Every initialization method trains a few candidate initializations for a
short time and keeps the best. The multi-start loop
(:meth:`InitializationMixin._run_multistart`) is shared by all models and
a candidate is just a function that sets the initial parameters.

The candidates are built from windows of data. A window is summarised by
a feature vector (:meth:`InitializationMixin._window_features`) and a
model knows how to set its parameters from one feature vector per state
(:meth:`InitializationMixin._set_parameters_from_features`). For Gaussian
observation models the features are the mean and the matrix logarithm of
the covariance, so averaging features is a log-Euclidean average of
covariances and always gives a valid covariance back. The candidate
generators are:

- :code:`random_subset`: the parameters from :code:`reset()`.
- :code:`random_state_time_course`: each state is the average of the
  windows assigned to it by a sampled state time course.
- :code:`random_windows`: each state is one random window.
- :code:`kmeans`: windows are clustered with k-means and each state is a
  cluster centre.

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
from osl_dynamics.simulation import HMM

_logger = logging.getLogger("osl-dynamics")

# A candidate initialization: a name, the dataset to train on and a
# function that sets the initial parameters (or None to keep the
# parameters from reset())
Candidate = Tuple[str, tf.data.Dataset, Optional[Callable]]

# Number of batches used to score candidates with the free energy
N_SCORING_BATCHES = 20


def logm_features(covariances: np.ndarray) -> np.ndarray:
    """Vectorise symmetric positive definite matrices by their logarithm.

    The Euclidean distance between feature vectors is the Frobenius
    distance between the matrix logarithms (log-Euclidean metric).

    Parameters
    ----------
    covariances : np.ndarray
        Shape is (..., n_channels, n_channels).

    Returns
    -------
    features : np.ndarray
        Shape is (..., n_channels * (n_channels + 1) / 2).
    """
    covariances = 0.5 * (covariances + np.swapaxes(covariances, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    # Guard against windows that are too short to be full rank
    floor = np.maximum(1e-6 * eigenvalues[..., -1:], 1e-12)
    eigenvalues = np.maximum(eigenvalues, floor)
    log_cov = (eigenvectors * np.log(eigenvalues)[..., None, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )
    n_channels = covariances.shape[-1]
    i, j = np.triu_indices(n_channels)
    weights = np.where(i == j, 1.0, np.sqrt(2.0))
    return log_cov[..., i, j] * weights


def logm_features_inverse(features: np.ndarray, n_channels: int) -> np.ndarray:
    """Inverse of :func:`logm_features`.

    Parameters
    ----------
    features : np.ndarray
        Shape is (..., n_channels * (n_channels + 1) / 2).
    n_channels : int
        Number of channels.

    Returns
    -------
    covariances : np.ndarray
        Shape is (..., n_channels, n_channels).
    """
    i, j = np.triu_indices(n_channels)
    weights = np.where(i == j, 1.0, np.sqrt(2.0))
    log_cov = np.zeros(features.shape[:-1] + (n_channels, n_channels))
    log_cov[..., i, j] = features / weights
    log_cov[..., j, i] = features / weights
    eigenvalues, eigenvectors = np.linalg.eigh(log_cov)
    return (eigenvectors * np.exp(eigenvalues)[..., None, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )


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
    def _samples_per_step(self) -> int:
        """Number of data samples per step of the model's state dynamics."""
        return 1

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
        and the matrix logarithm of the covariance.

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
        features = logm_features(covariances)
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
            covariances = logm_features_inverse(features, n_channels).astype(np.float32)
            self.set_covariances(covariances, update_initializer=True)

    # ------------------------------------------------------------------
    # Multi-start loop

    def _initialization_settings(self, n_epochs, n_init, take, select):
        """Fill in settings from the config."""
        return (
            n_epochs or self.config.n_init_epochs,
            n_init or self.config.n_init,
            take or self.config.init_take,
            select or self.config.init_select,
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
        n_batches = max(round(n_total_batches * take), 1)
        _logger.info(f"Using {n_batches} out of {n_total_batches} batches")
        return training_dataset.take(n_batches)

    def _scoring_dataset(self, training_dataset, select: str):
        """A fixed set of batches to score candidates with the free energy."""
        if select == "loss":
            return None
        if select != "free_energy":
            raise ValueError("select must be 'loss' or 'free_energy'.")
        if not hasattr(self, "free_energy"):
            raise ValueError(
                f"{self.config.model_name} has no free_energy method, "
                "use select='loss'."
            )
        # cache() fixes the batches: every candidate is scored on the same
        # data even though the dataset is reshuffled each iteration
        return training_dataset.take(N_SCORING_BATCHES).cache()

    def _run_multistart(
        self,
        candidates: List[Candidate],
        n_epochs: int,
        scoring_dataset=None,
        **kwargs,
    ) -> dict:
        """Train each candidate initialization and keep the best.

        Parameters
        ----------
        candidates : list of (str, tf.data.Dataset, callable or None)
            Name, dataset to train on and a function that sets the initial
            parameters. The model is reset before each candidate.
        n_epochs : int
            Number of epochs to train each candidate.
        scoring_dataset : tf.data.Dataset, optional
            Batches to evaluate the free energy on after training. If
            :code:`None`, candidates are compared by the training loss
            of the last epoch.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method.

        Returns
        -------
        history : dict
            The training history of the best initialization.
        """
        best_score = np.inf
        for name, dataset, setup in candidates:
            _logger.info(f"Initialization {name}")
            self.reset()
            if setup is not None:
                setup()

            try:
                history = self.fit(dataset, epochs=n_epochs, **kwargs)
            except tf.errors.InvalidArgumentError as e:
                _logger.warning(e)
                _logger.warning("Training failed! Skipping initialization.")
                continue

            if scoring_dataset is None:
                score = history["loss"][-1]
            else:
                score = self.free_energy(scoring_dataset)
                _logger.info(f"Free energy: {score}")
            if score < best_score:
                best_name = name
                best_score = score
                best_history = history
                best_weights = self.get_weights()
                best_setup = setup

        if best_score == np.inf:
            raise ValueError("No valid initializations were found.")

        _logger.info(f"Using initialization {best_name}")
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
        select: Optional[str],
        kwargs: dict,
        window_length: Optional[int] = None,
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
        n_epochs, n_init, take, select : optional
            Settings, filled in from the config if :code:`None`.
        kwargs : dict
            Keyword arguments for the fit method, possibly including
            context keyword arguments.
        window_length : int, optional
            Length of the windows the candidates are built from. Only
            used with a Data object, the default is the sequence length.
        uses_features : bool, optional
            Does the method set parameters from window features? If so
            the model must support that and the windows should be long
            enough.

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
        n_epochs, n_init, take, select = self._initialization_settings(
            n_epochs, n_init, take, select
        )
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
        scoring_dataset = self._scoring_dataset(training_dataset, select)

        # Dataset to build candidates from
        sequence_length = self._dataset_sequence_length()
        length = window_length or sequence_length
        if uses_features and length < self._min_window_samples:
            _logger.warning(
                f"Windows of {length} samples are short for {self.config.n_channels} "
                "channels, the initial parameters will be poorly estimated. "
                "Consider passing a longer window_length."
            )
        if window_length is None or window_length == sequence_length:
            window_dataset = subset
        elif hasattr(training_data, "dataset"):
            window_dataset = training_data.dataset(
                window_length, self.config.batch_size, shuffle=True
            )
        else:
            raise ValueError(
                "window_length can only be changed when a Data object is passed."
            )

        candidates = make_candidates(window_dataset, n_init)
        candidates = [(name_, subset, setup) for name_, setup in candidates]

        with self._initialization_context(**context_kwargs):
            return self._run_multistart(
                candidates, n_epochs, scoring_dataset=scoring_dataset, **kwargs
            )

    # ------------------------------------------------------------------
    # Building candidates from windows

    def _collect_windows(self, dataset, n_windows: int) -> List[np.ndarray]:
        """Take the first n_windows sequences from a (shuffled) dataset."""
        windows = []
        for batch in dataset:
            windows.extend(np.asarray(batch["data"]))
            if len(windows) >= n_windows:
                break
        if len(windows) < n_windows:
            _logger.warning(
                f"Only {len(windows)} windows available, {n_windows} requested."
            )
        return windows[:n_windows]

    def _collect_features(self, dataset, max_windows: int) -> np.ndarray:
        """Features of up to max_windows sequences from a (shuffled) dataset."""
        features = []
        n_windows = 0
        for batch in dataset:
            windows = list(np.asarray(batch["data"]))
            features.append(self._window_features(windows))
            n_windows += len(windows)
            if n_windows >= max_windows:
                break
        return np.concatenate(features)[:max_windows]

    def _collect_ordered_features(self, datasets, max_windows: int):
        """Features of every stride-th window of each session, in time order.

        Parameters
        ----------
        datasets : list of tf.data.Dataset
            Unshuffled dataset for each session.
        max_windows : int
            Maximum number of windows to keep in total.

        Returns
        -------
        features : np.ndarray
            Shape is (n_windows, n_features).
        session : np.ndarray
            Session index of each window.
        stride : int
            Number of windows between consecutive kept windows.
        """
        n_total = sum(d.osld_metadata["n_sequences"] for d in datasets)
        stride = max(1, int(np.ceil(n_total / max_windows)))
        features = []
        session = []
        n = 0
        for i, dataset in enumerate(datasets):
            for batch in dataset:
                windows = list(np.asarray(batch["data"]))
                keep = [w for j, w in enumerate(windows) if (n + j) % stride == 0]
                n += len(windows)
                if keep:
                    features.append(self._window_features(keep))
                    session.extend([i] * len(keep))
        return np.concatenate(features), np.array(session), stride

    def _trans_prob_from_labels(
        self, labels: np.ndarray, session: np.ndarray, n_samples_per_step: int
    ) -> np.ndarray:
        """Transition probability matrix from a sequence of window labels.

        Transitions between consecutive windows of the same session are
        counted (with add-one smoothing) and the window-scale matrix is
        converted to a per-sample matrix assuming at most one transition
        per window step.

        Parameters
        ----------
        labels : np.ndarray
            State label of each window.
        session : np.ndarray
            Session index of each window.
        n_samples_per_step : int
            Number of steps of the state dynamics between the start of
            consecutive windows.

        Returns
        -------
        trans_prob : np.ndarray
            Shape is (n_states, n_states).
        """
        n_states = self._n_states
        counts = np.ones((n_states, n_states))
        same_session = session[1:] == session[:-1]
        np.add.at(counts, (labels[:-1][same_session], labels[1:][same_session]), 1)
        window_trans_prob = counts / counts.sum(axis=1, keepdims=True)
        trans_prob = np.eye(n_states) + (window_trans_prob - np.eye(n_states)) / (
            n_samples_per_step
        )
        trans_prob = np.maximum(trans_prob, 1e-6)
        return (trans_prob / trans_prob.sum(axis=1, keepdims=True)).astype(np.float32)

    def _state_time_course_features(self, dataset, stay_prob: float) -> np.ndarray:
        """Average features of the windows a sampled state time course
        assigns to each state."""
        n_states = self._n_states
        total = None
        counts = np.zeros(n_states)
        for batch in dataset:
            data = np.concatenate(batch["data"])
            if data.shape[0] < n_states * self._min_window_samples:
                raise ValueError(
                    f"Not enough time points in a batch ({data.shape[0]}) to "
                    f"estimate {n_states} states with {self.config.n_channels} "
                    "channels. Increase batch_size or sequence_length."
                )
            labels = self._sample_labels(data.shape[0], stay_prob)
            present = [
                k
                for k in range(n_states)
                if np.sum(labels == k) >= self._min_window_samples
            ]
            if not present:
                continue
            features = self._window_features([data[labels == k] for k in present])
            if total is None:
                total = np.zeros((n_states, features.shape[1]))
            total[present] += features
            counts[present] += 1
        if np.any(counts == 0):
            raise ValueError(
                "random_state_time_course_initialization sampled a state time "
                "course where a state never activates. Try increasing the "
                "batch_size or sequence_length, or use another method."
            )
        return total / counts[:, None]

    # ------------------------------------------------------------------
    # Public methods

    def random_subset_initialization(
        self,
        training_data,
        n_epochs: Optional[int] = None,
        n_init: Optional[int] = None,
        take: Optional[float] = None,
        select: Optional[str] = None,
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
        select : str, optional
            How to pick the best initialization: :code:`'loss'` (training
            loss of the last epoch) or :code:`'free_energy'` (free energy
            on a fixed set of batches). By default we use the value passed
            in the config.
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
            select,
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
        select: Optional[str] = None,
        stay_prob: float = 0.9,
        **kwargs,
    ) -> Optional[dict]:
        """Random state time course initialization.

        A state time course is sampled and each state's parameters are set
        from the data assigned to it. Note, with a lot of data every state
        ends up close to the average of the whole dataset, so the
        candidates differ little. The model is then trained for a few
        epochs and the best candidate is kept.

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
        select : str, optional
            How to pick the best initialization: :code:`'loss'` or
            :code:`'free_energy'`. By default we use the value passed in
            the config.
        stay_prob : float, optional
            Stay probability (diagonal for the transition probability
            matrix). Other states have uniform probability. Ignored by
            models that learn a transition probability matrix, which
            sample from their initial matrix.
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
                features = self._state_time_course_features(dataset, stay_prob)
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
            select,
            kwargs,
            uses_features=True,
        )

    def set_random_state_time_course_initialization(
        self, training_dataset: tf.data.Dataset, stay_prob: float = 0.9
    ) -> None:
        """Sets the observation model based on a random state time course.

        Parameters
        ----------
        training_dataset : tf.data.Dataset
            Training data.
        stay_prob : float, optional
            Stay probability (diagonal for the transition probability
            matrix). Other states have uniform probability.
        """
        _logger.info("Setting parameters from a random state time course")
        features = self._state_time_course_features(training_dataset, stay_prob)
        self._set_parameters_from_features(features)

    def random_windows_initialization(
        self,
        training_data,
        n_epochs: Optional[int] = None,
        n_init: Optional[int] = None,
        take: Optional[float] = None,
        select: Optional[str] = None,
        window_length: Optional[int] = None,
        **kwargs,
    ) -> Optional[dict]:
        """Random windows initialization.

        Each state's parameters are set from one randomly chosen window of
        data, so the states start genuinely different. The model is then
        trained for a few epochs and the best candidate is kept.

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
        select : str, optional
            How to pick the best initialization: :code:`'loss'` or
            :code:`'free_energy'`. By default we use the value passed in
            the config.
        window_length : int, optional
            Number of samples in each window. Default is the sequence
            length. Can only be changed when a Data object is passed.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method. Variational models also
            accept :code:`n_kl_annealing_epochs`.

        Returns
        -------
        history : dict
            The training history of the best initialization.
        """
        n_states = self._n_states

        def make_candidates(dataset, n_init):
            windows = self._collect_windows(dataset, n_init * n_states)
            if len(windows) < n_states:
                raise ValueError(
                    f"Only {len(windows)} windows are available but {n_states} "
                    "are needed for one initialization. Increase take or "
                    "reduce window_length."
                )
            features = self._window_features(windows)
            candidates = []
            for n in range(len(windows) // n_states):
                f = features[n * n_states : (n + 1) * n_states]
                candidates.append(
                    (str(n), lambda f=f: self._set_parameters_from_features(f))
                )
            return candidates

        return self._multistart(
            "Random windows",
            training_data,
            make_candidates,
            n_epochs,
            n_init,
            take,
            select,
            kwargs,
            window_length=window_length,
            uses_features=True,
        )

    def kmeans_initialization(
        self,
        training_data,
        n_epochs: Optional[int] = None,
        n_init: Optional[int] = None,
        take: Optional[float] = None,
        select: Optional[str] = None,
        window_length: Optional[int] = None,
        max_windows: int = 20000,
        init_trans_prob: bool = False,
        **kwargs,
    ) -> Optional[dict]:
        """K-means initialization.

        Windows of data are clustered with k-means (one cluster per state)
        and each state's parameters are set from its cluster centre. Each
        candidate is a different k-means solution. The model is then
        trained for a few epochs and the best candidate is kept.

        For Gaussian observation models the clustering is done on the
        matrix logarithm of the window covariances (and the window means
        if means are learnt), so the cluster centres are log-Euclidean
        averages of covariances.

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
        select : str, optional
            How to pick the best initialization: :code:`'loss'` or
            :code:`'free_energy'`. By default we use the value passed in
            the config.
        window_length : int, optional
            Number of samples in each window. Default is the sequence
            length. Can only be changed when a Data object is passed.
        max_windows : int, optional
            Maximum number of windows to cluster. Windows are taken at
            random from the dataset, or evenly spaced in time if
            :code:`init_trans_prob=True`.
        init_trans_prob : bool, optional
            Should we also set the initial transition probability matrix
            from the transitions between the cluster labels of consecutive
            windows? Needs a Data object and a model with a transition
            probability matrix.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method. Variational models also
            accept :code:`n_kl_annealing_epochs`.

        Returns
        -------
        history : dict
            The training history of the best initialization.
        """
        from sklearn.cluster import KMeans

        n_states = self._n_states
        if init_trans_prob:
            if not hasattr(self, "set_trans_prob"):
                raise ValueError(
                    f"{self.config.model_name} has no transition probability matrix."
                )
            if not hasattr(training_data, "dataset"):
                raise ValueError("init_trans_prob=True needs a Data object.")

        def make_candidates(dataset, n_init):
            _logger.info("Clustering windows")
            if init_trans_prob:
                # Windows in time order so transitions can be counted
                length = window_length or self._dataset_sequence_length()
                ordered = training_data.dataset(
                    length, self.config.batch_size, shuffle=False, concatenate=False
                )
                features, session, stride = self._collect_ordered_features(
                    ordered, max_windows
                )
            else:
                features = self._collect_features(dataset, max_windows)

            candidates = []
            for n in range(n_init):
                kmeans = KMeans(
                    n_clusters=n_states,
                    n_init=1,
                    random_state=np.random.randint(2**31 - 1),
                ).fit(features)
                centres = kmeans.cluster_centers_
                if init_trans_prob:
                    trans_prob = self._trans_prob_from_labels(
                        kmeans.labels_,
                        session,
                        max(1, stride * length // self._samples_per_step),
                    )
                else:
                    trans_prob = None

                def setup(centres=centres, trans_prob=trans_prob):
                    self._set_parameters_from_features(centres)
                    if trans_prob is not None:
                        self.set_trans_prob(trans_prob, update_initializer=True)

                candidates.append((str(n), setup))
            return candidates

        return self._multistart(
            "K-means",
            training_data,
            make_candidates,
            n_epochs,
            n_init,
            take,
            select,
            kwargs,
            window_length=window_length,
            uses_features=True,
        )

    def single_subject_initialization(
        self,
        training_data,
        n_epochs: Optional[int] = None,
        n_init: Optional[int] = None,
        select: Optional[str] = None,
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
        select : str, optional
            How to pick the best initialization: :code:`'loss'` or
            :code:`'free_energy'`. By default we use the value passed in
            the config.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method. Variational models also
            accept :code:`n_kl_annealing_epochs`.

        Returns
        -------
        history : dict
            The training history of the best initialization.
        """
        n_epochs, n_init, _, select = self._initialization_settings(
            n_epochs, n_init, None, select
        )
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
        scoring_dataset = self._scoring_dataset(
            dtf.concatenate_datasets(datasets), select
        )

        # Pick n_init subjects at random
        subjects = np.random.choice(len(datasets), n_init, replace=False)
        candidates = [(f"subject {s}", datasets[s], None) for s in subjects]

        with self._initialization_context(**context_kwargs):
            return self._run_multistart(
                candidates, n_epochs, scoring_dataset=scoring_dataset, **kwargs
            )
