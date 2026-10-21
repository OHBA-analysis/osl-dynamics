"""Initialization of model parameters before the full training.

Every initialization method trains a few candidate initializations for a
short time and keeps the one with the best loss. The multi-start loop
(:meth:`InitializationMixin._run_multistart`) is shared by all models.
A model customises it through three hooks:

- :meth:`InitializationMixin._initialization_context`: settings that only
  apply while initializing (e.g. the number of KL annealing epochs).
- :meth:`InitializationMixin._restore_initialization`: how to load the
  weights of the best candidate.
- :meth:`InitializationMixin.set_random_state_time_course_initialization`:
  how to set the observation model from a sampled state time course.
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
# function that sets the initial parameters given that dataset (or None
# to keep the parameters from reset())
Candidate = Tuple[str, tf.data.Dataset, Optional[Callable]]


class InitializationMixin:
    """Initialization methods shared by all models.

    Expects the class to provide :code:`config`, :code:`reset`, :code:`fit`,
    :code:`get_weights`, :code:`set_weights` and :code:`make_dataset`.
    """

    # Keyword arguments of the public methods that configure
    # _initialization_context rather than fit
    _initialization_context_keys = ()

    @contextmanager
    def _initialization_context(self, **kwargs):
        """Settings that only apply while initializing. No-op by default."""
        yield

    def _restore_initialization(self, weights: list) -> None:
        """Load the weights of the best initialization."""
        self.reset()
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
        n_batches = max(round(n_total_batches * take), 1)
        _logger.info(f"Using {n_batches} out of {n_total_batches} batches")
        return training_dataset.take(n_batches)

    def _run_multistart(
        self, candidates: List[Candidate], n_epochs: int, **kwargs
    ) -> dict:
        """Train each candidate initialization and keep the best.

        Parameters
        ----------
        candidates : list of (str, tf.data.Dataset, callable or None)
            Name, dataset to train on and a function that sets the initial
            parameters given that dataset. The model is reset before each
            candidate.
        n_epochs : int
            Number of epochs to train each candidate.
        kwargs : keyword arguments, optional
            Keyword arguments for the fit method.

        Returns
        -------
        history : dict
            The training history of the best initialization.
        """
        best_loss = np.inf
        for name, dataset, setup in candidates:
            _logger.info(f"Initialization {name}")
            self.reset()
            if setup is not None:
                setup(dataset)

            try:
                history = self.fit(dataset, epochs=n_epochs, **kwargs)
            except tf.errors.InvalidArgumentError as e:
                _logger.warning(e)
                _logger.warning("Training failed! Skipping initialization.")
                continue

            loss = history["loss"][-1]
            if loss < best_loss:
                best_name = name
                best_loss = loss
                best_history = history
                best_weights = self.get_weights()

        if best_loss == np.inf:
            raise ValueError("No valid initializations were found.")

        _logger.info(f"Using initialization {best_name}")
        self._restore_initialization(best_weights)

        return best_history

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
            Keyword arguments for the fit method. Variational models also
            accept :code:`n_kl_annealing_epochs`.

        Returns
        -------
        history : dict
            The training history of the best initialization.
        """
        n_epochs, n_init, take = self._initialization_settings(n_epochs, n_init, take)
        if not n_init:
            _logger.info(
                "Number of initializations was set to zero. Skipping initialization."
            )
            return
        context_kwargs, kwargs = self._split_initialization_kwargs(kwargs)

        _logger.info("Random subset initialization")
        training_dataset = self.make_dataset(
            training_data, shuffle=True, concatenate=True
        )
        subset = self._subset(training_dataset, take)
        candidates = [(str(n), subset, None) for n in range(n_init)]

        with self._initialization_context(**context_kwargs):
            return self._run_multistart(candidates, n_epochs, **kwargs)

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
        **kwargs,
    ) -> Optional[dict]:
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
        n_epochs, n_init, take = self._initialization_settings(n_epochs, n_init, take)
        if not n_init:
            _logger.info(
                "Number of initializations was set to zero. Skipping initialization."
            )
            return
        context_kwargs, kwargs = self._split_initialization_kwargs(kwargs)

        _logger.info("Random state time course initialization")
        training_dataset = self.make_dataset(
            training_data, shuffle=True, concatenate=True
        )
        subset = self._subset(training_dataset, take)

        def setup(dataset):
            self.set_random_state_time_course_initialization(dataset, stay_prob)

        candidates = [(str(n), subset, setup) for n in range(n_init)]

        with self._initialization_context(**context_kwargs):
            return self._run_multistart(candidates, n_epochs, **kwargs)

    def single_subject_initialization(
        self,
        training_data,
        n_epochs: Optional[int] = None,
        n_init: Optional[int] = None,
        **kwargs,
    ) -> Optional[dict]:
        """Single subject initialization.

        Pick a subject at random, train a model, repeat a few times. Use
        the weights from the best model (judged using the final loss).

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

    def set_random_state_time_course_initialization(
        self, training_dataset: tf.data.Dataset, stay_prob: float = 0.9
    ) -> None:
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

        n_states = self._n_states
        n_channels = self.config.n_channels

        # Mean and covariance for each state
        means = np.zeros([n_states, n_channels], dtype=np.float32)
        covariances = np.zeros([n_states, n_channels, n_channels], dtype=np.float32)

        n_batches = 0
        for batch in training_dataset:
            # Concatenate all the sequences in this batch
            data = np.concatenate(batch["data"])

            if data.shape[0] < 2 * n_channels:
                raise ValueError(
                    "Not enough time points in batch, "
                    "increase batch_size or sequence_length"
                )

            # Sample a state time course using the initial transition
            # probability matrix
            stc = self._sample_state_time_course(data.shape[0], stay_prob)

            # Make sure each state activates
            non_active_states = np.sum(stc, axis=0) < 2 * n_channels
            if np.any(non_active_states):
                for _ in range(100):
                    new_stc = self._sample_state_time_course(data.shape[0], stay_prob)
                    new_active_states = np.sum(new_stc, axis=0) != 0
                    for j in range(n_states):
                        if non_active_states[j] and new_active_states[j]:
                            stc[:, j] = new_stc[:, j]
                    non_active_states = np.sum(stc, axis=0) < 2 * n_channels
                    if not np.any(non_active_states):
                        break

            if np.any(non_active_states):
                # Some states still haven't activated
                raise ValueError(
                    "random_state_time_course_initialization can't simulate a state "
                    "time course where each state activates.\n"
                    "Try increasing the batch_size or sequence_length.\n"
                    "Or switch to using model.random_subset_initialization() instead."
                )

            # Calculate the mean/covariance for each state for this batch
            m = []
            C = []
            for j in range(n_states):
                x = data[stc[:, j] == 1]
                mu = np.mean(x, axis=0)
                if n_channels == 1:
                    sigma = np.var(x).reshape(1, 1)
                else:
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
