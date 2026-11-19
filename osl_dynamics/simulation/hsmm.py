"""Hidden Semi-Markov Models (HSMMs)."""

from typing import Optional, Union

import numpy as np

from osl_dynamics.simulation.base import Simulation
from osl_dynamics.simulation.obs_mod import MVN
from osl_dynamics.utils import array_ops


class HSMM:
    """Hidden Semi-Markov Model (HSMM) state model.

    The state lifetimes follow a Gamma distribution. The next state is
    sampled from a transition probability matrix with zero probability for
    self-transitions.

    Parameters
    ----------
    gamma_shape : float
        Shape parameter for the Gamma distribution of state lifetimes.
    gamma_scale : float
        Scale parameter for the Gamma distribution of state lifetimes.
    off_diagonal_trans_prob : np.ndarray, optional
        Transition probabilities for out of state transitions.
    full_trans_prob : np.ndarray, optional
        A transition probability matrix, the diagonal of which will be ignored.
    n_states : int, optional
        Number of states. Needed if no transition probability matrix is
        passed.
    """

    def __init__(
        self,
        gamma_shape: float,
        gamma_scale: float,
        off_diagonal_trans_prob: Optional[np.ndarray] = None,
        full_trans_prob: Optional[np.ndarray] = None,
        n_states: Optional[int] = None,
    ) -> None:
        # Validation
        if off_diagonal_trans_prob is not None and full_trans_prob is not None:
            raise ValueError(
                "Exactly one of off_diagonal_trans_prob and full_trans_prob "
                "must be specified."
            )

        # Get the number of states from trans_prob
        if off_diagonal_trans_prob is not None:
            self.n_states = off_diagonal_trans_prob.shape[0]
        elif full_trans_prob is not None:
            self.n_states = full_trans_prob.shape[0]

        # Both off_diagonal_trans_prob and full_trans_prob are None
        elif n_states is None:
            raise ValueError(
                "If off_diagonal_trans_prob and full_trans_prob are not given, "
                "n_states must be passed."
            )
        else:
            self.n_states = n_states

        self.off_diagonal_trans_prob = off_diagonal_trans_prob
        self.full_trans_prob = full_trans_prob
        self.construct_off_diagonal_trans_prob()

        # Parameters of the lifetime distribution
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale

    def construct_off_diagonal_trans_prob(self) -> None:
        if (self.off_diagonal_trans_prob is None) and (self.full_trans_prob is None):
            self.off_diagonal_trans_prob = np.ones([self.n_states, self.n_states])

        if self.full_trans_prob is not None:
            self.off_diagonal_trans_prob = (
                self.full_trans_prob / self.full_trans_prob.sum(axis=1)[:, None]
            )

        np.fill_diagonal(self.off_diagonal_trans_prob, 0)
        self.off_diagonal_trans_prob = (
            self.off_diagonal_trans_prob
            / self.off_diagonal_trans_prob.sum(axis=1)[:, None]
        )

    def generate_state_sequence(self, n_samples: int) -> np.ndarray:
        """Generate a sequence of states.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        states : np.ndarray
            State at each sample (0 to n_states - 1). Shape is (n_samples,).
        """
        cumsum_off_diagonal_trans_prob = np.cumsum(
            self.off_diagonal_trans_prob,
            axis=1,
        )
        states = np.zeros(n_samples, dtype=int)

        current_state = np.random.randint(0, self.n_states)
        current_position = 0

        while current_position < n_samples:
            state_lifetime = np.round(
                np.random.gamma(shape=self.gamma_shape, scale=self.gamma_scale)
            ).astype(int)

            states[current_position : current_position + state_lifetime] = current_state

            current_state = np.argmin(
                cumsum_off_diagonal_trans_prob[current_state] < np.random.uniform()
            )
            current_position += state_lifetime

        return states

    def generate_states(self, n_samples: int) -> np.ndarray:
        """Generate a state time course.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        state_time_course : np.ndarray
            One-hot state time course. Shape is (n_samples, n_states).
        """
        states = self.generate_state_sequence(n_samples)
        return array_ops.get_one_hot(states, n_states=self.n_states)


class MixedHSMM:
    """HSMM where some of the states are mixtures of modes.

    The HSMM (:code:`hsmm`) has one state for each mode (active on its own)
    followed by one state for each mixture in :code:`mixed_state_vectors`.
    The mode time course is the mixing vector of the active HSMM state.

    Parameters
    ----------
    gamma_shape : float
        Shape parameter for the Gamma distribution of state lifetimes.
    gamma_scale : float
        Scale parameter for the Gamma distribution of state lifetimes.
    mixed_state_vectors : np.ndarray
        Mixing vector of each mixed state, e.g. with 3 modes
        :code:`[[0.5, 0.5, 0], [0.1, 0, 0.9]]`.
        Shape is (n_mixed_states, n_modes).
    off_diagonal_trans_prob : np.ndarray, optional
        Transition probabilities for out of state transitions, shape
        (n_modes + n_mixed_states, n_modes + n_mixed_states).
    full_trans_prob : np.ndarray, optional
        A transition probability matrix, the diagonal of which will be ignored.
    """

    def __init__(
        self,
        gamma_shape: float,
        gamma_scale: float,
        mixed_state_vectors: np.ndarray,
        off_diagonal_trans_prob: Optional[np.ndarray] = None,
        full_trans_prob: Optional[np.ndarray] = None,
    ) -> None:
        self.mixed_state_vectors = np.asarray(mixed_state_vectors)
        self.n_mixed_states, self.n_modes = self.mixed_state_vectors.shape
        self.state_vectors = np.append(
            np.eye(self.n_modes), self.mixed_state_vectors, axis=0
        )
        self.hsmm = HSMM(
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            off_diagonal_trans_prob=off_diagonal_trans_prob,
            full_trans_prob=full_trans_prob,
            n_states=self.n_modes + self.n_mixed_states,
        )
        self.off_diagonal_trans_prob = self.hsmm.off_diagonal_trans_prob

    def generate_modes(self, n_samples: int) -> np.ndarray:
        """Generate a mode time course.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        mode_time_course : np.ndarray
            Mixing coefficients. Shape is (n_samples, n_modes).
        """
        return self.state_vectors[self.hsmm.generate_state_sequence(n_samples)]


class HSMM_MVN(Simulation):
    """HSMM with a multivariate normal observation model.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    gamma_shape : float
        Shape parameter for the Gamma distribution of state lifetimes.
    gamma_scale : float
        Scale parameter for the Gamma distribution of state lifetimes.
    off_diagonal_trans_prob : np.ndarray, optional
        Transition probabilities for out of state transitions.
    full_trans_prob : np.ndarray, optional
        A transition probability matrix, the diagonal of which will be ignored.
    means : np.ndarray or str, optional
        Mean vector for each state, shape should be (n_states, n_channels).
        Or :code:`'zero'` or :code:`'random'`.
    covariances : numpy.ndarray or str, optional
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels). Or :code:`'random'`.
    n_states : int, optional
        Number of states.
    n_channels : int, optional
        Number of channels in the observation model.
    observation_error : float, optional
        Standard deviation of random noise to be added to the observations.
    """

    def __init__(
        self,
        n_samples: int,
        gamma_shape: float,
        gamma_scale: float,
        off_diagonal_trans_prob: Optional[np.ndarray] = None,
        full_trans_prob: Optional[np.ndarray] = None,
        means: Optional[Union[np.ndarray, str]] = None,
        covariances: Optional[Union[np.ndarray, str]] = None,
        n_states: Optional[int] = None,
        n_channels: Optional[int] = None,
        observation_error: float = 0.0,
    ) -> None:
        obs_model = MVN(
            means=means,
            covariances=covariances,
            n_channels=n_channels,
            observation_error=observation_error,
        )
        obs_model.initialize(n_states)
        self.hsmm = HSMM(
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            off_diagonal_trans_prob=off_diagonal_trans_prob,
            full_trans_prob=full_trans_prob,
            n_states=obs_model._n,
        )
        super().__init__(self.hsmm, obs_model, n_samples)


class MixedHSMM_MVN(Simulation):
    """HSMM with mixed states and a multivariate normal observation model.

    Each mixture of modes has its own row/column in the transition
    probability matrix and its lifetime is sampled from a Gamma
    distribution. See :class:`MixedHSMM`.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    gamma_shape : float
        Shape parameter for the Gamma distribution of state lifetimes.
    gamma_scale : float
        Scale parameter for the Gamma distribution of state lifetimes.
    mixed_state_vectors : np.ndarray
        Mixing vector of each mixed state. Shape is (n_mixed_states, n_modes).
    off_diagonal_trans_prob : np.ndarray, optional
        Transition probabilities for out of state transitions.
    full_trans_prob : np.ndarray, optional
        A transition probability matrix, the diagonal of which will be ignored.
    means : np.ndarray or str, optional
        Mean vector for each mode, shape should be (n_modes, n_channels).
        Or :code:`'zero'` or :code:`'random'`.
    covariances : numpy.ndarray or str, optional
        Covariance matrix for each mode, shape should be (n_modes, n_channels,
        n_channels). Or :code:`'random'`.
    n_channels : int, optional
        Number of channels in the observation model.
    observation_error : float, optional
        Standard deviation of random noise to be added to the observations.
    """

    def __init__(
        self,
        n_samples: int,
        gamma_shape: float,
        gamma_scale: float,
        mixed_state_vectors: np.ndarray,
        off_diagonal_trans_prob: Optional[np.ndarray] = None,
        full_trans_prob: Optional[np.ndarray] = None,
        means: Optional[Union[np.ndarray, str]] = None,
        covariances: Optional[Union[np.ndarray, str]] = None,
        n_channels: Optional[int] = None,
        observation_error: float = 0.0,
    ) -> None:
        self.hsmm = MixedHSMM(
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            mixed_state_vectors=mixed_state_vectors,
            off_diagonal_trans_prob=off_diagonal_trans_prob,
            full_trans_prob=full_trans_prob,
        )
        obs_model = MVN(
            means=means,
            covariances=covariances,
            n_channels=n_channels,
            observation_error=observation_error,
        )
        super().__init__(self.hsmm, obs_model, n_samples)
