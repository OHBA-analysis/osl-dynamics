"""Classes for simulating Hidden Semi-Markov Models (HSMMs).

"""

import logging

import numpy as np
from vrad.array_ops import get_one_hot
from vrad.simulation import Simulation
from vrad.utils.decorators import auto_repr, auto_yaml

_logger = logging.getLogger("VRAD")


class HSMMSimulation(Simulation):
    """Hidden Semi-Markov Model Simulation.

    We sample the state using a transition probability matrix with zero
    probability for self-transitions. The lifetime of each state is sampled
    from a Gamma distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    gamma_shape : float
        Shape parameter for the gamma distribution of state lifetimes.
    gamma_scale : float
        Scale parameter for the gamma distribution of state lifetimes.
    off_diagonal_trans_prob : np.ndarray
        Transition probabilities for out of state transitions.
    full_trans_prob : np.ndarray
        A transition probability matrix, the diagonal of which will be ignored.
    n_states : int
        Number of states. Inferred from the transition probability matrix
        or means/covariances if None.
    n_channels : int
        Number of channels in the observation model.
        Inferred from means or covariances if None.
    means : np.ndarray
        Mean vector for each state, shape should be (n_states, n_channels).
    covariances : numpy.ndarray
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels).
    zero_means : bool
        If True, means are set to zero, otherwise they are sampled from a normal
        distribution.
    random_covariances : bool
        Should we simulate random covariances? False gives structured covariances.
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    simulate : bool
        Should data be simulated? Can be called using .simulate later.
    random_seed : int
        Seed for reproducibility.
    """

    @auto_yaml
    @auto_repr
    def __init__(
        self,
        n_samples: int,
        gamma_shape: float,
        gamma_scale: float,
        off_diagonal_trans_prob: np.ndarray = None,
        full_trans_prob: np.ndarray = None,
        n_states: int = None,
        n_channels: int = None,
        means: np.ndarray = None,
        covariances: np.ndarray = None,
        zero_means: bool = None,
        random_covariances: bool = None,
        observation_error: float = 0.0,
        simulate: bool = True,
        random_seed: int = None,
    ):
        # Validation
        if off_diagonal_trans_prob is not None and full_trans_prob is not None:
            raise ValueError(
                "Exactly one of off_diagonal_trans_prob and full_trans_prob "
                "must be specified."
            )

        # Get the number of states
        if n_states is None:
            for arg in [off_diagonal_trans_prob, full_trans_prob, means, covariances]:
                if arg is not None:
                    n_states = arg.shape[0]
                    break
            if n_states is None:
                raise ValueError(
                    "If n_states is not passed, either off_diagonal_trans_prob, "
                    + "full_trans_prob, means or covariances must be passed."
                )

        self.off_diagonal_trans_prob = off_diagonal_trans_prob
        self.full_trans_prob = full_trans_prob

        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            means=means,
            covariances=covariances,
            zero_means=zero_means,
            random_covariances=random_covariances,
            observation_error=observation_error,
            simulate=simulate,
            random_seed=random_seed,
        )

    def construct_off_diagonal_trans_prob(self):
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

        with np.printoptions(linewidth=np.nan):
            _logger.info(
                f"off_diagonal_trans_prob is:\n{str(self.off_diagonal_trans_prob)}"
            )

    def generate_states(self):
        self.construct_off_diagonal_trans_prob()
        cumsum_off_diagonal_trans_prob = np.cumsum(self.off_diagonal_trans_prob, axis=1)
        alpha = np.zeros(self.n_samples, dtype=np.int)

        gamma_sample = self._rng.gamma
        random_sample = self._rng.uniform
        current_state = self._rng.integers(0, self.n_states)
        current_position = 0

        while current_position < len(alpha):
            state_lifetime = np.round(
                gamma_sample(shape=self.gamma_shape, scale=self.gamma_scale)
            ).astype(np.int)
            alpha[current_position : current_position + state_lifetime] = current_state

            rand = random_sample()
            current_state = np.argmin(
                cumsum_off_diagonal_trans_prob[current_state] < rand
            )
            current_position += state_lifetime

        one_hot_alpha = get_one_hot(alpha, n_states=self.n_states)

        return one_hot_alpha


class MixedHSMMSimulation(HSMMSimulation):
    """Hidden Semi-Markov Model Simulation with a mixture of states at each time point.

    Each mixture of states has it's own row/column in the transition probability matrix.
    The lifetime of each state mixture is sampled from a Gamma distribution.

    state_mixing_vectors is a 2D numpy array containing mixtures of the
    the states that can be simulated, e.g. with n_states=3 we could have
    state_mixing_vectors=[[0.5, 0.5, 0], [0.1, 0, 0.9]]

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    mixed_state_vectors : np.ndarray
        Vectors containing mixing factors for mixed states.
    gamma_shape : float
        Shape parameter for the gamma distribution of state lifetimes.
    gamma_scale : float
        Scale parameter for the gamma distribution of state lifetimes.
    off_diagonal_trans_prob : np.ndarray
        Transition probabilities for out of state transitions.
    full_trans_prob : np.ndarray
        A transition probability matrix, the diagonal of which will be ignored.
    n_states : int
        Number of states. Inferred from the mixed_state_vectors if None.
    n_channels : int
        Number of channels in the observation model.
        Inferred from means or covariances if None.
    means : np.ndarray
        Mean vector for each state, shape should be (n_states, n_channels).
    covariances : numpy.ndarray
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels).
    zero_means : bool
        If True, means are set to zero, otherwise they are sampled from a normal
        distribution.
    random_covariances : bool
        Should we simulate random covariances? False gives structured covariances.
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    simulate : bool
        Should data be simulated? Can be called using .simulate later.
    random_seed : int
        Seed for reproducibility.
    """

    @auto_yaml
    @auto_repr
    def __init__(
        self,
        n_samples: int,
        mixed_state_vectors: np.ndarray,
        gamma_shape: float,
        gamma_scale: float,
        off_diagonal_trans_prob: np.ndarray = None,
        full_trans_prob: np.ndarray = None,
        n_states: int = None,
        n_channels: int = None,
        means: np.ndarray = None,
        covariances: np.ndarray = None,
        zero_means: bool = None,
        random_covariances: bool = None,
        observation_error: float = 0.0,
        simulate: bool = True,
        random_seed: int = None,
    ):
        self.mixed_state_vectors = mixed_state_vectors
        self.n_mixed_states = mixed_state_vectors.shape[0]

        if n_states is None:
            # Get the number of states from the mixed state vectors
            n_states = mixed_state_vectors.shape[1]

        self.construct_state_vectors(n_states)

        super().__init__(
            n_samples=n_samples,
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            off_diagonal_trans_prob=off_diagonal_trans_prob,
            full_trans_prob=full_trans_prob,
            n_states=n_states,
            n_channels=n_channels,
            means=means,
            covariances=covariances,
            zero_means=zero_means,
            random_covariances=random_covariances,
            observation_error=observation_error,
            simulate=simulate,
            random_seed=random_seed,
        )

    def construct_state_vectors(self, n_states):
        non_mixed_state_vectors = get_one_hot(np.arange(n_states))
        self.state_vectors = np.append(
            non_mixed_state_vectors, self.mixed_state_vectors, axis=0
        )

    def construct_off_diagonal_trans_prob(self):
        if self.off_diagonal_trans_prob is None:
            self.off_diagonal_trans_prob = np.ones(
                [
                    self.n_states + self.n_mixed_states,
                    self.n_states + self.n_mixed_states,
                ]
            )

        np.fill_diagonal(self.off_diagonal_trans_prob, 0)
        self.off_diagonal_trans_prob = (
            self.off_diagonal_trans_prob
            / self.off_diagonal_trans_prob.sum(axis=1)[:, None]
        )

        with np.printoptions(linewidth=np.nan):
            _logger.info(
                f"off_diagonal_trans_prob is:\n{str(self.off_diagonal_trans_prob)}"
            )

    def generate_states(self):
        self.construct_off_diagonal_trans_prob()
        cumsum_off_diagonal_trans_prob = np.cumsum(self.off_diagonal_trans_prob, axis=1)
        alpha = np.zeros([self.n_samples, self.n_states])

        gamma_sample = self._rng.gamma
        random_sample = self._rng.uniform
        current_state = self._rng.integers(0, self.n_states)
        current_position = 0

        while current_position < len(alpha):
            state_lifetime = np.round(
                gamma_sample(shape=self.gamma_shape, scale=self.gamma_scale)
            ).astype(np.int)

            alpha[
                current_position : current_position + state_lifetime
            ] = self.state_vectors[current_state]

            rand = random_sample()
            current_state = np.argmin(
                cumsum_off_diagonal_trans_prob[current_state] < rand
            )
            current_position += state_lifetime

        return alpha
