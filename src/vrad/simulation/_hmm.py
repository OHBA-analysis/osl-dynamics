"""Classes for simulation Hidden Markov Models (HMMs).

"""

import numpy as np
from vrad.array_ops import get_one_hot
from vrad.simulation import Simulation
from vrad.utils.decorators import auto_repr, auto_yaml


class HMMSimulation(Simulation):
    """Class providing methods to simulate an HMM.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    trans_prob : numpy.ndarray
        Transition probability matrix.
    n_channels : int
        Number of channels in the observation model. Inferred from covariances if None.
    means : np.ndarray
        Mean vector for each state, shape should be (n_states, n_channels).
    covariances : np.ndarray
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels).
    zero_means : bool
        If True, means will be set to zero.
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
        trans_prob: np.ndarray,
        n_channels: int = None,
        means: np.ndarray = None,
        covariances: np.ndarray = None,
        zero_means: bool = None,
        random_covariances: bool = None,
        observation_error: float = 0.0,
        simulate: bool = True,
        random_seed: int = None,
    ):
        self.trans_prob = trans_prob

        if means is not None:
            if trans_prob.shape[0] != means.shape[0]:
                raise ValueError(
                    "Mismatch in the number of states in trans_prob and means."
                )

        if covariances is not None:
            if trans_prob.shape[0] != covariances.shape[0]:
                raise ValueError(
                    "Mismatch in the number of states in trans_prob and covariances."
                )

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=trans_prob.shape[0],
            means=means,
            covariances=covariances,
            zero_means=zero_means,
            random_covariances=random_covariances,
            observation_error=observation_error,
            simulate=simulate,
            random_seed=random_seed,
        )

    def generate_states(self) -> np.ndarray:
        rands = [
            iter(
                self._rng.choice(
                    self.n_states, size=self.n_samples, p=self.trans_prob[:, i]
                )
            )
            for i in range(self.n_states)
        ]
        states = np.zeros(self.n_samples, int)
        for sample in range(1, self.n_samples):
            states[sample] = next(rands[states[sample - 1]])
        return get_one_hot(states, n_states=self.n_states)


class SequenceHMMSimulation(HMMSimulation):
    """Class providing methods to simulate a sequential HMM.

    In a sequential HMM, each transition can only move to the next state numerically.
    So, 1 -> 2, 2 -> 3, 3 -> 4, etc.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    stay_prob : float
        Probability of staying in the current state (diagonals of transition matrix).
    n_channels : int
        Number of channels in the observation model. Inferred from covariances if None.
    means : np.ndarray
        Mean vector for each state, shape should be (n_states, n_channels).
    covariances : np.ndarray
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels).
    zero_means : bool
        If True, means will be set to zero.
    random_covariances : bool
        Should we simulate random covariances? False gives structured covariances.
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    simulate : bool
        Should data be simulated? Can be called using .simulate later.
    random_seed : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int,
        stay_prob: float,
        n_states: int,
        n_channels: int = None,
        means: np.ndarray = None,
        covariances: np.ndarray = None,
        zero_means: bool = None,
        random_covariances: bool = None,
        observation_error: float = 0.0,
        simulate: bool = True,
        random_seed: int = None,
    ):
        self.stay_prob = stay_prob
        trans_prob = self.construct_trans_prob(n_states, stay_prob)

        super().__init__(
            n_samples=n_samples,
            trans_prob=trans_prob,
            n_channels=n_channels,
            means=means,
            covariances=covariances,
            zero_means=zero_means,
            random_covariances=random_covariances,
            observation_error=observation_error,
            simulate=simulate,
            random_seed=random_seed,
        )

    @staticmethod
    def construct_trans_prob(n_states: int, stay_prob: float) -> np.ndarray:
        trans_prob = np.zeros([n_states, n_states])
        np.fill_diagonal(trans_prob, stay_prob)
        np.fill_diagonal(trans_prob[:, 1:], 1 - stay_prob)
        trans_prob[-1, 0] = 1 - stay_prob
        return trans_prob


class UniformHMMSimulation(HMMSimulation):
    """Class providing methods to simulate a uniform HMM.

    In a uniform HMM, the chance of moving to any state other than the current one is
    equal. So there is a diagonal term and an off-diagonal term for the transition
    probability matrix.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    stay_prob : float
        Probability of staying in the current state (diagonals of transition matrix).
    n_channels : int
        Number of channels in the observation model. Inferred from covariances if None.
    means : np.ndarray
        Mean vector for each state, shape should be (n_states, n_channels).
    covariances : np.ndarray
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels).
    zero_means : bool
        If True, means will be set to zero.
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
        stay_prob: float,
        n_states: int,
        n_channels: int = None,
        means: np.ndarray = None,
        covariances: np.ndarray = None,
        zero_means: bool = None,
        random_covariances: bool = None,
        observation_error: float = 0.0,
        simulate: bool = True,
        random_seed: int = None,
    ):
        self.stay_prob = stay_prob
        self.trans_prob = self.construct_trans_prob(n_states, stay_prob)

        super().__init__(
            n_samples=n_samples,
            trans_prob=trans_prob,
            n_channels=n_channels,
            means=means,
            covariances=covariances,
            zero_means=zero_means,
            random_covariances=random_covariances,
            observation_error=observation_error,
            simulate=simulate,
            random_seed=random_seed,
        )

    @staticmethod
    def construct_trans_prob(n_states: int, stay_prob: float):
        single_trans_prob = (1 - stay_prob) / (n_states - 1)
        trans_prob = np.ones((n_states, n_states)) * single_trans_prob
        trans_prob[np.diag_indices(n_states)] = stay_prob
        return trans_prob
