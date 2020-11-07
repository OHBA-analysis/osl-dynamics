"""Classes for simulation Hidden Markov Models (HMMs).

"""

import numpy as np
from vrad.array_ops import get_one_hot
from vrad.simulation import Simulation
from vrad.utils.decorators import auto_repr, auto_yaml


class HMMSimulation(Simulation):
    """Class providing methods to simulate a hidden Markov model.

    Parameters
    ----------
    trans_prob : numpy.ndarray
        Transition probability matrix.
    n_samples : int
        Number of samples to draw from the model.
    n_states : int
        Number of states in the Markov chain.
    zero_means : bool
        Should means vary over channels and states?
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    covariances : numpy.ndarray
        The covariances matrices for each state in the observation
    n_channels : int
        Number of channels in the observation model. Inferred from covariances if None.
    random_covariance_weights : bool
        Randomly sample covariances.
    random_seed : int
        Seed for reproducibility.
    simulate : bool
        Should data be simulated? Can be called using .simulate later.
    """

    @auto_yaml
    @auto_repr
    def __init__(
        self,
        trans_prob: np.ndarray,
        n_samples: int,
        n_states: int,
        zero_means: bool,
        observation_error: float,
        covariances: np.ndarray,
        n_channels: int = None,
        random_covariance_weights: bool = False,
        random_seed: int = None,
        simulate: bool = True,
    ):
        if covariances is not None:
            n_channels = covariances.shape[1]

        if n_states != trans_prob.shape[0]:
            raise ValueError(
                f"Number of states ({n_states}) and shape of transition "
                + f"probability matrix {trans_prob.shape} incomptible."
            )

        self.trans_prob = trans_prob
        self.cumsum_trans_prob = None

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            zero_means=zero_means,
            random_covariance_weights=random_covariance_weights,
            observation_error=observation_error,
            covariances=covariances,
            random_seed=random_seed,
            simulate=simulate,
        )

        if not simulate:
            self.state_time_course = self.generate_states()

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
    """Class providing methods to simulate a sequential hidden Markov model.

    In a sequential HMM, each transition can only move to the next state numerically.
    So, 1 -> 2, 2 -> 3, 3 -> 4, etc.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    n_channels : int
        Number of channels in the observation model. Inferred from covariances if None.
    n_states : int
        Number of states in the Markov chain.
    zero_means : bool
        Should means vary over channels and states?
    stay_prob : float
        Probability of staying in the current state (diagonals of transition matrix).
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    random_covariance_weights : bool
        Randomly sample covariances.
    random_seed : int
        Seed for reproducibility.
    simulate : bool
        Should data be simulated? Can be called using .simulate later.
    """

    def __init__(
        self,
        n_samples: int,
        n_channels: int,
        n_states: int,
        zero_means: bool,
        stay_prob: float,
        observation_error: float,
        random_covariance_weights: bool = False,
        random_seed: int = None,
        simulate: bool = True,
    ):

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            zero_means=zero_means,
            random_covariance_weights=random_covariance_weights,
            observation_error=observation_error,
            trans_prob=self.construct_trans_prob_matrix(n_states, stay_prob),
            random_seed=random_seed,
            simulate=simulate,
        )

    @staticmethod
    def construct_trans_prob_matrix(n_states: int, stay_prob: float) -> np.ndarray:

        trans_prob = np.zeros([n_states, n_states])
        np.fill_diagonal(trans_prob, stay_prob)
        np.fill_diagonal(trans_prob[:, 1:], 1 - stay_prob)
        trans_prob[-1, 0] = 1 - stay_prob
        return trans_prob


class BasicHMMSimulation(HMMSimulation):
    """Class providing methods to simulate a basic hidden Markov model.

    In a basic HMM, the chance of moving to any state other than the current one is
    equal. So there is a diagonal term and an off-diagonal term for the transition
    probability matrix.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    n_channels : int
        Number of channels in the observation model. Inferred from covariances if None.
    n_states : int
        Number of states in the Markov chain.
    zero_means : bool
        Should means vary over channels and states?
    stay_prob : float
        Probability of staying in the current state (diagonals of transition matrix).
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    random_covariance_weights : bool
        Randomly sample covariances.
    random_seed : int
        Seed for reproducibility.
    simulate : bool
        Should data be simulated? Can be called using .simulate later.
    """

    @auto_yaml
    @auto_repr
    def __init__(
        self,
        n_samples: int,
        n_channels: int,
        n_states: int,
        zero_means: bool,
        stay_prob: float,
        observation_error: float,
        random_covariance_weights: bool = False,
        random_seed: int = None,
        simulate: bool = True,
    ):

        self.stay_prob = stay_prob

        self.trans_prob = None

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            zero_means=zero_means,
            random_covariance_weights=random_covariance_weights,
            observation_error=observation_error,
            trans_prob=self.construct_trans_prob_matrix(n_states, stay_prob),
            random_seed=random_seed,
            simulate=simulate,
        )

    @staticmethod
    def construct_trans_prob_matrix(n_states: int, stay_prob: float):
        single_trans_prob = (1 - stay_prob) / (n_states - 1)
        trans_prob = np.ones((n_states, n_states)) * single_trans_prob
        trans_prob[np.diag_indices(n_states)] = stay_prob
        return trans_prob


class UniHMMSimulation(HMMSimulation):
    def __init__(
        self,
        n_samples: int,
        n_channels: int,
        n_states: int,
        zero_means: bool,
        stay_prob: float,
        observation_error: float,
        random_covariance_weights: bool = False,
        random_seed: int = None,
    ):

        self.stay_prob = stay_prob

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            zero_means=zero_means,
            random_covariance_weights=random_covariance_weights,
            observation_error=observation_error,
            trans_prob=self.construct_trans_prob_matrix(stay_prob),
            random_seed=random_seed,
        )

    @staticmethod
    def construct_trans_prob_matrix(stay_prob: float) -> np.ndarray:
        """An HMM with equal transfer probabilities for all non-active states.

        Returns
        -------
        alpha_sim : np.array
            State time course
        """
        trans_prob = np.ones([2, 2]) * (1 - stay_prob)
        trans_prob.trans_prob[np.diag_indices(2)] = stay_prob
        return trans_prob


class RandomHMMSimulation(HMMSimulation):
    """Class providing methods to simulate a random hidden Markov model.

    State choice is completely random with all elements of the transition probability
    matrix set to the same value.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    n_channels : int
        Number of channels in the observation model. Inferred from covariances if None.
    n_states : int
        Number of states in the Markov chain.
    zero_means : bool
        Should means vary over channels and states?
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    random_covariance_weights : bool
        Randomly sample covariances.
    random_seed : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int,
        n_channels: int,
        n_states: int,
        zero_means: bool,
        observation_error: float,
        random_covariance_weights: bool = False,
        random_seed: int = None,
    ):

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            zero_means=zero_means,
            random_covariance_weights=random_covariance_weights,
            observation_error=observation_error,
            trans_prob=self.construct_trans_prob_matrix(n_states),
            random_seed=random_seed,
        )

    @staticmethod
    def construct_trans_prob_matrix(n_states: int):
        return np.ones((n_states, n_states)) * 1 / n_states

    def generate_states(self) -> np.ndarray:
        # TODO: Ask Mark what FOS is short for (frequency of state?)
        z = np.random.choice(self.n_states, size=self.n_samples)
        alpha_sim = get_one_hot(z, self.n_states)
        return alpha_sim
