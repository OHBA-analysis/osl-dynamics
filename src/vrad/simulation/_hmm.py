"""Classes for simulation Hidden Markov Models (HMMs).

"""

import numpy as np
from vrad.array_ops import get_one_hot
from vrad.simulation import Simulation
from vrad.utils.decorators import auto_repr, auto_yaml, timing


class HMMSimulation(Simulation):
    @auto_yaml
    @auto_repr
    def __init__(
        self,
        trans_prob: np.ndarray,
        n_samples: int,
        n_states: int,
        sim_varying_means: bool,
        observation_error: float,
        covariances: np.ndarray,
        n_channels: int = None,
        random_covariance_weights: bool = False,
        markov_lag: int = 1,
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

        self.markov_lag = markov_lag
        self.trans_prob = trans_prob
        self.cumsum_trans_prob = None

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            sim_varying_means=sim_varying_means,
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
    def __init__(
        self,
        n_samples: int,
        n_channels: int,
        n_states: int,
        sim_varying_means: bool,
        stay_prob: float,
        observation_error: float,
        random_covariance_weights: bool = False,
        markov_lag: int = 1,
        random_seed: int = None,
        simulate: bool = True,
    ):

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            sim_varying_means=sim_varying_means,
            random_covariance_weights=random_covariance_weights,
            observation_error=observation_error,
            markov_lag=markov_lag,
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
    @auto_yaml
    @auto_repr
    def __init__(
        self,
        n_samples: int,
        n_channels: int,
        n_states: int,
        sim_varying_means: bool,
        stay_prob: float,
        observation_error: float,
        random_covariance_weights: bool = False,
        markov_lag: int = 1,
        random_seed: int = None,
        simulate: bool = True,
    ):

        self.markov_lag = markov_lag
        self.stay_prob = stay_prob

        self.trans_prob = None

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            sim_varying_means=sim_varying_means,
            random_covariance_weights=random_covariance_weights,
            observation_error=observation_error,
            trans_prob=self.construct_trans_prob_matrix(n_states, stay_prob),
            random_seed=random_seed,
            simulate=simulate,
        )

    @staticmethod
    def construct_trans_prob_matrix(n_states: int, stay_prob: float):
        """Standard sequential HMM

        Returns
        -------
        alpha_sim : np.array
            State time course

        """
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
        sim_varying_means: bool,
        stay_prob: float,
        observation_error: float,
        random_covariance_weights: bool = False,
        markov_lag: int = 1,
        random_seed: int = None,
    ):

        self.markov_lag = markov_lag
        self.stay_prob = stay_prob

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            sim_varying_means=sim_varying_means,
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
    def __init__(
        self,
        n_samples: int,
        n_channels: int,
        n_states: int,
        sim_varying_means: bool,
        observation_error: float,
        random_covariance_weights: bool = False,
        random_seed: int = None,
    ):

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            sim_varying_means=sim_varying_means,
            random_covariance_weights=random_covariance_weights,
            observation_error=observation_error,
            trans_prob=self.construct_trans_prob_matrix(n_states),
            random_seed=random_seed,
        )

    @staticmethod
    def construct_trans_prob_matrix(n_states: int):
        return np.ones((n_states, n_states)) * 1 / n_states

    def generate_states(self) -> np.ndarray:
        """Totally random state selection HMM

        Returns
        -------
        alpha_sim : np.array
            State time course
        """
        # TODO: Ask Mark what FOS is short for (frequency of state?)
        z = np.random.choice(self.n_states, size=self.n_samples)
        alpha_sim = get_one_hot(z, self.n_states)
        return alpha_sim
