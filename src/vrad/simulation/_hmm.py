import numpy as np
from vrad.array_ops import get_one_hot
from vrad.simulation import Simulation
from vrad.utils.decorators import auto_repr, auto_yaml


class HMMSimulation(Simulation):
    @auto_yaml
    @auto_repr
    def __init__(
        self,
        trans_prob: np.ndarray,
        n_samples: int = 20000,
        n_channels: int = 7,
        n_states: int = 4,
        sim_varying_means: bool = False,
        random_covariance_weights: bool = False,
        observation_error: float = 0.2,
        markov_lag: int = 1,
        covariances: np.ndarray = None,
        random_seed: int = None,
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
        )

    def generate_states(self) -> np.ndarray:

        self.cumsum_trans_prob = np.cumsum(self.trans_prob, axis=1)
        alpha_sim = np.zeros((self.n_samples, self.n_states))
        z = np.zeros([self.n_samples + 1], dtype=int)
        rands = self._rng.random(self.n_samples)

        for tt in range(self.markov_lag - 1, self.n_samples):
            tmp = rands[tt]
            for kk in range(self.cumsum_trans_prob.shape[1]):
                if tmp < self.cumsum_trans_prob[z[tt - self.markov_lag], kk]:
                    z[tt] = kk
                    break
            alpha_sim[tt, z[tt]] = 1
        return alpha_sim


class SequenceHMMSimulation(HMMSimulation):
    def __init__(
        self,
        n_samples: int = 20000,
        n_channels: int = 7,
        n_states: int = 4,
        sim_varying_means: bool = False,
        markov_lag: int = 1,
        stay_prob: float = 0.95,
        random_covariance_weights: bool = False,
        observation_error: float = 0.2,
        random_seed: int = None,
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
        n_samples: int = 20000,
        n_channels: int = 7,
        n_states: int = 4,
        sim_varying_means: bool = False,
        markov_lag: int = 1,
        stay_prob: float = 0.95,
        random_covariance_weights: bool = False,
        observation_error: float = 0.2,
        random_seed: int = None,
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
        n_samples: int = 20000,
        n_channels: int = 7,
        n_states: int = 4,
        sim_varying_means: bool = False,
        markov_lag: int = 1,
        stay_prob: float = 0.95,
        random_covariance_weights: bool = False,
        observation_error: float = 0.2,
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
        n_samples: int = 20000,
        n_channels: int = 7,
        n_states: int = 4,
        sim_varying_means: bool = False,
        random_covariance_weights: bool = False,
        observation_error: float = 0.2,
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
