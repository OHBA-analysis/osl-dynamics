"""Simulations for MEG data

This module allows the user to conveniently simulate MEG data. Instantiating the
`Simulation` class automatically takes the user's input parameters and produces data
which can be analysed.

"""
import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from taser.array_ops import get_one_hot
from taser.decorators import auto_repr, auto_yaml


class Simulation(ABC):
    """Class for making the simulation of MEG data easy!

    Parameters
    ----------
    n_samples : int
        Number of time points to generate
    n_channels : int
        Number of channels to create
    n_states : int
        Number of states to simulate
    sim_varying_means : bool
        If False, means will be set to zero.
    random_covariance_weights : bool
        Should the simulation use random covariance weights? False gives structured
        covariances.
    e_std : float
        The standard deviation of noise added to the signal from a normal distribution.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        n_channels: int = 7,
        n_states: int = 4,
        sim_varying_means: bool = False,
        random_covariance_weights: bool = False,
        e_std: float = 0.2,
        djs: np.ndarray = None,
        simulate: bool = True,
    ):

        self.n_samples = n_samples
        self.n_channels = n_channels
        self.n_states = n_states
        self.sim_varying_means = sim_varying_means
        self.random_covariance_weights = random_covariance_weights
        self.e_std = e_std

        self.state_time_course = None
        self.djs = self.create_djs() if djs is None else djs
        self.time_series = None

        if simulate:
            self.simulate()

    def simulate(self):
        self.state_time_course = self.generate_states()
        self.time_series = self.simulate_data()

    def __array__(self):
        return self.time_series

    def __getattr__(self, attr):
        if attr == "time_series":
            raise NameError("time_series has not yet been created.")
        if attr[:2] == "__":
            raise AttributeError(f"No attribute called {attr}.")
        return getattr(self.time_series, attr)

    @abstractmethod
    def generate_states(self) -> np.ndarray:
        """State generation must be implemented by subclasses.

        """
        pass

    def plot_alphas(self, n_points: int = 1000):
        """Method for plotting the state time course of a simulation.

        Parameters
        ----------
        n_points : int
            Number of time points to plot.

        Returns
        -------

        """
        plt.figure()
        plt.plot(self.state_time_course[0:n_points])
        plt.show()

    def create_djs(self, identity_factor: float = 0.0001) -> np.ndarray:
        """Create the covariance matrices for the simulation

        Parameters
        ----------
        identity_factor : float
            Factor by which to scale the identity matrix which is added to the
            covariance.

        Returns
        -------
        djs_sim : np.array
            The covariance matrices of the simulation

        """
        if self.random_covariance_weights:
            tilde_cov_weights = np.random.normal(
                size=(self.n_states, self.n_channels, self.n_channels)
            )
        else:
            tilde_cov_weights = np.zeros(
                (self.n_states, self.n_channels, self.n_channels)
            )
            np.fill_diagonal(
                tilde_cov_weights[: self.n_states, : self.n_states, : self.n_states],
                val=1,
            )

        scaled_identity = (
            np.tile(np.eye(self.n_channels), [self.n_states, 1, 1]) * identity_factor
        )
        djs_sim = (
            tilde_cov_weights @ tilde_cov_weights.transpose([0, 2, 1]) + scaled_identity
        )

        normalisation = np.trace(djs_sim, axis1=1, axis2=2).reshape((-1, 1, 1))
        djs_sim /= normalisation
        return djs_sim

    def simulate_data(self) -> np.ndarray:
        """Simulate a time course of MEG data.

        Returns
        -------
        data_sim : np.array
            A float32 array containing a simulated time course of simulated data.

        """
        if self.sim_varying_means:
            mus_sim = np.random.normal((self.n_states, self.n_channels))
        else:
            mus_sim = np.zeros((self.n_states, self.n_channels))

        data_sim = np.zeros((self.n_samples, self.n_channels))
        for i in range(self.n_states):
            data_sim[
                self.state_time_course.argmax(axis=1) == i
            ] = np.random.default_rng().multivariate_normal(
                mus_sim[i],
                self.djs[i],
                size=np.count_nonzero(self.state_time_course.argmax(axis=1) == i),
            )

        data_sim += np.random.default_rng().normal(
            scale=self.e_std, size=data_sim.shape
        )

        return data_sim.astype(np.float32)

    def plot_data(self, n_points: int = 1000):
        """Method for plotting simulated data.

        Parameters
        ----------
        n_points : int
            Number of time points to plot.
        """
        n_points = min(n_points, self.n_samples)
        fig, y_axes = plt.subplots(
            1, min(self.n_channels, 10), figsize=(20, 3), sharey="row"
        )
        for y_axis, y_channel in zip(y_axes, self.time_series):
            y_axis.plot(np.arange(n_points), y_channel[:n_points])
        fig, alpha_axes = plt.subplots(1, min(self.n_states, 10), figsize=(15, 3))
        for alpha_axis, alpha_channel in zip(alpha_axes, self.state_time_course.T):
            alpha_axis.plot(np.arange(n_points), alpha_channel[:n_points])
        plt.tight_layout()
        plt.show()


class HMMSimulation(Simulation):
    @auto_yaml
    @auto_repr
    def __init__(
        self,
        trans_prob: np.ndarray,
        n_samples: int = 20000,
        n_channels: int = 7,
        sim_varying_means: bool = False,
        random_covariance_weights: bool = False,
        e_std: float = 0.2,
        markov_lag: int = 1,
        djs: np.ndarray = None,
    ):
        if djs is not None:
            n_channels = djs.shape[1]

        self.markov_lag = markov_lag
        self.trans_prob = trans_prob
        self.cumsum_trans_prob = None

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=trans_prob.shape[0],
            sim_varying_means=sim_varying_means,
            random_covariance_weights=random_covariance_weights,
            e_std=e_std,
            djs=djs,
        )

    def generate_states(self) -> np.ndarray:

        self.cumsum_trans_prob = np.cumsum(self.trans_prob, axis=1)
        alpha_sim = np.zeros((self.n_samples, self.n_states))
        z = np.zeros([self.n_samples + 1], dtype=int)
        rands = np.random.rand(self.n_samples)

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
        e_std: float = 0.2,
    ):

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            sim_varying_means=sim_varying_means,
            random_covariance_weights=random_covariance_weights,
            e_std=e_std,
            markov_lag=markov_lag,
            trans_prob=self.construct_trans_prob_matrix(n_states, stay_prob),
        )

    @staticmethod
    def construct_trans_prob_matrix(n_states: int, stay_prob: float) -> np.ndarray:

        trans_prob = np.zeros([n_states, n_states])
        np.fill_diagonal(trans_prob, 0.95)
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
        e_std: float = 0.2,
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
            e_std=e_std,
            trans_prob=self.construct_trans_prob_matrix(n_states, stay_prob),
        )

    @staticmethod
    def construct_trans_prob_matrix(n_states: int, stay_prob: float):
        """Standard sequential HMM

        Returns
        -------
        alpha_sim : np.array
            State time course

        """
        single_trans_prob = (1 - stay_prob) / n_states
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
        e_std: float = 0.2,
    ):

        self.markov_lag = markov_lag
        self.stay_prob = stay_prob

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            sim_varying_means=sim_varying_means,
            random_covariance_weights=random_covariance_weights,
            e_std=e_std,
            trans_prob=self.construct_trans_prob_matrix(stay_prob),
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
        e_std: float = 0.2,
    ):

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            sim_varying_means=sim_varying_means,
            random_covariance_weights=random_covariance_weights,
            e_std=e_std,
            trans_prob=self.construct_trans_prob_matrix(n_states),
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


class HiddenSemiMarkovSimulation(Simulation):
    @auto_yaml
    @auto_repr
    def __init__(
        self,
        n_samples: int = 20000,
        n_channels: int = 7,
        n_states: int = 4,
        sim_varying_means: bool = False,
        random_covariance_weights: bool = False,
        e_std: float = 0.2,
        off_diagonal_trans_prob: np.ndarray = None,
        full_trans_prob: np.ndarray = None,
        stay_prob: float = 0.95,
        gamma_shape: float = 5,
        gamma_scale: float = 10,
    ):
        self.off_diagonal_trans_prob = off_diagonal_trans_prob
        self.full_trans_prob = full_trans_prob
        self.cumsum_off_diagonal_trans_prob = None
        self.stay_prob = stay_prob

        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            sim_varying_means=sim_varying_means,
            random_covariance_weights=random_covariance_weights,
            e_std=e_std,
        )

    def construct_off_diagonal_trans_prob(self):
        if self.off_diagonal_trans_prob is not None and (
            self.full_trans_prob is not None
        ):
            raise ValueError(
                "Exactly one of off_diagonal_trans_prob and full_trans_prob must be "
                "specified. "
            )

        if (self.off_diagonal_trans_prob is None) and (self.full_trans_prob is None):
            self.off_diagonal_trans_prob = np.ones([self.n_states, self.n_states])
            np.fill_diagonal(self.off_diagonal_trans_prob, 0)

            self.off_diagonal_trans_prob = (
                self.off_diagonal_trans_prob
                / self.off_diagonal_trans_prob.sum(axis=1)[:, None]
            )

        if self.off_diagonal_trans_prob is not None:
            self.off_diagonal_trans_prob = self.off_diagonal_trans_prob

        if self.full_trans_prob is not None:
            self.off_diagonal_trans_prob = (
                self.full_trans_prob / self.full_trans_prob.sum(axis=1)[:, None]
            )

        with np.printoptions(linewidth=np.nan):
            logging.info(
                f"off_diagonal_trans_prob is:\n{str(self.off_diagonal_trans_prob)}"
            )

    def generate_states(self):
        self.construct_off_diagonal_trans_prob()
        self.cumsum_off_diagonal_trans_prob = np.cumsum(
            self.off_diagonal_trans_prob, axis=1
        )
        alpha_sim = np.zeros(self.n_samples, dtype=np.int)

        gamma_sample = np.random.default_rng().gamma
        random_sample = np.random.default_rng().uniform
        current_state = np.random.default_rng().integers(0, self.n_states)
        current_position = 0

        while current_position < len(alpha_sim):
            state_lifetime = np.round(
                gamma_sample(shape=self.gamma_shape, scale=self.gamma_scale)
            ).astype(np.int)
            alpha_sim[
                current_position : current_position + state_lifetime
            ] = current_state

            tmp = random_sample()
            for kk in range(self.cumsum_off_diagonal_trans_prob.shape[1]):
                if tmp < self.cumsum_off_diagonal_trans_prob[current_state, kk]:
                    break

            current_position += state_lifetime
            current_state = kk

        logging.debug(f"n_states present in alpha sim = {len(np.unique(alpha_sim))}")

        one_hot_alpha_sim = get_one_hot(alpha_sim, n_states=self.n_states)

        logging.debug(f"one_hot_alpha_sim.shape = {one_hot_alpha_sim.shape}")

        return one_hot_alpha_sim
