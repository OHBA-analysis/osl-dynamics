from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from vrad.utils import plotting


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
    observation_error : float
        The standard deviation of noise added to the signal from a normal distribution.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        n_channels: int = 7,
        n_states: int = 4,
        sim_varying_means: bool = False,
        random_covariance_weights: bool = False,
        observation_error: float = 0.2,
        covariances: np.ndarray = None,
        simulate: bool = True,
    ):

        self.n_samples = n_samples
        self.n_channels = n_channels
        self.n_states = n_states
        self.sim_varying_means = sim_varying_means
        self.random_covariance_weights = random_covariance_weights
        self.observation_error = observation_error

        self.state_time_course = None
        self.covariances = (
            self.create_covariances() if covariances is None else covariances
        )
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

    def plot_alphas(self, n_points: int = 1000, filename: str = None):
        """Method for plotting the state time course of a simulation.

        Parameters
        ----------
        n_points : int
            Number of time points to plot.

        Returns
        -------

        """
        plotting.plot_state_time_courses(
            self.state_time_course, n_samples=n_points, filename=filename
        )

    def create_covariances(self, identity_factor: float = 0.0001) -> np.ndarray:
        """Create the covariance matrices for the simulation

        Parameters
        ----------
        identity_factor : float
            Factor by which to scale the identity matrix which is added to the
            covariance.

        Returns
        -------
        covariances_sim : np.array
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
        covariances_sim = (
            tilde_cov_weights @ tilde_cov_weights.transpose([0, 2, 1]) + scaled_identity
        )

        normalisation = np.trace(covariances_sim, axis1=1, axis2=2).reshape((-1, 1, 1))
        covariances_sim /= normalisation
        return covariances_sim

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
                self.covariances[i],
                size=np.count_nonzero(self.state_time_course.argmax(axis=1) == i),
            )

        data_sim += np.random.default_rng().normal(
            scale=self.observation_error, size=data_sim.shape
        )

        return data_sim.astype(np.float32)

    def plot_data(self, n_points: int = 1000, filename: str = None):
        """Method for plotting simulated data.

        Parameters
        ----------
        n_points : int
            Number of time points to plot.
        """
        n_points = min(n_points, self.n_samples)
        plotting.plot_time_series(
            self.time_series, n_samples=n_points, filename=filename
        )
