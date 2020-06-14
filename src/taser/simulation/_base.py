from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt


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
