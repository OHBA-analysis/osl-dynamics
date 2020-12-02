"""Base class for simulations.

"""

from abc import ABC, abstractmethod

import numpy as np
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
    means : np.ndarary
        Mean vector for each state, shape should be (n_states, n_channels).
    covariances : np.ndarray
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels).
    zero_means : bool
        If True, means will be set to zero, otherwise they are sampled from a
        normal distribution.
    random_covariances : bool
        Should we simulate random covariances? False gives structured covariances.
    observation_error : float
        The standard deviation of noise added to the signal from a normal distribution.
    simulate : bool
        Should we simulate the time series?
    random_seed : int
        Seed for the random number generator
    """

    def __init__(
        self,
        n_samples: int,
        n_channels: int,
        n_states: int,
        means: np.ndarray,
        covariances: np.ndarray,
        zero_means: bool,
        random_covariances: bool,
        observation_error: float,
        simulate: bool,
        random_seed: int,
    ):
        n_channels = validate_means_covariances(
            n_channels, means, covariances, zero_means, random_covariances
        )

        self._rng = np.random.default_rng(random_seed)

        self.n_samples = n_samples
        self.n_channels = n_channels
        self.n_states = n_states

        self.zero_means = zero_means
        self.random_covariances = random_covariances
        self.means = self.create_means() if means is None else means
        self.covariances = (
            self.create_covariances() if covariances is None else covariances
        )

        self.observation_error = observation_error
        self.state_time_course = None
        self.time_series = None

        if simulate:
            self.simulate()

    def simulate(self):
        self.state_time_course = self.generate_states()
        self.time_series = self.simulate_data()

    def __array__(self):
        return self.time_series

    def __iter__(self):
        return iter([self.time_series])

    def __getattr__(self, attr):
        if attr == "time_series":
            raise NameError("time_series has not yet been created.")
        if attr[:2] == "__":
            raise AttributeError(f"No attribute called {attr}.")
        return getattr(self.time_series, attr)

    def __len__(self):
        return 1

    @abstractmethod
    def generate_states(self) -> np.ndarray:
        """State generation must be implemented by subclasses."""
        pass

    def plot_alphas(self, n_points: int = 1000, filename: str = None):
        """Method for plotting the state time course of a simulation.

        Parameters
        ----------
        n_points : int
            Number of time points to plot.
        filename : str
            Filename to save plot to.

        """
        plotting.plot_state_time_courses(
            self.state_time_course, n_samples=n_points, filename=filename
        )

    def create_means(self) -> np.ndarray:
        """Create the mean vectors for the simulation.

        Returns
        -------
        means : np.ndarray
            Mean vector for each state.
        """
        if self.zero_means:
            means = np.zeros([self.n_states, self.n_channels])
        else:
            # Random normally distributed means for each channel
            means = self._rng.normal(size=[self.n_states, self.n_channels])

        return means

    def create_covariances(self, eps: float = 1e-6) -> np.ndarray:
        """Create the covariance matrices for the simulation.

        Create a covariance matrix using the formula Cov = W W^T.

        Parameters
        ----------
        eps : float
            Small value to add to the diagonal to ensure covariance matrices are
            invertible. Default is 1e-6.

        Returns
        -------
        covariances : np.array
            The covariance matrices of the simulation.
        """
        if self.random_covariances:
            # Randomly select the elements of W from a normal distribution.
            W = self._rng.normal(size=[self.n_states, self.n_channels, self.n_channels])
        else:
            # Simulate single channel activation
            W = np.zeros([self.n_states, self.n_channels, self.n_channels])
            np.fill_diagonal(
                W[: self.n_states, : self.n_states, : self.n_states], val=1
            )

        # A small value to add to the diagonal to ensure the covariances are invertible
        eps = np.tile(np.eye(self.n_channels), [self.n_states, 1, 1]) * eps

        # Calculate the covariances
        covariances = W @ W.transpose([0, 2, 1]) + eps

        # Trace normalisation
        covariances /= np.trace(covariances, axis1=1, axis2=2)[
            ..., np.newaxis, np.newaxis
        ]

        return covariances

    def simulate_data(self) -> np.ndarray:
        """Simulate a time course of MEG data.

        Returns
        -------
        data : np.array
            A float32 array containing a simulated time course of simulated data.
        """

        # State time course, shape=(n_samples, n_states)
        # This contains the mixing factors of each states at each time point
        stc = self.state_time_course

        # Array to hold the simulated data
        data = np.zeros((self.n_samples, self.n_channels))

        # Loop through all unique combinations of states
        for alpha in np.unique(stc, axis=0):

            # Mean and covariance for this combination of states
            mu = np.sum(self.means * alpha[:, np.newaxis], axis=0)
            sigma = np.sum(self.covariances * alpha[:, np.newaxis, np.newaxis], axis=0)

            # Generate data for the time points that this combination of states is
            # active
            data[np.all(stc == alpha, axis=1)] = self._rng.multivariate_normal(
                mu, sigma, size=np.count_nonzero(np.all(stc == alpha, axis=1))
            )

        # Add an error to the data at all time points
        data += self._rng.normal(scale=self.observation_error, size=data.shape)

        return data.astype(np.float32)

    def standardize(self):
        """Standardizes the data.

        The time series data is z-transformed and the covariances are converted
        to correlation matrices.
        """
        means = np.mean(self.time_series, axis=0)
        standard_deviations = np.std(self.time_series, axis=0)

        # Z-transform
        self.time_series -= means
        self.time_series /= standard_deviations

        # Convert covariance matrices to correlation matrices
        self.covariances /= np.outer(standard_deviations, standard_deviations)[
            np.newaxis, ...
        ]

    def plot_data(self, n_points: int = 1000, filename: str = None):
        """Method for plotting simulated data.

        Parameters
        ----------
        n_points : int
            Number of time points to plot.
        filename : str
            Filename to save plot to.
        """
        n_points = min(n_points, self.n_samples)
        plotting.plot_time_series(
            self.time_series, n_samples=n_points, filename=filename
        )


def validate_means_covariances(
    n_channels: int,
    means: np.ndarray,
    covariances: np.ndarray,
    zero_means: bool,
    random_covariances: bool,
):
    """Validates mean vectors and covariance matrices.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    means : np.ndarray
        Mean vectors. Shape should be (n_states, n_channels).
    covariances : np.ndarray
        Covariance matrices. Shape should be (n_states, n_channels, n_channels).
    zero_means : bool
        If True, means will be set to zero, otherwise they are sampled from a
        normal distribution.
    random_covariances : bool
        Should we simulate random covariances? False gives structured covariances.

    Returns
    -------
    n_channels : int
        Number of channels.
    """

    # If no means or covariances have been passed, check the number of channels
    # has been passed so we can generate the means and covariances
    if means is None and covariances is None and n_channels is None:
        raise ValueError("If n_channels is None, means or covariances must be passed.")

    # If no means have been passed, check we have the options for generating
    # the means
    if means is None and zero_means is None:
        raise ValueError("Either means or zero_means must be passed.")

    # If no covariances have been passed, check we have the options for generating
    # the covariances.
    if covariances is None and random_covariances is None:
        raise ValueError("Either covariances or random_covariances must be passed.")

    # Check shapes are consistent
    if means is not None and covariances is not None:
        if means.shape[0] != covariances.shape[0]:
            raise ValueError(
                "Number of mean vectors and covariance matrices do not match. "
                + f"means.shape[0]={means.shape[0]}, "
                + f"covariances.shape[0]={covariances.shape[0]}."
            )
        if means.shape[1] != covariances.shape[1]:
            raise ValueError(
                "There is a mismatch in the number of channels in the mean "
                + "vectors and covariances matrices. "
                + f"means.shape[1]={means.shape[1]}, "
                + f"covariances[1]={covariances.shape[1]}."
            )

    # Set the number of channels from the mean vectors or covariance matrices
    if means is None and covariances is not None:
        n_channels = covariances.shape[1]
    if means is not None and covariances is None:
        n_channels = means.shape[1]

    return n_channels
