"""Multivariate normal observation model.

"""

from typing import Union

import numpy as np
from statsmodels.stats.moment_helpers import cov2corr


class MVN:
    """Class that generates data from a multivariate normal distribution.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    n_modes : int
        Number of modes.
    means : np.ndarray or str
        Mean vector for each mode, shape should be (n_modes, n_channels).
        Either a numpy array or 'zero' or 'random'.
    covariances : np.ndarray or str
        Covariance matrix for each mode, shape should be (n_modes,
        n_channels, n_channels). Either a numpy array or 'random'.
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for the random number generator.
    """

    def __init__(
        self,
        means: Union[np.ndarray, str],
        covariances: Union[np.ndarray, str],
        n_modes: int = None,
        n_channels: int = None,
        observation_error: float = 0.0,
        random_seed: int = None,
    ):
        self._rng = np.random.default_rng(random_seed)
        self.observation_error = observation_error

        # Both the means and covariances were passed as numpy arrays
        if isinstance(means, np.ndarray) and isinstance(covariances, np.ndarray):
            if means.shape[0] != covariances.shape[0]:
                raise ValueError(
                    "means and covariances have a different number of modes."
                )
            if means.shape[1] != covariances.shape[1]:
                raise ValueError(
                    "means and covariances have a different number of channels."
                )
            self.n_modes = means.shape[0]
            self.n_channels = means.shape[1]
            self.means = means
            self.covariances = covariances

        # Only the means were passed as a numpy array
        elif isinstance(means, np.ndarray) and not isinstance(covariances, np.ndarray):
            self.n_modes = means.shape[0]
            self.n_channels = means.shape[1]
            self.means = means
            self.covariances = self.create_covariances(covariances)

        # Only the covariances were passed a numpy array
        elif not isinstance(means, np.ndarray) and isinstance(covariances, np.ndarray):
            self.n_modes = covariances.shape[0]
            self.n_channels = covariances.shape[1]
            self.means = self.create_means(means)
            self.covariances = covariances

        # Neither means or covariances were passed as numpy arrays
        elif not isinstance(means, np.ndarray) and not isinstance(
            covariances, np.ndarray
        ):
            if n_modes is None or n_channels is None:
                raise ValueError(
                    "If we are generating and means and covariances, "
                    + "n_modes and n_channels must be passed."
                )
            self.n_modes = n_modes
            self.n_channels = n_channels
            self.means = self.create_means(means)
            self.covariances = self.create_covariances(covariances)

        else:
            raise ValueError("means and covariance arugments not passed correctly.")

        # Get the var and fc from self.covariance
        self.variances = np.zeros([self.n_modes, self.n_channels])
        for i in range(self.n_modes):
            self.variances[i] = np.sqrt(np.diag(self.covariances[i]))
        # enforce positive variance
        self.variances = np.maximum(self.variances, 1e-6)

        self.fcs = np.zeros([self.n_modes, self.n_channels, self.n_channels])
        for i in range(self.n_modes):
            self.fcs[i] = cov2corr(self.covariances[i])

    def create_means(self, option):
        if option == "zero":
            means = np.zeros([self.n_modes, self.n_channels])
        elif option == "random":
            means = self._rng.normal(size=[self.n_modes, self.n_channels])
        else:
            raise ValueError("means must be a np.array or 'zero' or 'random'.")
        return means

    def create_covariances(self, option, eps=1e-6):

        if option == "random":
            # Randomly sample the elements of W from a normal distribution
            W = self._rng.normal(
                0, 0.1, size=[self.n_modes, self.n_channels, self.n_channels]
            )

            # A small value to add to the diagonal to ensure the covariances are
            # invertible
            eps = np.tile(np.eye(self.n_channels), [self.n_modes, 1, 1]) * eps

            # Calculate the covariances
            covariances = W @ W.transpose([0, 2, 1]) + eps

            # Add a large activation to a small number of the channels at random
            n_active_channels = max(1, self.n_channels // self.n_modes)
            for i in range(self.n_modes):
                active_channels = np.unique(
                    self._rng.integers(0, self.n_channels, size=n_active_channels)
                )
                covariances[i, active_channels, active_channels] += 0.25

        else:
            raise ValueError("covariances must be a np.ndarray or 'random'.")

        return covariances

    def simulate_data(self, mode_time_course):
        n_samples = mode_time_course.shape[0]

        # Initialise array to hold data
        data = np.zeros((n_samples, self.n_channels))

        # Loop through all unique combinations of modes
        for alpha in np.unique(mode_time_course, axis=0):

            # Mean and covariance for this combination of modes
            mu = np.sum(self.means * alpha[:, np.newaxis], axis=0)
            sigma = np.sum(self.covariances * alpha[:, np.newaxis, np.newaxis], axis=0)

            # Generate data for the time points that this combination of modes is
            # active
            data[
                np.all(mode_time_course == alpha, axis=1)
            ] = self._rng.multivariate_normal(
                mu,
                sigma,
                size=np.count_nonzero(np.all(mode_time_course == alpha, axis=1)),
            )

        # Add an error to the data at all time points
        data += self._rng.normal(scale=self.observation_error, size=data.shape)

        return data.astype(np.float32)

    def simulate_data_multiple_scales(self, state_time_courses):

        # Here state_time_courses.shape = (n_samples, n_modes, 3)
        # It contains 3 different time courses for mean, variance, and fc

        n_samples = state_time_courses.shape[0]

        # Initialise array to hold data
        data = np.zeros([n_samples, self.n_channels])

        # Loop through all unique combinations of states
        for time_courses in np.unique(state_time_courses, axis=0):
            # Extract the 3 different time courses
            alpha = time_courses[:, 0]
            beta = time_courses[:, 1]
            gamma = time_courses[:, 2]

            # Mean, variance, fc for this combination of 3 time courses
            mu = np.sum(self.means * alpha[:, np.newaxis], axis=0)
            G = np.diag(np.sum(self.variances * beta[:, np.newaxis], axis=0))
            F = np.sum(self.fcs * gamma[:, np.newaxis, np.newaxis], axis=0)

            sigma = np.matmul(G, np.matmul(F, G))

            # Generate data for the time points that this combination of states is active
            data[
                np.all(state_time_courses == time_courses, axis = (1,2))
            ] = self._rng.multivariate_normal(
                mu, sigma, size = np.count_nonzero(np.all(state_time_courses == time_courses, axis = (1,2)))
            )

        # Add an error to the data at all time points
        data += self._rng.normal(scale=self.observation_error, size = data.shape)

        return data.astype(np.float32)
