"""Multivariate normal observation model.

"""

import numpy as np
from osl_dynamics import array_ops


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
        means,
        covariances,
        n_modes=None,
        n_channels=None,
        observation_error=0.0,
        random_seed=None,
    ):
        self._rng = np.random.default_rng(random_seed)
        self.observation_error = observation_error
        self.instantaneous_covs = None

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
            self.covariances, self.W = self.create_covariances(covariances)

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
            self.covariances, self.W = self.create_covariances(covariances)

        else:
            raise ValueError("means and covariance arugments not passed correctly.")

    def create_means(self, option):
        if option == "zero":
            means = np.zeros([self.n_modes, self.n_channels])
        elif option == "random":
            means = self._rng.normal(0, 0.2, size=[self.n_modes, self.n_channels])
        else:
            raise ValueError("means must be a np.array or 'zero' or 'random'.")
        return means

    def create_covariances(self, option, eps=1e-6):
        if option == "random":
            # Randomly sample the elements of W from a normal distribution
            W = self._rng.normal(
                0, 0.1, size=[self.n_modes, self.n_channels, self.n_channels]
            )

            # Add a large activation to a small number of the channels at random
            n_active_channels = max(1, 2 * self.n_channels // self.n_modes)
            for i in range(self.n_modes):
                active_channels = np.unique(
                    self._rng.integers(0, self.n_channels, size=n_active_channels)
                )
                W[i, active_channels] += 1 / self.n_channels

            # A small value to add to the diagonal to ensure the covariances are
            # invertible
            eps = np.tile(np.eye(self.n_channels), [self.n_modes, 1, 1]) * eps

            # Calculate the covariances
            covariances = W @ W.transpose([0, 2, 1]) + eps

        else:
            raise ValueError("covariances must be a np.ndarray or 'random'.")

        return covariances, W

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


class MDyn_MVN(MVN):
    """Class that generates data from a multivariate normal distribution.

    Multi-time-scale version of MVN.

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
        means,
        covariances,
        n_modes=None,
        n_channels=None,
        observation_error=0.0,
        random_seed=None,
    ):
        super().__init__(
            means=means,
            covariances=covariances,
            n_modes=n_modes,
            n_channels=n_channels,
            observation_error=observation_error,
            random_seed=random_seed,
        )

        # Get the std and FC from self.covariance
        self.stds = array_ops.cov2std(self.covariances)
        self.fcs = array_ops.cov2corr(self.covariances)

    def simulate_data(self, state_time_courses):
        """Simulates data.

        Parameters
        ----------
        state_time_courses : np.ndarray
            It contains 2 different time courses for mean+standard deviations
            and functional connectiivty. Shape is (2, n_samples, n_modes).

        Returns
        -------
        np.ndarray
            Simulated data. Shape is (n_samples, n_channels).
        """
        # Reshape state_time_courses so that the multi-time-scale dimension
        # is last
        state_time_courses = np.rollaxis(state_time_courses, 0, 3)

        # Number of samples to simulate
        n_samples = state_time_courses.shape[0]

        # Initialise array to hold data
        data = np.zeros([n_samples, self.n_channels])
        self.instantaneous_covs = np.zeros(
            [n_samples, self.n_channels, self.n_channels]
        )

        # Loop through all unique combinations of states
        for time_courses in np.unique(state_time_courses, axis=0):

            # Extract the different time courses
            alpha = time_courses[:, 0]
            gamma = time_courses[:, 1]

            # Mean, standard deviation, FC for this combination of time courses
            mu = np.sum(self.means * alpha[:, np.newaxis], axis=0)
            G = np.diag(np.sum(self.stds * alpha[:, np.newaxis], axis=0))
            F = np.sum(self.fcs * gamma[:, np.newaxis, np.newaxis], axis=0)

            # Calculate covariance matrix from the standard deviation and FC
            sigma = G @ F @ G

            self.instantaneous_covs[
                np.all(state_time_courses == time_courses, axis=(1, 2))
            ] = sigma

            # Generate data for the time points that this combination of states
            # is active
            data[
                np.all(state_time_courses == time_courses, axis=(1, 2))
            ] = self._rng.multivariate_normal(
                mu,
                sigma,
                size=np.count_nonzero(
                    np.all(state_time_courses == time_courses, axis=(1, 2))
                ),
            )

        # Add an error to the data at all time points
        data += self._rng.normal(scale=self.observation_error, size=data.shape)

        return data.astype(np.float32)


class MSubj_MVN(MVN):
    """Class that generates data from a multivariate normal distribution for multiple subjects.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    n_modes : int
        Number of modes.
    n_subjects : int
        Number of subjects.
    means : np.ndarray or str
        Group mean vector for each mode, shape should be (n_modes, n_channels).
        Either a numpy array or 'zero' or 'random'.
    covariances : np.ndarray or str
        Group covariance matrix for each mode, shape should be (n_modes,
        n_channels, n_channels). Either a numpy array or 'random'.
    subject_maps_std : float
        Standard deviation when generating subject specific means and covariances
        from the group means and covariances.
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for the random number generator.
    """

    def __init__(
        self,
        means,
        covariances,
        subject_maps_std=0.01,
        n_modes=None,
        n_channels=None,
        n_subjects=1,
        observation_error=0.0,
        random_seed=None,
    ):
        super().__init__(
            means=means,
            covariances=covariances,
            n_modes=n_modes,
            n_channels=n_channels,
            observation_error=observation_error,
            random_seed=random_seed,
        )
        self.n_subjects = n_subjects
        self.subject_maps_std = subject_maps_std

        # Simulate means and covariances for each subject
        self.subject_means = self.create_subject_means(means)
        self.subject_covariances = self.create_subject_covariances(covariances)

    def create_subject_means(self, option):
        if option == "zero":
            subject_means = np.zeros([self.n_subjects, self.n_modes, self.n_channels])
        elif option == "random":
            subject_means = self._rng.normal(
                loc=self.means,
                scale=self.subject_maps_std,
                size=(self.n_subjects, self.n_modes, self.n_channels),
            )
        else:
            raise ValueError("means must be 'zero' or 'random'.")
        return subject_means

    def create_subject_covariances(self, option, eps=1e-6):
        if option == "random":
            # Add subject specific perturbation to W matrices
            subject_W = self._rng.normal(
                loc=self.W,
                scale=self.subject_maps_std,
                size=(self.n_subjects, self.n_modes, self.n_channels, self.n_channels),
            )
            # A small value to add to the diagonal to ensure the covariances are invertible
            eps = (
                np.tile(np.eye(self.n_channels), [self.n_subjects, self.n_modes, 1, 1])
                * eps
            )
            subject_covariances = subject_W @ subject_W.transpose([0, 1, 3, 2]) + eps
        else:
            raise ValueError("covariances must be 'random'.")

        return subject_covariances

    def simulate_subject_data(self, subject, mode_time_course):
        """Simulate single subject data."""
        n_samples = mode_time_course.shape[0]

        # Initialise array to hold data
        data = np.zeros((n_samples, self.n_channels))

        # Loop through all unique combinations of modes
        for alpha in np.unique(mode_time_course, axis=0):

            # Mean and covariance for this combination of modes
            mu = np.sum(self.subject_means[subject] * alpha[:, None], axis=0)
            sigma = np.sum(
                self.subject_covariances[subject] * alpha[:, None, None], axis=0
            )

            # Generate data for the time points that this combination of modes is active
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

    def simulate_multi_subject_data(self, mode_time_courses):
        """Simulates data.

        Parameters
        ----------
        mode_time_courses : np.ndarray
            It contains n_subjects time courses. Shape is (n_subjects, n_samples, n_modes).

        Returns
        -------
        np.ndarray
            Simulated data for subjects. Shape is (n_subjects, n_samples, n_channels).
        """
        # Initialise list to hold data
        data = []
        for subject in range(self.n_subjects):
            data.append(self.simulate_subject_data(subject, mode_time_courses[subject]))
        data = np.array(data)
        return data
