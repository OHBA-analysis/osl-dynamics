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
    n_covariances_act : int
        Number of iterations to add activations to covariance matrices.
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
        n_covariances_act=1,
        observation_error=0.0,
        random_seed=None,
    ):
        self._rng = np.random.default_rng(random_seed)
        self.n_covariances_act = n_covariances_act
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

    def create_means(self, option, mu=0, sigma=0.2):
        if option == "zero":
            means = np.zeros([self.n_modes, self.n_channels])
        elif option == "random":
            means = self._rng.normal(mu, sigma, size=[self.n_modes, self.n_channels])
        else:
            raise ValueError("means must be a np.array or 'zero' or 'random'.")
        return means

    def create_covariances(self, option, activation_strength=1, eps=1e-6):
        if option == "random":
            # Randomly sample the elements of W from a normal distribution
            W = self._rng.normal(
                0, 0.1, size=[self.n_modes, self.n_channels, self.n_channels]
            )

            # Add a large activation to a small number of the channels at random
            activation_strength_multipliers = np.linspace(1, 5, self.n_covariances_act)
            for j in range(self.n_covariances_act):
                n_active_channels = max(1, 2 * self.n_channels // self.n_modes)
                for i in range(self.n_modes):
                    active_channels = np.unique(
                        self._rng.integers(0, self.n_channels, size=n_active_channels)
                    )
                    W[i, active_channels] += (
                        activation_strength_multipliers[j]
                        * activation_strength
                        / self.n_channels
                    )

            # A small value to add to the diagonal to ensure the covariances are
            # invertible
            eps = np.tile(np.eye(self.n_channels), [self.n_modes, 1, 1]) * eps

            # Calculate the covariances
            covariances = W @ W.transpose([0, 2, 1]) + eps

        else:
            raise ValueError("covariances must be a np.ndarray or 'random'.")

        return covariances

    def simulate_data(self, state_time_course):
        n_samples = state_time_course.shape[0]

        # Initialise array to hold data
        data = np.zeros((n_samples, self.n_channels))
        self.instantaneous_covs = np.zeros(
            [n_samples, self.n_channels, self.n_channels]
        )

        # Loop through all unique combinations of modes
        for alpha in np.unique(state_time_course, axis=0):

            # Mean and covariance for this combination of modes
            mu = np.sum(self.means * alpha[:, np.newaxis], axis=0)
            sigma = np.sum(self.covariances * alpha[:, np.newaxis, np.newaxis], axis=0)

            self.instantaneous_covs[np.all(state_time_course == alpha, axis=1)] = sigma

            # Generate data for the time points that this combination of modes is
            # active
            data[
                np.all(state_time_course == alpha, axis=1)
            ] = self._rng.multivariate_normal(
                mu,
                sigma,
                size=np.count_nonzero(np.all(state_time_course == alpha, axis=1)),
            )

        # Add an error to the data at all time points
        data += self._rng.normal(scale=self.observation_error, size=data.shape)

        return data.astype(np.float32)


class MDyn_MVN(MVN):
    """Class that generates data from a multivariate normal distribution.

    Multi-time-scale version of MVN.

    Parameters
    ----------
    means : np.ndarray or str
        Mean vector for each mode, shape should be (n_modes, n_channels).
        Either a numpy array or 'zero' or 'random'.
    covariances : np.ndarray or str
        Covariance matrix for each mode, shape should be (n_modes,
        n_channels, n_channels). Either a numpy array or 'random'.
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    n_covariances_act : int
        Number of iterations to add activations to covariance matrices.
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
        n_covariances_act=1,
        observation_error=0.0,
        random_seed=None,
    ):
        super().__init__(
            means=means,
            covariances=covariances,
            n_modes=n_modes,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
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
    subject_means : np.ndarray or str
        Subject mean vector for each mode for each subject, shape should be
        (n_subjects, n_modes, n_channels). Either a numpy array or 'zero' or 'random'.
    subject_covariances : np.ndarray or str
        Subject covariance matrix for each mode for each subject, shape should be
        (n_subjects, n_modes, n_channels, n_channels). Either a numpy array or 'random'.
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    n_covariances_act : int
        Number of iterations to add activations to covariance matrices.

    n_subjects : int
        Number of subjects.
    n_subject_embedding_dim : int
        Dimension of subject embeddings.
    n_mode_embedding_dim : int
        Dimension of mode embeddings.
    subject_embedding_scale : float
        Standard deviation when generating subject embeddings with a normal distribution.
    n_groups : int
        Number of groups of subjects when generating subject embeddings.
    between_group_scale : float
        Standard deviation when generating centroids of groups of subject embeddings.
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for the random number generator.
    """

    def __init__(
        self,
        subject_means,
        subject_covariances,
        n_modes=None,
        n_channels=None,
        n_covariances_act=1,
        n_subjects=None,
        n_subject_embedding_dim=None,
        n_mode_embedding_dim=None,
        subject_embedding_scale=None,
        n_groups=None,
        between_group_scale=None,
        observation_error=0.0,
        random_seed=None,
    ):

        self._rng = np.random.default_rng(random_seed)
        self.n_covariances_act = n_covariances_act
        self.observation_error = observation_error
        self.instantaneous_covs = None
        self.n_subject_embedding_dim = n_subject_embedding_dim
        self.n_mode_embedding_dim = n_mode_embedding_dim
        self.subject_embedding_scale = subject_embedding_scale
        self.n_groups = n_groups
        self.between_group_scale = between_group_scale

        # Both the subject means and covariances were passed as numpy arrays
        if isinstance(subject_means, np.ndarray) and isinstance(
            subject_covariances, np.ndarray
        ):
            if subject_means.ndim != 3:
                raise ValueError(
                    "subject_means must have shape (n_subjects, n_modes, n_channels)."
                )
            if subject_covariances.ndim != 4:
                raise ValueError(
                    "subject_covariances must have shape "
                    + "(n_subjects, n_modes, n_channels, n_channels)."
                )
            if subject_means.shape[0] != subject_covariances.shape[0]:
                raise ValueError(
                    "subject_means and subject_covariances have a different number of subjects."
                )
            if subject_means.shape[1] != subject_covariances.shape[1]:
                raise ValueError(
                    "subject_means and subject_covariances have a different number of modes."
                )
            if subject_means.shape[2] != subject_covariances.shape[2]:
                raise ValueError(
                    "subject_means and subject_covariances have a different number of channels."
                )
            self.n_subjects = subject_means.shape[0]
            self.n_modes = subject_means.shape[1]
            self.n_channels = subject_means.shape[2]
            self.n_groups = None
            self.between_group_scale = None
            self.n_subject_embedding_dim = None
            self.n_mode_embedding_dim = None
            self.subject_embedding_scale = None

            self.group_means = None
            self.subject_means = subject_means

            self.group_covariances = None
            self.subject_covariances = subject_covariances

        # Only the subject means were passed as a numpy array
        elif isinstance(subject_means, np.ndarray) and not isinstance(
            subject_covariances, np.ndarray
        ):
            self.n_subjects = subject_means.shape[0]
            self.n_modes = subject_means.shape[1]
            self.n_channels = subject_means.shape[2]

            self.validate_subject_embedding_parameters()
            self.create_subject_embeddings()

            self.group_means = None
            self.subject_means = subject_means

            self.group_covariances = super().create_covariances(subject_covariances)
            self.subject_covariances = self.create_subject_covariances()

        # Only the subject covariances were passed as a numpy array
        elif not isinstance(subject_means, np.ndarray) and isinstance(
            subject_covariances, np.ndarray
        ):
            self.n_subjects = subject_covariances.shape[0]
            self.n_modes = subject_covariances.shape[1]
            self.n_channels = subject_covariances.shape[2]

            self.validate_subject_embedding_parameters()
            self.create_subject_embeddings()

            self.group_means = super().create_means(subject_means)
            self.subject_means = self.create_subject_means(subject_means)

            self.group_covariances = None
            self.subject_covariances = subject_covariances

        # Neither subject means or nor covariances were passed as numpy arrays
        elif not isinstance(subject_means, np.ndarray) and not isinstance(
            subject_covariances, np.ndarray
        ):
            if n_subjects is None or n_modes is None or n_channels is None:
                raise ValueError(
                    "If we are generating subject means and covariances, "
                    + "n_subjects, n_modes, n_channels must be passed."
                )

            self.n_subjects = n_subjects
            self.n_modes = n_modes
            self.n_channels = n_channels

            self.validate_subject_embedding_parameters()
            self.create_subject_embeddings()

            self.group_means = super().create_means(subject_means)
            self.subject_means = self.create_subject_means(subject_means)

            self.group_covariances = super().create_covariances(subject_covariances)
            self.subject_covariances = self.create_subject_covariances()

    def validate_subject_embedding_parameters(self):
        if self.n_subject_embedding_dim is None:
            raise ValueError(
                "Subject means or covariances not passed, please pass 'n_subject_embedding_dim'!"
            )
        if self.n_mode_embedding_dim is None:
            raise ValueError(
                "Subject means or covariances not passed, please pass 'n_mode_embedding_dim'!"
            )
        if self.subject_embedding_scale is None:
            raise ValueError(
                "Subject means or covariances not passed, please pass 'subject_embedding_scale'!"
            )
        if self.n_groups is None:
            raise ValueError(
                "Subject means or covariances not passed, please pass 'n_groups'!"
            )
        if self.between_group_scale is None:
            raise ValueError(
                "Subject means or covariances not passed, please pass 'between_group_scale'!"
            )

    def create_subject_embeddings(self):
        # Assign groups to subjects
        assigned_groups = self._rng.choice(self.n_groups, self.n_subjects)
        group_locs = self._rng.normal(
            scale=self.between_group_scale,
            size=[self.n_groups, self.n_subject_embedding_dim],
        )

        subject_embeddings = np.zeros([self.n_subjects, self.n_subject_embedding_dim])
        for i in range(self.n_groups):
            group_mask = assigned_groups == i
            subject_embeddings[group_mask] = self._rng.multivariate_normal(
                mean=group_locs[i],
                cov=self.subject_embedding_scale * np.eye(self.n_subject_embedding_dim),
                size=[np.sum(group_mask)],
            )

        self.assigned_groups = assigned_groups
        self.subject_embeddings = subject_embeddings

    def create_linear_transform(self, input_dim, output_dim, scale=0.1):
        linear_transform = self._rng.normal(scale=scale, size=(output_dim, input_dim))
        return linear_transform / np.sqrt(
            np.sum(np.square(linear_transform), axis=-1, keepdims=True)
        )

    def create_subject_means_deviations(self):
        means_mode_embeddings_lienar_transform = self.create_linear_transform(
            self.n_channels, self.n_mode_embedding_dim
        )
        self.means_mode_embeddings = (
            means_mode_embeddings_lienar_transform @ self.group_means.T
        ).T

        # Match the shapes for concatenation
        concat_subject_embeddings = np.broadcast_to(
            self.subject_embeddings[:, None, :],
            (
                self.n_subjects,
                self.n_modes,
                self.n_subject_embedding_dim,
            ),
        )
        concat_means_mode_embeddings = np.broadcast_to(
            self.means_mode_embeddings[None, :, :],
            (
                self.n_subjects,
                self.n_modes,
                self.n_mode_embedding_dim,
            ),
        )
        self.means_concat_embeddings = np.concatenate(
            [concat_subject_embeddings, concat_means_mode_embeddings], axis=-1
        )
        means_linear_transform = self.create_linear_transform(
            self.n_subject_embedding_dim + self.n_mode_embedding_dim,
            self.n_channels,
        )
        self.means_deviations = np.squeeze(
            means_linear_transform[None, None, ...]
            @ self.means_concat_embeddings[..., None]
        )

    def create_subject_covariances_deviations(self):
        covariances_mode_embeddings_linear_transform = self.create_linear_transform(
            self.n_channels * (self.n_channels + 1) // 2, self.n_mode_embedding_dim
        )
        group_cholesky_covariances = np.linalg.cholesky(self.group_covariances)
        m, n = np.tril_indices(self.n_channels)
        flattened_group_cholesky_covariances = group_cholesky_covariances[:, m, n]
        self.covariances_mode_embeddings = (
            covariances_mode_embeddings_linear_transform
            @ flattened_group_cholesky_covariances.T
        ).T

        # Match the shapes for concatenation
        concat_subject_embeddings = np.broadcast_to(
            self.subject_embeddings[:, None, :],
            (
                self.n_subjects,
                self.n_modes,
                self.n_subject_embedding_dim,
            ),
        )
        concat_covarainces_mode_embeddings = np.broadcast_to(
            self.covariances_mode_embeddings[None, :, :],
            (
                self.n_subjects,
                self.n_modes,
                self.n_mode_embedding_dim,
            ),
        )
        self.covariances_concat_embeddings = np.concatenate(
            [concat_subject_embeddings, concat_covarainces_mode_embeddings], axis=-1
        )
        covariances_linear_transform = self.create_linear_transform(
            self.n_subject_embedding_dim + self.n_mode_embedding_dim,
            self.n_channels * (self.n_channels + 1) // 2,
        )
        self.flattened_covariances_cholesky_deviations = np.squeeze(
            covariances_linear_transform[None, None, ...]
            @ self.covariances_concat_embeddings[..., None]
        )

    def create_subject_means(self, option):
        if option == "zero":
            subject_means = np.zeros([self.n_subjects, self.n_modes, self.n_channels])
        else:
            self.create_subject_means_deviations()
            subject_means = self.group_means[None, ...] + self.means_deviations
        return subject_means

    def create_subject_covariances(self, eps=1e-6):
        self.create_subject_covariances_deviations()
        group_cholesky_covariances = np.linalg.cholesky(self.group_covariances)
        m, n = np.tril_indices(self.n_channels)
        flattened_group_cholesky_covariances = group_cholesky_covariances[:, m, n]
        flattened_subject_cholesky_covariances = (
            flattened_group_cholesky_covariances[None, ...]
            + self.flattened_covariances_cholesky_deviations
        )

        subject_cholesky_covariances = np.zeros(
            [self.n_subjects, self.n_modes, self.n_channels, self.n_channels]
        )
        for i in range(self.n_subjects):
            for j in range(self.n_modes):
                subject_cholesky_covariances[
                    i, j, m, n
                ] = flattened_subject_cholesky_covariances[i, j]

        subject_covariances = subject_cholesky_covariances @ np.transpose(
            subject_cholesky_covariances, (0, 1, 3, 2)
        )

        # A small value to add to the diagonal to ensure the covariances are invertible
        subject_covariances += eps * np.eye(self.n_channels)

        return subject_covariances

    def simulate_subject_data(self, subject, mode_time_course):
        """Simulate single subject data.

        Parameters
        ----------
        subject : int
            Subject number.
        mode_time_course : np.ndarray
            Mode time course. Shape is (n_samples, n_modes).

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape is (n_samples, n_channels).
        """
        n_samples = mode_time_course.shape[0]

        # Initialise array to hold data
        data = np.zeros((n_samples, self.n_channels))
        instantaneous_covs = np.zeros([n_samples, self.n_channels, self.n_channels])

        # Loop through all unique combinations of modes
        for alpha in np.unique(mode_time_course, axis=0):

            # Mean and covariance for this combination of modes
            mu = np.sum(self.subject_means[subject] * alpha[:, None], axis=0)
            sigma = np.sum(
                self.subject_covariances[subject] * alpha[:, None, None], axis=0
            )

            instantaneous_covs[np.all(mode_time_course == alpha, axis=1)] = sigma

            # Generate data for the time points that this combination of modes is active
            data[
                np.all(mode_time_course == alpha, axis=1)
            ] = self._rng.multivariate_normal(
                mu,
                sigma,
                size=np.count_nonzero(np.all(mode_time_course == alpha, axis=1)),
            )

        self.instantaneous_covs.append(instantaneous_covs)
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
        data = []
        self.instantaneous_covs = []
        for subject in range(self.n_subjects):
            data.append(self.simulate_subject_data(subject, mode_time_courses[subject]))

        self.instantaneous_covs = np.array(self.instantaneous_covs)
        return np.array(data)
