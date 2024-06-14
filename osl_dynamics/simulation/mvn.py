"""Multivariate normal observation model.

"""

import numpy as np

from osl_dynamics import array_ops


class MVN:
    """Class that generates data from a multivariate normal distribution.

    Parameters
    ----------
    means : np.ndarray or str
        Mean vector for each mode, shape should be (n_modes, n_channels).
        Either a numpy array or :code:`'zero'` or :code:`'random'`.
    covariances : np.ndarray or str
        Covariance matrix for each mode, shape should be (n_modes, n_channels,
        n_channels). Either a numpy array or :code:`'random'`.
    n_modes : int, optional
        Number of modes.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
    """

    def __init__(
        self,
        means,
        covariances,
        n_modes=None,
        n_channels=None,
        n_covariances_act=1,
        observation_error=0.0,
    ):
        self.n_covariances_act = n_covariances_act
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
                    "n_modes and n_channels must be passed."
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
            means = np.random.normal(
                mu,
                sigma,
                size=[self.n_modes, self.n_channels],
            )
        else:
            raise ValueError("means must be a np.array or 'zero' or 'random'.")
        return means

    def create_covariances(self, option, activation_strength=1, eps=1e-6):
        if option == "random":
            # Randomly sample the elements of W from a normal distribution
            W = np.random.normal(
                0, 0.1, size=[self.n_modes, self.n_channels, self.n_channels]
            )

            # Add a large activation to a small number of the channels at random
            activation_strength_multipliers = np.linspace(1, 5, self.n_covariances_act)
            for j in range(self.n_covariances_act):
                n_active_channels = max(1, 2 * self.n_channels // self.n_modes)
                for i in range(self.n_modes):
                    active_channels = np.unique(
                        np.random.randint(
                            0,
                            self.n_channels,
                            size=n_active_channels,
                        )
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
        """Simulate time series data.

        Parameters
        ----------
        state_time_course : np.ndarray
            2D array containing state activations.
            Shape must be (n_samples, n_states).

        Returns
        -------
        data : np.ndarray
            Time series data. Shape is (n_samples, n_channels).
        """
        n_samples = state_time_course.shape[0]

        # Initialise array to hold data
        data = np.zeros((n_samples, self.n_channels))

        # Loop through all unique combinations of modes
        for alpha in np.unique(state_time_course, axis=0):
            # Mean and covariance for this combination of modes
            mu = np.sum(self.means * alpha[:, np.newaxis], axis=0)
            sigma = np.sum(
                self.covariances * alpha[:, np.newaxis, np.newaxis],
                axis=0,
            )

            # Generate data for the time points that this combination of
            # modes is active
            data[np.all(state_time_course == alpha, axis=1)] = (
                np.random.multivariate_normal(
                    mu,
                    sigma,
                    size=np.count_nonzero(np.all(state_time_course == alpha, axis=1)),
                )
            )

        # Add an error to the data at all time points
        data += np.random.normal(scale=self.observation_error, size=data.shape)

        return data.astype(np.float32)

    def get_instantaneous_covariances(self, state_time_course):
        """Get the ground truth covariance at each time point.

        Parameters
        ----------
        state_time_course : np.ndarray
            2D array containing state activations.
            Shape must be (n_samples, n_states).

        Returns
        -------
        inst_covs : np.ndarray
            Instantaneous covariances.
            Shape is (n_samples, n_channels, n_channels).
        """
        # Initialise an array to hold the data
        n_samples = state_time_course.shape[0]
        inst_covs = np.empty([n_samples, self.n_channels, self.n_channels])

        # Loop through all unique combinations of modes
        for alpha in np.unique(state_time_course, axis=0):
            # Covariance for this combination of modes
            sigma = np.sum(
                self.covariances * alpha[:, np.newaxis, np.newaxis],
                axis=0,
            )
            inst_covs[np.all(state_time_course == alpha, axis=1)] = sigma

        return inst_covs.astype(np.float32)


class MDyn_MVN(MVN):
    """Class that generates data from a multivariate normal distribution.

    Multi-time-scale version of MVN.

    Parameters
    ----------
    means : np.ndarray or str
        Mean vector for each mode, shape should be (n_modes, n_channels).
        Either a numpy array or :code:`'zero'` or :code:`'random'`.
    covariances : np.ndarray or str
        Covariance matrix for each mode, shape should be (n_modes, n_channels,
        n_channels). Either a numpy array or :code:`'random'`.
    n_modes : int, optional
        Number of modes.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
    """

    def __init__(
        self,
        means,
        covariances,
        n_modes=None,
        n_channels=None,
        n_covariances_act=1,
        observation_error=0.0,
    ):
        super().__init__(
            means=means,
            covariances=covariances,
            n_modes=n_modes,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
            observation_error=observation_error,
        )

        # Get the stds and corrs from self.covariance
        self.stds = array_ops.cov2std(self.covariances)
        self.corrs = array_ops.cov2corr(self.covariances)

    def simulate_data(self, state_time_courses):
        """Simulates data.

        Parameters
        ----------
        state_time_courses : np.ndarray
            Should contain two time courses: one for the mean and standard
            deviations and another for functional connectiivty. Shape is
            (2, n_samples, n_modes).

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape is (n_samples, n_channels).
        """
        # Reshape state_time_courses so that the multi-time-scale dimension
        # is last
        state_time_courses = np.rollaxis(state_time_courses, 0, 3)

        # Number of samples to simulate
        n_samples = state_time_courses.shape[0]

        # Initialise array to hold data
        data = np.zeros([n_samples, self.n_channels])

        # Loop through all unique combinations of states
        for time_courses in np.unique(state_time_courses, axis=0):
            # Extract the different time courses
            alpha = time_courses[:, 0]
            beta = time_courses[:, 1]

            # Mean, standard deviation, corr for this combination of time courses
            mu = np.sum(self.means * alpha[:, np.newaxis], axis=0)
            G = np.diag(np.sum(self.stds * alpha[:, np.newaxis], axis=0))
            F = np.sum(self.corrs * beta[:, np.newaxis, np.newaxis], axis=0)

            # Calculate covariance matrix from the standard deviation and corr
            sigma = G @ F @ G

            # Generate data for the time points that this combination of states
            # is active
            data[np.all(state_time_courses == time_courses, axis=(1, 2))] = (
                np.random.multivariate_normal(
                    mu,
                    sigma,
                    size=np.count_nonzero(
                        np.all(state_time_courses == time_courses, axis=(1, 2))
                    ),
                )
            )

        # Add an error to the data at all time points
        data += np.random.normal(scale=self.observation_error, size=data.shape)

        return data.astype(np.float32)

    def get_instantaneous_covariances(self, state_time_courses):
        """Get the ground truth covariance at each time point.

        Parameters
        ----------
        state_time_courses : np.ndarray
            Should contain two time courses: one for the mean and standard
            deviations and another for functional connectiivty. Shape is
            (2, n_samples, n_modes).

        Returns
        -------
        inst_covs : np.ndarray
            Instantaneous covariances.
            Shape is (n_samples, n_channels, n_channels).
        """
        # Reshape state_time_courses so that the multi-time-scale dimension
        # is last
        state_time_courses = np.rollaxis(state_time_courses, 0, 3)

        # Number of samples to simulate
        n_samples = state_time_courses.shape[0]

        # Initialise an array to hold data
        inst_covs = np.empty([n_samples, self.n_channels, self.n_channels])

        # Loop through all unique combinations of states
        for time_courses in np.unique(state_time_courses, axis=0):
            # Extract the different time courses
            alpha = time_courses[:, 0]
            beta = time_courses[:, 1]

            # Mean, standard deviation, corr for this combination of time courses
            G = np.diag(np.sum(self.stds * alpha[:, np.newaxis], axis=0))
            F = np.sum(self.corrs * beta[:, np.newaxis, np.newaxis], axis=0)

            # Calculate covariance matrix from the standard deviation and corr
            sigma = G @ F @ G

            inst_covs[np.all(state_time_courses == time_courses, axis=(1, 2))] = sigma

        return inst_covs.astype(np.float32)


class MSess_MVN(MVN):
    """Class that generates data from a multivariate normal distribution for
    multiple sessions.

    Parameters
    ----------
    session_means : np.ndarray or str
        Mean vector for each mode for each session, shape should be
        (n_sessions, n_modes, n_channels). Either a numpy array or
        :code:`'zero'` or :code:`'random'`.
    session_covariances : np.ndarray or str
        Covariance matrix for each mode for each session, shape should
        be (n_sessions, n_modes, n_channels, n_channels). Either a numpy
        array or :code:`'random'`.
    n_modes : int, optional
        Number of modes.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    embedding_vectors : np.ndarray, optional
        Embedding vectors for each session, shape should be
        (n_sessions, embeddings_dim).
    n_sessions : int, optional
        Number of sessions.
    embeddings_dim : int, optional
        Dimension of embeddings.
    spatial_embeddings_dim : int, optional
        Dimension of spatial embeddings.
    embeddings_scale : float, optional
        Standard deviation when generating embeddings with a normal
        distribution.
    n_groups : int, optional
        Number of groups when generating embeddings.
    between_group_scale : float, optional
        Standard deviation when generating centroids of groups of
        embeddings.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
    """

    def __init__(
        self,
        session_means,
        session_covariances,
        n_modes=None,
        n_channels=None,
        n_covariances_act=1,
        embedding_vectors=None,
        n_sessions=None,
        embeddings_dim=None,
        spatial_embeddings_dim=None,
        embeddings_scale=None,
        n_groups=None,
        between_group_scale=None,
        observation_error=0.0,
    ):
        self.n_covariances_act = n_covariances_act
        self.observation_error = observation_error
        self.embeddings_dim = embeddings_dim
        self.spatial_embeddings_dim = spatial_embeddings_dim
        self.embeddings_scale = embeddings_scale
        self.n_groups = n_groups
        self.between_group_scale = between_group_scale

        if embedding_vectors is not None:
            n_sessions = embedding_vectors.shape[0]
            self.n_sessions = n_sessions
            embeddings_dim = embedding_vectors.shape[1]
            self.embeddings_dim = embeddings_dim

        # Both the session means and covariances were passed as numpy arrays
        if isinstance(session_means, np.ndarray) and isinstance(
            session_covariances, np.ndarray
        ):
            if session_means.ndim != 3:
                raise ValueError(
                    "session_means must have shape (n_sessions, n_modes, n_channels)."
                )
            if session_covariances.ndim != 4:
                raise ValueError(
                    "session_covariances must have shape "
                    "(n_sessions, n_modes, n_channels, n_channels)."
                )
            if session_means.shape[0] != session_covariances.shape[0]:
                raise ValueError(
                    "session_means and session_covariances have a different  "
                    "number of arrays."
                )
            if session_means.shape[1] != session_covariances.shape[1]:
                raise ValueError(
                    "session_means and session_covariances have a different "
                    "number of modes."
                )
            if session_means.shape[2] != session_covariances.shape[2]:
                raise ValueError(
                    "session_means and session_covariances have a different "
                    "number of channels."
                )
            self.n_sessions = session_means.shape[0]
            self.n_modes = session_means.shape[1]
            self.n_channels = session_means.shape[2]
            self.n_groups = None
            self.group_centroids = None
            self.between_group_scale = None
            self.embeddings_dim = None
            self.spatial_embeddings_dim = None
            self.embeddings_scale = None

            self.group_means = None
            self.session_means = session_means

            self.group_covariances = None
            self.session_covariances = session_covariances

        # Only the session means were passed as a numpy array
        elif isinstance(session_means, np.ndarray) and not isinstance(
            session_covariances, np.ndarray
        ):
            self.n_sessions = session_means.shape[0]
            self.n_modes = session_means.shape[1]
            self.n_channels = session_means.shape[2]

            self.validate_embedding_parameters(embedding_vectors)
            self.create_embeddings(embedding_vectors)

            self.group_means = None
            self.session_means = session_means

            self.group_covariances = super().create_covariances(session_covariances)
            self.session_covariances = self.create_session_covariances()

        # Only the session covariances were passed as a numpy array
        elif not isinstance(session_means, np.ndarray) and isinstance(
            session_covariances, np.ndarray
        ):
            self.n_sessions = session_covariances.shape[0]
            self.n_modes = session_covariances.shape[1]
            self.n_channels = session_covariances.shape[2]

            if not session_means == "zero":
                self.validate_embedding_parameters(embedding_vectors)
                self.create_embeddings(embedding_vectors)

            self.group_means = super().create_means(session_means)
            self.session_means = self.create_session_means(session_means)

            self.group_covariances = None
            self.session_covariances = session_covariances

        # Neither session means or nor covariances were passed as numpy arrays
        elif not isinstance(session_means, np.ndarray) and not isinstance(
            session_covariances, np.ndarray
        ):
            if n_sessions is None or n_modes is None or n_channels is None:
                raise ValueError(
                    "If we are generating array means and covariances, "
                    "n_sessions, n_modes, n_channels must be passed."
                )

            self.n_sessions = n_sessions
            self.n_modes = n_modes
            self.n_channels = n_channels

            self.validate_embedding_parameters(embedding_vectors)
            self.create_embeddings(embedding_vectors)

            self.group_means = super().create_means(session_means)
            self.session_means = self.create_session_means(session_means)

            self.group_covariances = super().create_covariances(session_covariances)
            self.session_covariances = self.create_session_covariances()

    def validate_embedding_parameters(self, embedding_vectors):
        if embedding_vectors is None:
            if self.embeddings_dim is None:
                raise ValueError(
                    "Session means or covariances not passed, please pass "
                    "'embeddings_dim'."
                )
            if self.spatial_embeddings_dim is None:
                raise ValueError(
                    "Session means or covariances not passed, please pass "
                    "'spatial_embeddings_dim'."
                )
            if self.embeddings_scale is None:
                raise ValueError(
                    "Session means or covariances not passed, please pass "
                    "'embeddings_scale'."
                )
            if self.n_groups is None:
                raise ValueError(
                    "Session means or covariances not passed, please pass 'n_groups'."
                )
            if self.between_group_scale is None:
                raise ValueError(
                    "Session means or covariances not passed, please pass "
                    "'between_group_scale'."
                )

    def create_embeddings(self, embedding_vectors):
        if embedding_vectors is None:
            # Assign groups to sessions
            assigned_groups = np.random.choice(self.n_groups, self.n_sessions)
            self.group_centroids = np.random.normal(
                scale=self.between_group_scale,
                size=[self.n_groups, self.embeddings_dim],
            )

            embeddings = np.zeros([self.n_sessions, self.embeddings_dim])
            for i in range(self.n_groups):
                group_mask = assigned_groups == i
                embeddings[group_mask] = np.random.multivariate_normal(
                    mean=self.group_centroids[i],
                    cov=self.embeddings_scale * np.eye(self.embeddings_dim),
                    size=[np.sum(group_mask)],
                )

            self.assigned_groups = assigned_groups
            self.embeddings = embeddings
        else:
            self.embeddings = embedding_vectors

    def create_linear_transform(self, input_dim, output_dim, scale=0.1):
        linear_transform = np.random.normal(
            scale=scale,
            size=(output_dim, input_dim),
        )
        return linear_transform / np.sqrt(
            np.sum(np.square(linear_transform), axis=-1, keepdims=True)
        )

    def create_session_means_deviations(self):
        means_spatial_embeddings_lienar_transform = self.create_linear_transform(
            self.n_channels, self.spatial_embeddings_dim
        )
        self.means_spatial_embeddings = (
            means_spatial_embeddings_lienar_transform @ self.group_means.T
        ).T

        # Match the shapes for concatenation
        concat_array_embeddings = np.broadcast_to(
            self.embeddings[:, None, :],
            (
                self.n_sessions,
                self.n_modes,
                self.embeddings_dim,
            ),
        )
        concat_means_spatial_embeddings = np.broadcast_to(
            self.means_spatial_embeddings[None, :, :],
            (
                self.n_sessions,
                self.n_modes,
                self.spatial_embeddings_dim,
            ),
        )
        self.means_concat_embeddings = np.concatenate(
            [concat_array_embeddings, concat_means_spatial_embeddings], axis=-1
        )
        means_linear_transform = self.create_linear_transform(
            self.embeddings_dim + self.spatial_embeddings_dim,
            self.n_channels,
        )
        self.means_deviations = np.squeeze(
            means_linear_transform[None, None, ...]
            @ self.means_concat_embeddings[..., None]
        )

    def create_session_covariances_deviations(self):
        covariances_spatial_embeddings_linear_transform = self.create_linear_transform(
            self.n_channels * (self.n_channels + 1) // 2, self.spatial_embeddings_dim
        )
        group_cholesky_covariances = np.linalg.cholesky(self.group_covariances)
        m, n = np.tril_indices(self.n_channels)
        flattened_group_cholesky_covariances = group_cholesky_covariances[:, m, n]
        self.covariances_spatial_embeddings = (
            covariances_spatial_embeddings_linear_transform
            @ flattened_group_cholesky_covariances.T
        ).T

        # Match the shapes for concatenation
        concat_array_embeddings = np.broadcast_to(
            self.embeddings[:, None, :],
            (
                self.n_sessions,
                self.n_modes,
                self.embeddings_dim,
            ),
        )
        concat_covarainces_spatial_embeddings = np.broadcast_to(
            self.covariances_spatial_embeddings[None, :, :],
            (
                self.n_sessions,
                self.n_modes,
                self.spatial_embeddings_dim,
            ),
        )
        self.covariances_concat_embeddings = np.concatenate(
            [concat_array_embeddings, concat_covarainces_spatial_embeddings],
            axis=-1,
        )
        covariances_linear_transform = self.create_linear_transform(
            self.embeddings_dim + self.spatial_embeddings_dim,
            self.n_channels * (self.n_channels + 1) // 2,
        )
        self.flattened_covariances_cholesky_deviations = np.squeeze(
            covariances_linear_transform[None, None, ...]
            @ self.covariances_concat_embeddings[..., None]
        )

    def create_session_means(self, option):
        if option == "zero":
            session_means = np.zeros([self.n_sessions, self.n_modes, self.n_channels])
        else:
            self.create_session_means_deviations()
            session_means = self.group_means[None, ...] + self.means_deviations
        return session_means

    def create_session_covariances(self, eps=1e-6):
        self.create_session_covariances_deviations()
        group_cholesky_covariances = np.linalg.cholesky(self.group_covariances)
        m, n = np.tril_indices(self.n_channels)
        flattened_group_cholesky_covariances = group_cholesky_covariances[:, m, n]
        flattened_session_cholesky_covariances = (
            flattened_group_cholesky_covariances[None, ...]
            + self.flattened_covariances_cholesky_deviations
        )

        session_cholesky_covariances = np.zeros(
            [self.n_sessions, self.n_modes, self.n_channels, self.n_channels]
        )
        for i in range(self.n_sessions):
            for j in range(self.n_modes):
                session_cholesky_covariances[i, j, m, n] = (
                    flattened_session_cholesky_covariances[i, j]
                )

        session_covariances = session_cholesky_covariances @ np.transpose(
            session_cholesky_covariances, (0, 1, 3, 2)
        )

        # A small value to add to the diagonal to ensure the covariances
        # are invertible
        session_covariances += eps * np.eye(self.n_channels)

        return session_covariances

    def simulate_session_data(self, session, mode_time_course):
        """Simulate single session data.

        Parameters
        ----------
        session : int
            Session number.
        mode_time_course : np.ndarray
            Mode time course. Shape is (n_samples, n_modes).

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape is (n_samples, n_channels).
        """
        # Initialise array to hold data
        n_samples = mode_time_course.shape[0]
        data = np.zeros((n_samples, self.n_channels))

        # Loop through all unique combinations of modes
        for alpha in np.unique(mode_time_course, axis=0):
            # Mean and covariance for this combination of modes
            mu = np.sum(self.session_means[session] * alpha[:, None], axis=0)
            sigma = np.sum(
                self.session_covariances[session] * alpha[:, None, None], axis=0
            )

            # Generate data for the time points that this combination of
            # modes is active
            data[np.all(mode_time_course == alpha, axis=1)] = (
                np.random.multivariate_normal(
                    mu,
                    sigma,
                    size=np.count_nonzero(np.all(mode_time_course == alpha, axis=1)),
                )
            )

        # Add an error to the data at all time points
        data += np.random.normal(scale=self.observation_error, size=data.shape)

        return data.astype(np.float32)

    def get_session_instantaneous_covariances(self, session, mode_time_course):
        """Get ground truth covariances at each time point for a particular session.

        Parameters
        ----------
        session : int
            Session number.
        mode_time_course : np.ndarray
            Mode time course. Shape is (n_samples, n_modes).

        Returns
        -------
        inst_covs : np.ndarray
            Instantaneous covariances for an session.
            Shape is (n_samples, n_channels, n_channels).
        """
        # Initialise array to hold data
        n_samples = mode_time_course.shape[0]
        inst_covs = np.zeros([n_samples, self.n_channels, self.n_channels])

        # Loop through all unique combinations of modes
        for alpha in np.unique(mode_time_course, axis=0):
            # Covariance for this combination of modes
            sigma = np.sum(
                self.session_covariances[session] * alpha[:, None, None], axis=0
            )
            inst_covs[np.all(mode_time_course == alpha, axis=1)] = sigma

        return inst_covs.astype(np.float32)

    def get_instantaneous_covariances(self, mode_time_courses):
        """Get ground truth covariance at each time point for each session.

        Parameters
        ----------
        mode_time_courses : np.ndarray
            Mode time courses.
            Shape is (n_sessions, n_samples, n_modes).

        Returns
        -------
        inst_covs : np.ndarray
            Instantaneous covariances.
            Shape is (n_sessions, n_samples, n_channels, n_channels).
        """
        inst_covs = []
        for session in range(self.n_sessions):
            inst_covs.append(
                self.get_session_instantaneous_covariances(
                    session, mode_time_courses[session]
                )
            )
        return np.array(inst_covs)

    def simulate_multi_session_data(self, mode_time_courses):
        """Simulates data.

        Parameters
        ----------
        mode_time_courses : np.ndarray
            It contains n_sessions time courses.
            Shape is (n_sessions, n_samples, n_modes).

        Returns
        -------
        data : np.ndarray
            Simulated data for sessions.
            Shape is (n_sessions, n_samples, n_channels).
        """
        data = []
        for session in range(self.n_sessions):
            data.append(self.simulate_session_data(session, mode_time_courses[session]))
        return np.array(data)
