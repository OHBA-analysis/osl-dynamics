"""Observation models for simulations."""

from typing import Optional, Tuple, Union

import numpy as np

from osl_dynamics.utils import array_ops


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
        means: Union[np.ndarray, str],
        covariances: Union[np.ndarray, str],
        n_modes: Optional[int] = None,
        n_channels: Optional[int] = None,
        n_covariances_act: int = 1,
        observation_error: float = 0.0,
    ) -> None:
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
            raise ValueError("means and covariance arguments not passed correctly.")

    def create_means(
        self, option: str, mu: float = 0, sigma: float = 0.2
    ) -> np.ndarray:
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

    def create_covariances(
        self, option: str, activation_strength: float = 1, eps: float = 1e-6
    ) -> np.ndarray:
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

    def simulate_data(self, state_time_course: np.ndarray) -> np.ndarray:
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

    def get_instantaneous_covariances(
        self, state_time_course: np.ndarray
    ) -> np.ndarray:
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
        means: Union[np.ndarray, str],
        covariances: Union[np.ndarray, str],
        n_modes: Optional[int] = None,
        n_channels: Optional[int] = None,
        n_covariances_act: int = 1,
        observation_error: float = 0.0,
    ) -> None:
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

    def simulate_data(self, state_time_courses: np.ndarray) -> np.ndarray:
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

    def get_instantaneous_covariances(
        self, state_time_courses: np.ndarray
    ) -> np.ndarray:
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
    """Class that generates data from a multivariate normal distribution for multiple sessions.

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
        session_means: Union[np.ndarray, str],
        session_covariances: Union[np.ndarray, str],
        n_modes: Optional[int] = None,
        n_channels: Optional[int] = None,
        n_covariances_act: int = 1,
        embedding_vectors: Optional[np.ndarray] = None,
        n_sessions: Optional[int] = None,
        embeddings_dim: Optional[int] = None,
        spatial_embeddings_dim: Optional[int] = None,
        embeddings_scale: Optional[float] = None,
        n_groups: Optional[int] = None,
        between_group_scale: Optional[float] = None,
        observation_error: float = 0.0,
    ) -> None:
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

    def validate_embedding_parameters(
        self, embedding_vectors: Optional[np.ndarray]
    ) -> None:
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

    def create_embeddings(self, embedding_vectors: Optional[np.ndarray]) -> None:
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

    def create_linear_transform(
        self, input_dim: int, output_dim: int, scale: float = 0.1
    ) -> np.ndarray:
        linear_transform = np.random.normal(
            scale=scale,
            size=(output_dim, input_dim),
        )
        return linear_transform / np.sqrt(
            np.sum(np.square(linear_transform), axis=-1, keepdims=True)
        )

    def create_session_means_deviations(self) -> None:
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

    def create_session_covariances_deviations(self) -> None:
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

    def create_session_means(self, option: str) -> np.ndarray:
        if option == "zero":
            session_means = np.zeros([self.n_sessions, self.n_modes, self.n_channels])
        else:
            self.create_session_means_deviations()
            session_means = self.group_means[None, ...] + self.means_deviations
        return session_means

    def create_session_covariances(self, eps: float = 1e-6) -> np.ndarray:
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

    def simulate_session_data(
        self, session: int, mode_time_course: np.ndarray
    ) -> np.ndarray:
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

    def get_session_instantaneous_covariances(
        self, session: int, mode_time_course: np.ndarray
    ) -> np.ndarray:
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

    def get_instantaneous_covariances(
        self, mode_time_courses: np.ndarray
    ) -> np.ndarray:
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

    def simulate_multi_session_data(self, mode_time_courses: np.ndarray) -> np.ndarray:
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


class MAR:
    r"""Class that generates data from a multivariate autoregressive (MAR) model.

    A :math:`p`-order MAR model can be written as

    .. math::
        x_t = A_1 x_{t-1} + ... + A_p x_{t-p} + \epsilon_t

    where :math:`\epsilon_t \sim N(0, \Sigma)`. The MAR model is therefore
    parameterized by the MAR coefficients (:math:`A`) and covariance
    (:math:`\Sigma`).

    Parameters
    ----------
    coeffs : np.ndarray
        Array of MAR coefficients, :math:`A`. Shape must be
        (n_states, n_lags, n_channels, n_channels).
    covs : np.ndarray
        Covariance of the error :math:`\epsilon_t`. Shape must be
        (n_states, n_channels) or (n_states, n_channels, n_channels).

    Note
    ----
    This model is also known as VAR or MVAR.
    """

    def __init__(self, coeffs: np.ndarray, covs: np.ndarray) -> None:
        # Validation
        if coeffs.ndim != 4:
            raise ValueError(
                "coeffs must be a (n_states, n_lags, n_channels, n_channels) array."
            )

        if covs.ndim == 2:
            covs = np.array([np.diag(c) for c in covs])

        if coeffs.shape[0] != covs.shape[0]:
            raise ValueError("Different number of states in coeffs and covs passed.")

        if coeffs.shape[-1] != covs.shape[-1]:
            raise ValueError("Different number of channels in coeffs and covs passed.")

        # Model parameters
        self.coeffs = coeffs
        self.covs = covs
        self.order = coeffs.shape[1]

        # Number of states and channels
        self.n_states = coeffs.shape[0]
        self.n_channels = coeffs.shape[2]

    def simulate_data(self, state_time_course: np.ndarray) -> np.ndarray:
        """Simulate time series data.

        We simulate MAR data based on the hidden state at each time point.

        Parameters
        ----------
        state_time_course : np.ndarray
            State time course. Shape must be (n_samples, n_states).
            States must be mutually exclusive.

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape is (n_samples, n_channels).
        """
        n_samples = state_time_course.shape[0]
        data = np.empty([n_samples, self.n_channels])

        # Generate the noise term first
        for i in range(self.n_states):
            time_points_active = state_time_course[:, i] == 1
            n_time_points_active = np.count_nonzero(time_points_active)
            data[time_points_active] = np.random.multivariate_normal(
                np.zeros(self.n_channels),
                self.covs[i],
                size=n_time_points_active,
            )

        # Generate the MAR process
        for t in range(n_samples):
            state = state_time_course[t].argmax()
            for lag in range(min(t, self.order)):
                data[t] += np.dot(self.coeffs[state, lag], data[t - lag - 1])

        return data.astype(np.float32)


class OscillatoryBursts:
    """Oscillatory burst observation model.

    Generates sinusoidal oscillatory bursts at specified frequencies. Each mode
    has an associated frequency and a set of active channels defined by a
    channel activity matrix. During active periods (determined by the state
    time course), channels produce sinusoidal signals at the mode's frequency
    with a slowly varying phase offset.

    Parameters
    ----------
    n_modes : int
        Number of frequency modes.
    n_channels : int
        Number of channels.
    true_freqs : np.ndarray
        Frequencies for each mode in Hz. Shape: (n_modes,).
    channel_activity : np.ndarray
        Binary matrix indicating which channels are active for each mode.
        Shape: (n_modes, n_channels).
    sampling_frequency : float, optional
        Sampling frequency in Hz. Default: 100.
    snr : float, optional
        Signal-to-noise ratio. Default: 4.
    """

    def __init__(
        self,
        n_modes: int,
        n_channels: int,
        true_freqs: np.ndarray,
        channel_activity: np.ndarray,
        sampling_frequency: float = 100,
        snr: float = 4,
    ):
        self.n_modes = n_modes
        self.n_channels = n_channels
        self.true_freqs = np.asarray(true_freqs)
        self.channel_activity = np.asarray(channel_activity)
        self.sampling_frequency = sampling_frequency
        self.snr = snr

    def simulate_data(
        self,
        state_time_course: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate oscillatory burst data for a single subject.

        Parameters
        ----------
        state_time_course : np.ndarray
            One-hot encoded state time course. Shape: (n_samples, n_states).
            The first n_modes columns correspond to the oscillatory modes.
            Any additional columns (e.g. a "background" state) are ignored.

        Returns
        -------
        data : np.ndarray
            Simulated data with noise. Shape: (n_samples, n_channels).
        true_signal : np.ndarray
            Clean signal without noise. Shape: (n_samples, n_channels).
        """
        n_samples = state_time_course.shape[0]
        timestamps = np.arange(n_samples) / self.sampling_frequency

        # Generate sinusoidal activity for each mode and channel
        phase_diff = np.linspace(0, 0.25, self.n_channels)
        true_signal = np.zeros((n_samples, self.n_channels))

        for i in range(self.n_modes):
            active = state_time_course[:, i] == 1
            for j in range(self.n_channels):
                if self.channel_activity[i, j] == 1:
                    phase = (
                        (0.5 * np.sin(2 * np.pi * 0.005 * timestamps) + phase_diff[j])
                        * 2
                        * np.pi
                    )
                    true_signal[active, j] += np.sin(
                        2 * np.pi * self.true_freqs[i] * timestamps[active]
                        + phase[active]
                    )

        # Add noise
        noise_std = 1 / self.snr
        data = true_signal + np.random.normal(0, noise_std, true_signal.shape)

        return data, true_signal


class Poisson:
    """Class that generates Poisson time series data.

    The time series for each channel is a single Poisson observation. The rate
    of the poisson observation can be different for different states and
    channels.

    Parameters
    ----------
    rates : np.ndarray or str
        Rate vector for each mode, shape should be (n_states, n_channels).
        Either a numpy array or 'random'.
    n_channels : int
        Number of channels.
    n_modes : int
        Number of modes.
    """

    def __init__(
        self,
        rates: Union[np.ndarray, str],
        n_states: Optional[int] = None,
        n_channels: Optional[int] = None,
    ) -> None:
        if isinstance(rates, np.ndarray):
            self.n_states = rates.shape[0]
            self.n_channels = rates.shape[1]
            self.rates = rates

        elif not isinstance(rates, np.ndarray):
            if n_states is None or n_channels is None:
                raise ValueError(
                    "If we are generating rates, "
                    "n_states and n_channels must be passed."
                )
            self.n_states = n_states
            self.n_channels = n_channels
            self.rates = self.create_rates(rates)

    def create_rates(self, option: str, eps: float = 1e-2) -> np.ndarray:
        if option == "random":
            # Randomly sample the rates from a gamma distribution
            rates = np.random.gamma(
                shape=1.0, scale=1.1, size=(self.n_states, self.n_channels)
            )

            #  Add a large rate to a small number of the channels at random
            n_active_channels = max(1, self.n_channels // self.n_states)
            for i in range(self.n_states):
                active_channels = np.unique(
                    np.random.randint(0, self.n_channels, size=n_active_channels)
                )
                rates[i, active_channels] += 1

        else:
            raise NotImplementedError("Please use rates='random'.")

        return rates + eps

    def simulate_data(self, state_time_course: np.ndarray) -> np.ndarray:
        n_samples = state_time_course.shape[0]
        data = np.empty([n_samples, self.n_channels])

        # Generate data
        for i in range(n_samples):
            state = np.argmax(state_time_course[i])
            data[i] = np.random.poisson(self.rates[state])

        return data


class TDECovs:
    """Time-delay embedded covariance observation model.

    Generates time series data from TDE covariance matrices using conditional
    multivariate normal sampling. Each mode is defined by a ``CE x CE``
    covariance matrix (where ``C`` is the number of channels and ``E`` is the
    number of embeddings). At each time point, the current sample is drawn
    conditioned on the previous ``E-1`` samples.

    Parameters
    ----------
    true_tde_covs : list of np.ndarray
        List of ``n_modes`` TDE covariance matrices, each of shape
        ``(n_channels * n_embeddings, n_channels * n_embeddings)``.
        The row/column ordering is assumed to be blocks of ``E x E``
        matrices (i.e. channel-major ordering).
    n_embeddings : int, optional
        Number of time-delay embeddings.
    rho : float, optional
        Regularisation parameter for inverting the covariance.
    """

    def __init__(
        self,
        true_tde_covs: list,
        n_embeddings: int = 1,
        rho: float = 0.1,
    ):
        self.true_tde_covs = [np.asarray(c) for c in true_tde_covs]
        self.n_embeddings = n_embeddings
        self.rho = rho
        self.n_channels = self.true_tde_covs[0].shape[0] // n_embeddings
        self.n_modes = len(self.true_tde_covs)

    def _gen_data_from_tde_cov(
        self,
        tde_cov: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Generate a time series from a single TDE covariance matrix.

        Uses conditional multivariate normal sampling: at each time step,
        the current sample ``x_t`` is drawn from
        ``N(mu_cond, Sigma_cond)`` conditioned on the previous
        ``n_embeddings - 1`` samples.

        Parameters
        ----------
        tde_cov : np.ndarray
            TDE covariance matrix. Shape: (CE, CE).
        n_samples : int
            Number of time points to generate.

        Returns
        -------
        data : np.ndarray
            Generated time series. Shape: (n_samples, n_channels).
        """
        n_embeddings = self.n_embeddings
        n_channels = self.n_channels
        rho = self.rho

        # Reorder from channel-major (blocks of ExE) to embedding-major
        tde_cov = tde_cov.reshape(n_channels, n_embeddings, n_channels, n_embeddings)
        tde_cov = np.transpose(tde_cov, [1, 0, 3, 2])
        tde_cov = tde_cov.reshape(n_embeddings * n_channels, n_embeddings * n_channels)

        # Partition covariance for conditional distribution
        # See "Conditional Distributions":
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        Sig22 = tde_cov[:-n_channels, :-n_channels]
        Sig11 = tde_cov[-n_channels:, -n_channels:]
        Sig12 = tde_cov[-n_channels:, :-n_channels]

        # Initial condition
        x_2 = np.random.multivariate_normal(np.zeros(tde_cov.shape[0]), tde_cov, size=1)
        x_2 = x_2[:, :-n_channels].T  # (C*(E-1), 1)

        # Precompute projection and conditional covariance
        invSig22 = np.linalg.pinv(Sig22 + np.eye(Sig22.shape[0]) * rho)
        Sig_cond = (Sig11 - Sig12 @ invSig22 @ Sig12.T) + np.eye(n_channels) * 0.001
        proj = Sig12 @ invSig22

        # Generate data autoregressively
        data = np.zeros((n_samples, n_channels))
        for t in range(n_samples):
            mu = proj @ x_2
            x_1 = np.random.multivariate_normal(mu.flatten(), Sig_cond)
            data[t] = x_1
            x_2 = np.concatenate([x_2[n_channels:], x_1[:, np.newaxis]], axis=0)

        # Standardise
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        return data

    def simulate_data(
        self,
        state_time_course: np.ndarray,
    ) -> np.ndarray:
        """Simulate time series data.

        For each mode, generates a full-length time series from the mode's
        TDE covariance, then masks it by the state time course.

        Parameters
        ----------
        state_time_course : np.ndarray
            One-hot encoded state time course.
            Shape: (n_samples, n_modes).

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape: (n_samples, n_channels).
        """
        n_samples = state_time_course.shape[0]
        data = np.zeros((n_samples, self.n_channels))

        for i in range(self.n_modes):
            activity = self._gen_data_from_tde_cov(self.true_tde_covs[i], n_samples)
            data += state_time_course[:, i : i + 1] * activity

        return data
