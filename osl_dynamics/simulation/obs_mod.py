"""Observation models for simulations.

An observation model generates data given a state or mode time course. The
number of states or modes is taken from the parameter arrays, or from the
state model when the parameters are generated randomly (see
:class:`~osl_dynamics.simulation.Simulation`).
"""

from typing import List, Optional, Union

import numpy as np
from scipy.signal import lfilter, lfiltic

from osl_dynamics.utils import array_ops


class ObservationModel:
    """Base class for observation models.

    Subclasses set :code:`self._n` (the number of states or modes) when it
    can be inferred from the parameters, and implement
    :code:`_create_parameters` (generate any random parameters) and
    :code:`simulate_data`.
    """

    _n = None

    def initialize(self, n: Optional[int] = None) -> None:
        """Set the number of states or modes and generate random parameters.

        Parameters
        ----------
        n : int, optional
            Number of states or modes. Must match the parameter arrays if
            they were passed.
        """
        if self._n is None:
            if n is None:
                raise ValueError(
                    "The number of states or modes must be passed to generate "
                    "random parameters."
                )
            self._n = n
            self._create_parameters()
        elif n is not None and n != self._n:
            raise ValueError(
                f"The observation model has parameters for {self._n} states or "
                f"modes, but {n} were requested."
            )

    def _create_parameters(self) -> None:
        pass

    def _check_states(self, time_course: np.ndarray) -> None:
        """Raise an error if the time course is not mutually exclusive states."""
        if not (
            np.isin(time_course, (0, 1)).all() and (time_course.sum(axis=-1) == 1).all()
        ):
            raise ValueError(
                f"{type(self).__name__} needs a one-hot state time course "
                "(mutually exclusive states)."
            )

    def simulate_data(self, time_course: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def rescale(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        """Update the parameters after the data are standardized.

        Parameters
        ----------
        mu : np.ndarray
            Mean subtracted from each channel.
        sigma : np.ndarray
            Standard deviation each channel was divided by.
        """


class MVN(ObservationModel):
    """Multivariate normal observation model.

    At each time point the data are drawn from
    :math:`\\mathcal{N}(\\sum_k \\alpha_{kt} m_k, \\sum_k \\alpha_{kt} C_k)`,
    where :math:`\\alpha_t` is the state or mode time course.

    Parameters
    ----------
    means : np.ndarray or str
        Mean vector for each state or mode, shape (n, n_channels). Either a
        numpy array or :code:`'zero'` or :code:`'random'`.
    covariances : np.ndarray or str
        Covariance matrix for each state or mode, shape
        (n, n_channels, n_channels). Either a numpy array or
        :code:`'random'`.
    n_channels : int, optional
        Number of channels. Needed if neither :code:`means` nor
        :code:`covariances` is an array.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
    """

    def __init__(
        self,
        means: Union[np.ndarray, str],
        covariances: Union[np.ndarray, str],
        n_channels: Optional[int] = None,
        n_covariances_act: int = 1,
        observation_error: float = 0.0,
    ) -> None:
        self.n_covariances_act = n_covariances_act
        self.observation_error = observation_error
        self.means = means
        self.covariances = covariances

        means_is_array = isinstance(means, np.ndarray)
        covs_is_array = isinstance(covariances, np.ndarray)
        if means_is_array and covs_is_array:
            if means.shape[0] != covariances.shape[0]:
                raise ValueError(
                    "means and covariances have a different number of states "
                    "or modes."
                )
            if means.shape[1] != covariances.shape[1]:
                raise ValueError(
                    "means and covariances have a different number of channels."
                )
        if means_is_array:
            self._n, self.n_channels = means.shape[:2]
        elif covs_is_array:
            self._n, self.n_channels = covariances.shape[:2]
        elif n_channels is None:
            raise ValueError(
                "If we are generating means and covariances, n_channels must be "
                "passed."
            )
        else:
            self.n_channels = n_channels

        if self._n is not None:
            self._create_parameters()

    def _create_parameters(self) -> None:
        if not isinstance(self.means, np.ndarray):
            self.means = self.create_means(self.means)
        if not isinstance(self.covariances, np.ndarray):
            self.covariances = self.create_covariances(self.covariances)

    def create_means(
        self, option: str, mu: float = 0, sigma: float = 0.2
    ) -> np.ndarray:
        if option == "zero":
            means = np.zeros([self._n, self.n_channels])
        elif option == "random":
            means = np.random.normal(mu, sigma, size=[self._n, self.n_channels])
        else:
            raise ValueError("means must be a np.array or 'zero' or 'random'.")
        return means

    def create_covariances(
        self, option: str, activation_strength: float = 1, eps: float = 1e-6
    ) -> np.ndarray:
        if option == "random":
            # Randomly sample the elements of W from a normal distribution
            W = np.random.normal(
                0, 0.1, size=[self._n, self.n_channels, self.n_channels]
            )

            # Add a large activation to a small number of the channels at random
            activation_strength_multipliers = np.linspace(1, 5, self.n_covariances_act)
            for j in range(self.n_covariances_act):
                n_active_channels = max(1, 2 * self.n_channels // self._n)
                for i in range(self._n):
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
            eps = np.tile(np.eye(self.n_channels), [self._n, 1, 1]) * eps

            # Calculate the covariances
            covariances = W @ W.transpose([0, 2, 1]) + eps

        else:
            raise ValueError("covariances must be a np.ndarray or 'random'.")

        return covariances

    def simulate_data(self, time_course: np.ndarray) -> np.ndarray:
        """Simulate time series data.

        Parameters
        ----------
        time_course : np.ndarray
            State or mode time course. Shape must be (n_samples, n).

        Returns
        -------
        data : np.ndarray
            Time series data. Shape is (n_samples, n_channels).
        """
        n_samples = time_course.shape[0]

        # Initialize array to hold data
        data = np.zeros((n_samples, self.n_channels))

        # Loop through all unique combinations of states or modes
        for alpha in np.unique(time_course, axis=0):
            # Mean and covariance for this combination
            mu = np.sum(self.means * alpha[:, np.newaxis], axis=0)
            sigma = np.sum(
                self.covariances * alpha[:, np.newaxis, np.newaxis],
                axis=0,
            )

            # Generate data for the time points that this combination is
            # active
            data[np.all(time_course == alpha, axis=1)] = np.random.multivariate_normal(
                mu,
                sigma,
                size=np.count_nonzero(np.all(time_course == alpha, axis=1)),
            )

        # Add an error to the data at all time points
        data += np.random.normal(scale=self.observation_error, size=data.shape)

        return data.astype(np.float32)

    def get_instantaneous_covariances(self, time_course: np.ndarray) -> np.ndarray:
        """Get the ground truth covariance at each time point.

        Parameters
        ----------
        time_course : np.ndarray
            State or mode time course. Shape must be (n_samples, n).

        Returns
        -------
        inst_covs : np.ndarray
            Instantaneous covariances.
            Shape is (n_samples, n_channels, n_channels).
        """
        # Initialize an array to hold the data
        n_samples = time_course.shape[0]
        inst_covs = np.empty([n_samples, self.n_channels, self.n_channels])

        # Loop through all unique combinations of states or modes
        for alpha in np.unique(time_course, axis=0):
            sigma = np.sum(
                self.covariances * alpha[:, np.newaxis, np.newaxis],
                axis=0,
            )
            inst_covs[np.all(time_course == alpha, axis=1)] = sigma

        return inst_covs.astype(np.float32)

    def rescale(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        self.means = (self.means - mu[np.newaxis, ...]) / sigma[np.newaxis, ...]
        self.covariances = self.covariances / np.outer(sigma, sigma)[np.newaxis, ...]
        self.observation_error = self.observation_error / sigma


class OscillatoryMVN(MVN):
    """Multivariate normal observation model with oscillatory states or modes.

    Each state or mode :math:`k` has a spectral peak at its own frequency.
    The data are

    .. math::
        x_t = \\sum_k \\alpha_{kt} m_k
        + \\sum_k \\sqrt{\\alpha_{kt}} L_k y^{(k)}_t + \\epsilon_t

    where :math:`L_k` is the Cholesky factor of the covariance :math:`C_k`,
    :math:`\\epsilon_t` is white noise and :math:`y^{(k)}_t` contains
    independent, unit-variance AR(2) processes (one per channel). The AR(2)
    process for frequency :math:`f` and peak width :math:`w` (in Hz) has poles
    at :math:`r e^{\\pm i \\omega}`, with :math:`r = e^{-\\pi w / f_s}` and
    :math:`\\omega = 2 \\pi f / f_s`. The spectral peak is close to :math:`f`
    when :math:`w` is small compared to :math:`f`. The processes run through
    the whole recording and are started from their stationary distribution.

    With mutually exclusive states, the covariance of state :math:`k` is
    :math:`C_k` and its spectrum peaks at its frequency. With modes, the
    zero-lag covariance at time :math:`t` is :math:`\\sum_k \\alpha_{kt} C_k`,
    as in :class:`MVN`.

    Parameters
    ----------
    frequencies : np.ndarray or list
        Peak frequency (Hz) of each state or mode, shape (n,), or of each
        state or mode and channel, shape (n, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
    covariances : np.ndarray or str
        Covariance matrix for each state or mode, shape
        (n, n_channels, n_channels). Either a numpy array or
        :code:`'random'`.
    means : np.ndarray or str, optional
        Mean vector for each state or mode, shape (n, n_channels). Either a
        numpy array or :code:`'zero'` or :code:`'random'`.
    peak_width : float or np.ndarray, optional
        Width (Hz) of the spectral peaks. A float, or one value per state or
        mode, shape (n,).
    n_channels : int, optional
        Number of channels. Needed if it can't be inferred from the other
        arguments.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    observation_error : float, optional
        Standard deviation of the white noise added to the generated data.
    """

    def __init__(
        self,
        frequencies: Union[np.ndarray, List[float]],
        sampling_frequency: float,
        covariances: Union[np.ndarray, str],
        means: Union[np.ndarray, str] = "zero",
        peak_width: Union[float, np.ndarray] = 2.0,
        n_channels: Optional[int] = None,
        n_covariances_act: int = 1,
        observation_error: float = 0.0,
    ) -> None:
        frequencies = np.asarray(frequencies, dtype=float)
        if frequencies.ndim not in (1, 2):
            raise ValueError("frequencies must have shape (n,) or (n, n_channels).")
        if np.any(frequencies <= 0) or np.any(frequencies >= sampling_frequency / 2):
            raise ValueError(
                "frequencies must be between 0 and the Nyquist frequency "
                f"({sampling_frequency / 2} Hz)."
            )
        peak_width = np.asarray(peak_width, dtype=float)
        if peak_width.ndim > 1 or (
            peak_width.ndim == 1 and len(peak_width) != len(frequencies)
        ):
            raise ValueError("peak_width must be a float or have shape (n,).")
        if np.any(peak_width <= 0):
            raise ValueError("peak_width must be positive.")

        self.frequencies = frequencies
        self.peak_width = peak_width
        self.sampling_frequency = sampling_frequency

        if n_channels is None and frequencies.ndim == 2:
            n_channels = frequencies.shape[1]
        super().__init__(
            means=means,
            covariances=covariances,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
            observation_error=observation_error,
        )
        if frequencies.ndim == 2 and frequencies.shape[1] != self.n_channels:
            raise ValueError("frequencies has a different number of channels.")
        self.initialize(len(frequencies))

    def _ar2_coefficients(self) -> tuple:
        """AR(2) coefficients for each state or mode and channel.

        Returns
        -------
        a1, a2 : np.ndarray
            Coefficients of :math:`y_{t-1}` and :math:`y_{t-2}`.
            Shape is (n, n_channels).
        """
        f = np.broadcast_to(
            self.frequencies.reshape(self._n, -1), (self._n, self.n_channels)
        )
        w = np.broadcast_to(self.peak_width.reshape(-1, 1), (self._n, 1))
        r = np.exp(-np.pi * w / self.sampling_frequency)
        omega = 2 * np.pi * f / self.sampling_frequency
        return 2 * r * np.cos(omega), np.broadcast_to(-(r**2), f.shape)

    @staticmethod
    def _ar2_autocovariance(a1: np.ndarray, a2: np.ndarray) -> tuple:
        """Lag 0 and lag 1 autocovariance of an AR(2) process with unit-variance
        innovations."""
        gamma0 = (1 - a2) / ((1 + a2) * ((1 - a2) ** 2 - a1**2))
        gamma1 = a1 * gamma0 / (1 - a2)
        return gamma0, gamma1

    def _generate_rhythms(self, k: int, n_samples: int) -> np.ndarray:
        """Unit-variance AR(2) processes for state or mode :code:`k`.

        Returns
        -------
        y : np.ndarray
            Shape is (n_samples, n_channels).
        """
        a1, a2 = self._ar2_coefficients()
        a1, a2 = a1[k], a2[k]
        gamma0, gamma1 = self._ar2_autocovariance(a1, a2)

        # Initial values (y_{-1}, y_{-2}) from the stationary distribution
        z = np.random.normal(size=(2, self.n_channels))
        y1 = np.sqrt(gamma0) * z[0]
        y2 = (
            gamma1 / np.sqrt(gamma0) * z[0]
            + np.sqrt(gamma0 - gamma1**2 / gamma0) * z[1]
        )

        e = np.random.normal(size=(n_samples, self.n_channels))
        y = np.empty((n_samples, self.n_channels))
        for c in range(self.n_channels):
            a = [1, -a1[c], -a2[c]]
            zi = lfiltic([1], a, y=[y1[c], y2[c]])
            y[:, c], _ = lfilter([1], a, e[:, c], zi=zi)
        return y / np.sqrt(gamma0)

    def simulate_data(self, time_course: np.ndarray) -> np.ndarray:
        """Simulate time series data.

        Parameters
        ----------
        time_course : np.ndarray
            State or mode time course. Shape must be (n_samples, n).

        Returns
        -------
        data : np.ndarray
            Time series data. Shape is (n_samples, n_channels).
        """
        n_samples = time_course.shape[0]
        data = time_course @ self.means
        cholesky = np.linalg.cholesky(self.covariances)
        for k in range(self._n):
            y = self._generate_rhythms(k, n_samples)
            data += np.sqrt(time_course[:, k : k + 1]) * (y @ cholesky[k].T)
        data += np.random.normal(scale=self.observation_error, size=data.shape)
        return data.astype(np.float32)

    def power_spectra(self, frequencies: np.ndarray) -> np.ndarray:
        """Ground truth cross power spectral density of each state or mode.

        The spectra are one-sided: integrating the power spectral density from
        0 Hz to the Nyquist frequency gives the variance.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies (Hz) to evaluate the spectra at.

        Returns
        -------
        cpsd : np.ndarray
            Cross power spectral densities.
            Shape is (n, n_frequencies, n_channels, n_channels).
        """
        frequencies = np.asarray(frequencies, dtype=float)
        fs = self.sampling_frequency
        a1, a2 = self._ar2_coefficients()
        gamma0, _ = self._ar2_autocovariance(a1, a2)

        # Power spectral density of each unit-variance AR(2) process
        z = np.exp(-2j * np.pi * frequencies / fs)[None, :, None]
        transfer = 1 - a1[:, None, :] * z - a2[:, None, :] * z**2
        psd = (2 / fs) / np.abs(transfer) ** 2 / gamma0[:, None, :]

        # Mix the channels: C_k(f) = L_k diag(psd_k(f)) L_k^T
        cholesky = np.linalg.cholesky(self.covariances)
        cpsd = np.einsum("kij,kfj,klj->kfil", cholesky, psd, cholesky)

        # White noise
        noise = np.broadcast_to(self.observation_error, (self.n_channels,))
        cpsd += np.diag(noise**2 * 2 / fs)
        return cpsd


class MDyn_MVN(MVN):
    """Multivariate normal observation model with multi-time-scale dynamics.

    The means and standard deviations follow one time course and the
    correlations follow another.

    Parameters
    ----------
    means : np.ndarray or str
        Mean vector for each state or mode, shape (n, n_channels). Either a
        numpy array or :code:`'zero'` or :code:`'random'`.
    covariances : np.ndarray or str
        Covariance matrix for each state or mode, shape
        (n, n_channels, n_channels). Either a numpy array or
        :code:`'random'`.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
    """

    def _create_parameters(self) -> None:
        super()._create_parameters()

        # Get the stds and corrs from self.covariance
        self.stds = array_ops.cov2std(self.covariances)
        self.corrs = array_ops.cov2corr(self.covariances)

    def simulate_data(self, time_courses: np.ndarray) -> np.ndarray:
        """Simulates data.

        Parameters
        ----------
        time_courses : np.ndarray
            Should contain two time courses: one for the mean and standard
            deviations and another for functional connectivity. Shape is
            (2, n_samples, n).

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape is (n_samples, n_channels).
        """
        # Reshape time_courses so that the multi-time-scale dimension is last
        time_courses = np.rollaxis(time_courses, 0, 3)

        # Number of samples to simulate
        n_samples = time_courses.shape[0]

        # Initialize array to hold data
        data = np.zeros([n_samples, self.n_channels])

        # Loop through all unique combinations of states or modes
        for tc in np.unique(time_courses, axis=0):
            # Extract the different time courses
            alpha = tc[:, 0]
            beta = tc[:, 1]

            # Mean, standard deviation, corr for this combination
            mu = np.sum(self.means * alpha[:, np.newaxis], axis=0)
            G = np.diag(np.sum(self.stds * alpha[:, np.newaxis], axis=0))
            F = np.sum(self.corrs * beta[:, np.newaxis, np.newaxis], axis=0)

            # Calculate covariance matrix from the standard deviation and corr
            sigma = G @ F @ G

            # Generate data for the time points that this combination is active
            active = np.all(time_courses == tc, axis=(1, 2))
            data[active] = np.random.multivariate_normal(
                mu, sigma, size=np.count_nonzero(active)
            )

        # Add an error to the data at all time points
        data += np.random.normal(scale=self.observation_error, size=data.shape)

        return data.astype(np.float32)

    def get_instantaneous_covariances(self, time_courses: np.ndarray) -> np.ndarray:
        """Get the ground truth covariance at each time point.

        Parameters
        ----------
        time_courses : np.ndarray
            Should contain two time courses: one for the mean and standard
            deviations and another for functional connectivity. Shape is
            (2, n_samples, n).

        Returns
        -------
        inst_covs : np.ndarray
            Instantaneous covariances.
            Shape is (n_samples, n_channels, n_channels).
        """
        # Reshape time_courses so that the multi-time-scale dimension is last
        time_courses = np.rollaxis(time_courses, 0, 3)

        # Number of samples to simulate
        n_samples = time_courses.shape[0]

        # Initialize an array to hold data
        inst_covs = np.empty([n_samples, self.n_channels, self.n_channels])

        # Loop through all unique combinations of states or modes
        for tc in np.unique(time_courses, axis=0):
            # Extract the different time courses
            alpha = tc[:, 0]
            beta = tc[:, 1]

            # Standard deviation and corr for this combination
            G = np.diag(np.sum(self.stds * alpha[:, np.newaxis], axis=0))
            F = np.sum(self.corrs * beta[:, np.newaxis, np.newaxis], axis=0)

            # Calculate covariance matrix from the standard deviation and corr
            sigma = G @ F @ G

            inst_covs[np.all(time_courses == tc, axis=(1, 2))] = sigma

        return inst_covs.astype(np.float32)

    def rescale(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        super().rescale(mu, sigma)
        self.stds = self.stds / sigma[np.newaxis, ...]


class MSess_MVN(MVN):
    """Multivariate normal observation model with session-specific parameters.

    Parameters
    ----------
    session_means : np.ndarray or str
        Mean vector for each state or mode for each session, shape
        (n_sessions, n, n_channels). Either a numpy array or :code:`'zero'`
        or :code:`'random'`.
    session_covariances : np.ndarray or str
        Covariance matrix for each state or mode for each session, shape
        (n_sessions, n, n_channels, n_channels). Either a numpy array or
        :code:`'random'`.
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
        self.embedding_vectors = embedding_vectors
        self.session_means = session_means
        self.session_covariances = session_covariances
        self.n_sessions = n_sessions

        if embedding_vectors is not None:
            self.n_sessions = embedding_vectors.shape[0]
            self.embeddings_dim = embedding_vectors.shape[1]

        means_is_array = isinstance(session_means, np.ndarray)
        covs_is_array = isinstance(session_covariances, np.ndarray)

        if means_is_array and covs_is_array:
            if session_means.ndim != 3:
                raise ValueError(
                    "session_means must have shape (n_sessions, n, n_channels)."
                )
            if session_covariances.ndim != 4:
                raise ValueError(
                    "session_covariances must have shape "
                    "(n_sessions, n, n_channels, n_channels)."
                )
            if session_means.shape[0] != session_covariances.shape[0]:
                raise ValueError(
                    "session_means and session_covariances have a different "
                    "number of arrays."
                )
            if session_means.shape[1] != session_covariances.shape[1]:
                raise ValueError(
                    "session_means and session_covariances have a different "
                    "number of states or modes."
                )
            if session_means.shape[2] != session_covariances.shape[2]:
                raise ValueError(
                    "session_means and session_covariances have a different "
                    "number of channels."
                )

        if means_is_array:
            self.n_sessions, self._n, self.n_channels = session_means.shape
        elif covs_is_array:
            self.n_sessions, self._n, self.n_channels = session_covariances.shape[:3]
        elif self.n_sessions is None or n_channels is None:
            raise ValueError(
                "If we are generating means and covariances, n_sessions and "
                "n_channels must be passed."
            )
        else:
            self.n_channels = n_channels

        if self._n is not None:
            self._create_parameters()

    def _create_parameters(self) -> None:
        means_is_array = isinstance(self.session_means, np.ndarray)
        covs_is_array = isinstance(self.session_covariances, np.ndarray)

        # Both the session means and covariances were passed as numpy arrays
        if means_is_array and covs_is_array:
            self.n_groups = None
            self.group_centroids = None
            self.between_group_scale = None
            self.embeddings_dim = None
            self.spatial_embeddings_dim = None
            self.embeddings_scale = None
            self.group_means = None
            self.group_covariances = None

        # Only the session means were passed as a numpy array
        elif means_is_array:
            self.validate_embedding_parameters(self.embedding_vectors)
            self.create_embeddings(self.embedding_vectors)
            self.group_means = None
            self.group_covariances = super().create_covariances(
                self.session_covariances
            )
            self.session_covariances = self.create_session_covariances()

        # Only the session covariances were passed as a numpy array
        elif covs_is_array:
            option = self.session_means
            if not option == "zero":
                self.validate_embedding_parameters(self.embedding_vectors)
                self.create_embeddings(self.embedding_vectors)
            self.group_means = super().create_means(option)
            self.session_means = self.create_session_means(option)
            self.group_covariances = None

        # Neither session means or nor covariances were passed as numpy arrays
        else:
            self.validate_embedding_parameters(self.embedding_vectors)
            self.create_embeddings(self.embedding_vectors)
            option = self.session_means
            self.group_means = super().create_means(option)
            self.session_means = self.create_session_means(option)
            self.group_covariances = super().create_covariances(
                self.session_covariances
            )
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
        means_spatial_embeddings_linear_transform = self.create_linear_transform(
            self.n_channels, self.spatial_embeddings_dim
        )
        self.means_spatial_embeddings = (
            means_spatial_embeddings_linear_transform @ self.group_means.T
        ).T

        # Match the shapes for concatenation
        concat_array_embeddings = np.broadcast_to(
            self.embeddings[:, None, :],
            (self.n_sessions, self._n, self.embeddings_dim),
        )
        concat_means_spatial_embeddings = np.broadcast_to(
            self.means_spatial_embeddings[None, :, :],
            (self.n_sessions, self._n, self.spatial_embeddings_dim),
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
            (self.n_sessions, self._n, self.embeddings_dim),
        )
        concat_covariances_spatial_embeddings = np.broadcast_to(
            self.covariances_spatial_embeddings[None, :, :],
            (self.n_sessions, self._n, self.spatial_embeddings_dim),
        )
        self.covariances_concat_embeddings = np.concatenate(
            [concat_array_embeddings, concat_covariances_spatial_embeddings],
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
            session_means = np.zeros([self.n_sessions, self._n, self.n_channels])
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
            [self.n_sessions, self._n, self.n_channels, self.n_channels]
        )
        for i in range(self.n_sessions):
            for j in range(self._n):
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

    def simulate_data(self, time_course: np.ndarray) -> np.ndarray:
        raise TypeError(
            "MSess_MVN has session-specific parameters, use simulate_session_data "
            "or simulate_multi_session_data."
        )

    def simulate_session_data(
        self, session: int, time_course: np.ndarray
    ) -> np.ndarray:
        """Simulate single session data.

        Parameters
        ----------
        session : int
            Session number.
        time_course : np.ndarray
            State or mode time course. Shape is (n_samples, n).

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape is (n_samples, n_channels).
        """
        # Initialize array to hold data
        n_samples = time_course.shape[0]
        data = np.zeros((n_samples, self.n_channels))

        # Loop through all unique combinations of states or modes
        for alpha in np.unique(time_course, axis=0):
            # Mean and covariance for this combination
            mu = np.sum(self.session_means[session] * alpha[:, None], axis=0)
            sigma = np.sum(
                self.session_covariances[session] * alpha[:, None, None], axis=0
            )

            # Generate data for the time points that this combination is active
            active = np.all(time_course == alpha, axis=1)
            data[active] = np.random.multivariate_normal(
                mu, sigma, size=np.count_nonzero(active)
            )

        # Add an error to the data at all time points
        data += np.random.normal(scale=self.observation_error, size=data.shape)

        return data.astype(np.float32)

    def simulate_multi_session_data(self, time_courses: np.ndarray) -> np.ndarray:
        """Simulate data for all sessions.

        Parameters
        ----------
        time_courses : np.ndarray
            State or mode time course for each session.
            Shape is (n_sessions, n_samples, n).

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape is (n_sessions, n_samples, n_channels).
        """
        return np.array(
            [
                self.simulate_session_data(session, time_courses[session])
                for session in range(self.n_sessions)
            ]
        )

    def get_session_instantaneous_covariances(
        self, session: int, time_course: np.ndarray
    ) -> np.ndarray:
        """Get ground truth covariances at each time point for a session.

        Parameters
        ----------
        session : int
            Session number.
        time_course : np.ndarray
            State or mode time course. Shape is (n_samples, n).

        Returns
        -------
        inst_covs : np.ndarray
            Instantaneous covariances.
            Shape is (n_samples, n_channels, n_channels).
        """
        n_samples = time_course.shape[0]
        inst_covs = np.zeros([n_samples, self.n_channels, self.n_channels])
        for alpha in np.unique(time_course, axis=0):
            sigma = np.sum(
                self.session_covariances[session] * alpha[:, None, None], axis=0
            )
            inst_covs[np.all(time_course == alpha, axis=1)] = sigma
        return inst_covs.astype(np.float32)

    def get_instantaneous_covariances(self, time_courses: np.ndarray) -> np.ndarray:
        """Get ground truth covariance at each time point for each session.

        Parameters
        ----------
        time_courses : np.ndarray
            State or mode time course for each session.
            Shape is (n_sessions, n_samples, n).

        Returns
        -------
        inst_covs : np.ndarray
            Instantaneous covariances.
            Shape is (n_sessions, n_samples, n_channels, n_channels).
        """
        return np.array(
            [
                self.get_session_instantaneous_covariances(
                    session, time_courses[session]
                )
                for session in range(self.n_sessions)
            ]
        )

    def rescale(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        self.session_means = (self.session_means - mu[:, None, :]) / sigma[:, None, :]
        self.session_covariances = self.session_covariances / np.expand_dims(
            sigma[:, :, None] @ sigma[:, None, :], 1
        )
        self.observation_error = self.observation_error / sigma


class MAR(ObservationModel):
    r"""Multivariate autoregressive (MAR) observation model.

    A :math:`p`-order MAR model can be written as

    .. math::
        x_t = A_1 x_{t-1} + ... + A_p x_{t-p} + \epsilon_t

    where :math:`\epsilon_t \sim N(0, \Sigma)`. The MAR model is therefore
    parameterized by the MAR coefficients (:math:`A`) and covariance
    (:math:`\Sigma`). Each state has its own coefficients and covariance.

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
        self._n = coeffs.shape[0]
        self.n_channels = coeffs.shape[2]

    def simulate_data(self, time_course: np.ndarray) -> np.ndarray:
        """Simulate time series data.

        Parameters
        ----------
        time_course : np.ndarray
            State time course. Shape must be (n_samples, n_states).

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape is (n_samples, n_channels).
        """
        self._check_states(time_course)
        n_samples = time_course.shape[0]
        data = np.empty([n_samples, self.n_channels])

        # Generate the noise term first
        for i in range(self._n):
            time_points_active = time_course[:, i] == 1
            n_time_points_active = np.count_nonzero(time_points_active)
            data[time_points_active] = np.random.multivariate_normal(
                np.zeros(self.n_channels),
                self.covs[i],
                size=n_time_points_active,
            )

        # Generate the MAR process
        for t in range(n_samples):
            state = time_course[t].argmax()
            for lag in range(min(t, self.order)):
                data[t] += np.dot(self.coeffs[state, lag], data[t - lag - 1])

        return data.astype(np.float32)


class OscillatoryBursts(ObservationModel):
    """Oscillatory burst observation model.

    While a state is active, each of its channels produces a sinusoid at the
    state's frequency, with a slowly varying phase. White noise is added to
    all channels.

    Parameters
    ----------
    frequencies : np.ndarray or list
        Frequency (Hz) of each state, shape (n_states,), or of each session
        and state, shape (n_sessions, n_states). The frequency of a state with
        no active channels is ignored.
    channel_activity : np.ndarray
        Binary matrix indicating which channels are active for each state.
        Shape: (n_states, n_channels). A state with no active channels
        produces only noise.
    sampling_frequency : float, optional
        Sampling frequency in Hz.
    snr : float, optional
        Signal-to-noise ratio. The noise standard deviation is
        :code:`1 / snr`.
    """

    def __init__(
        self,
        frequencies: Union[np.ndarray, List[float]],
        channel_activity: np.ndarray,
        sampling_frequency: float = 100,
        snr: float = 4,
    ) -> None:
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.channel_activity = np.asarray(channel_activity)
        self.sampling_frequency = sampling_frequency
        self.snr = snr
        self._n, self.n_channels = self.channel_activity.shape
        if self.frequencies.shape[-1] != self._n:
            raise ValueError(
                "frequencies and channel_activity have a different number of states."
            )
        self.n_sessions = (
            self.frequencies.shape[0] if self.frequencies.ndim == 2 else None
        )

    def get_true_signal(
        self, time_course: np.ndarray, session: Optional[int] = None
    ) -> np.ndarray:
        """Noise-free signal.

        Parameters
        ----------
        time_course : np.ndarray
            State time course. Shape: (n_samples, n_states).
        session : int, optional
            Session number. Needed if there are session-specific frequencies.

        Returns
        -------
        true_signal : np.ndarray
            Shape: (n_samples, n_channels).
        """
        frequencies = self.frequencies if session is None else self.frequencies[session]
        n_samples = time_course.shape[0]
        timestamps = np.arange(n_samples) / self.sampling_frequency

        # Generate sinusoidal activity for each state and channel
        phase_diff = np.linspace(0, 0.25, self.n_channels)
        true_signal = np.zeros((n_samples, self.n_channels))
        for i in range(self._n):
            active = time_course[:, i] == 1
            for j in range(self.n_channels):
                if self.channel_activity[i, j] == 1:
                    phase = (
                        (0.5 * np.sin(2 * np.pi * 0.005 * timestamps) + phase_diff[j])
                        * 2
                        * np.pi
                    )
                    true_signal[active, j] += np.sin(
                        2 * np.pi * frequencies[i] * timestamps[active] + phase[active]
                    )
        return true_signal

    def simulate_data(self, time_course: np.ndarray) -> np.ndarray:
        """Simulate data.

        Parameters
        ----------
        time_course : np.ndarray
            State time course. Shape: (n_samples, n_states).

        Returns
        -------
        data : np.ndarray
            Simulated data with noise. Shape: (n_samples, n_channels).
        """
        if self.n_sessions is not None:
            raise TypeError(
                "There are session-specific frequencies, use simulate_session_data."
            )
        return self.simulate_session_data(None, time_course)

    def simulate_session_data(
        self, session: Optional[int], time_course: np.ndarray
    ) -> np.ndarray:
        """Simulate data for one session.

        Parameters
        ----------
        session : int
            Session number.
        time_course : np.ndarray
            State time course. Shape: (n_samples, n_states).

        Returns
        -------
        data : np.ndarray
            Simulated data with noise. Shape: (n_samples, n_channels).
        """
        self._check_states(time_course)
        true_signal = self.get_true_signal(time_course, session)
        noise_std = 1 / self.snr
        return true_signal + np.random.normal(0, noise_std, true_signal.shape)


class TDECovs(ObservationModel):
    """Time-delay embedded covariance observation model.

    Generates time series data from TDE covariance matrices using conditional
    multivariate normal sampling. Each state or mode is defined by a
    ``CE x CE`` covariance matrix (where ``C`` is the number of channels and
    ``E`` is the number of embeddings). A time series is generated for each
    state or mode, where the current sample is drawn conditioned on the
    previous ``E-1`` samples. The data are the sum of these time series
    weighted by the state or mode time course.

    Parameters
    ----------
    tde_covariances : list of np.ndarray
        TDE covariance matrix for each state or mode, each of shape
        ``(n_channels * n_embeddings, n_channels * n_embeddings)``.
        The row/column ordering is assumed to be blocks of ``E x E``
        matrices (i.e. channel-major ordering).
    n_embeddings : int, optional
        Number of time-delay embeddings.
    rho : float, optional
        Regularization parameter for inverting the covariance.
    """

    def __init__(
        self,
        tde_covariances: List[np.ndarray],
        n_embeddings: int = 1,
        rho: float = 0.1,
    ) -> None:
        self.tde_covariances = [np.asarray(c) for c in tde_covariances]
        self.n_embeddings = n_embeddings
        self.rho = rho
        self.n_channels = self.tde_covariances[0].shape[0] // n_embeddings
        self._n = len(self.tde_covariances)

    def _gen_data_from_tde_cov(
        self,
        tde_cov: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Generate a time series from a single TDE covariance matrix.

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

        # Standardize
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        return data

    def simulate_data(self, time_course: np.ndarray) -> np.ndarray:
        """Simulate time series data.

        Parameters
        ----------
        time_course : np.ndarray
            State or mode time course. Shape: (n_samples, n).

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape: (n_samples, n_channels).
        """
        n_samples = time_course.shape[0]
        data = np.zeros((n_samples, self.n_channels))
        for i in range(self._n):
            activity = self._gen_data_from_tde_cov(self.tde_covariances[i], n_samples)
            data += time_course[:, i : i + 1] * activity
        return data


class Poisson(ObservationModel):
    """Poisson observation model.

    Each channel is a Poisson count. The rate can be different for each
    state and channel.

    Parameters
    ----------
    rates : np.ndarray or str
        Rate vector for each state, shape (n_states, n_channels).
        Either a numpy array or :code:`'random'`.
    n_channels : int, optional
        Number of channels. Needed if :code:`rates='random'`.
    """

    def __init__(
        self,
        rates: Union[np.ndarray, str],
        n_channels: Optional[int] = None,
    ) -> None:
        self.rates = rates
        if isinstance(rates, np.ndarray):
            self._n, self.n_channels = rates.shape
        elif n_channels is None:
            raise ValueError("If we are generating rates, n_channels must be passed.")
        else:
            self.n_channels = n_channels

    def _create_parameters(self) -> None:
        self.rates = self.create_rates(self.rates)

    def create_rates(self, option: str, eps: float = 1e-2) -> np.ndarray:
        if option == "random":
            # Randomly sample the rates from a gamma distribution
            rates = np.random.gamma(
                shape=1.0, scale=1.1, size=(self._n, self.n_channels)
            )

            #  Add a large rate to a small number of the channels at random
            n_active_channels = max(1, self.n_channels // self._n)
            for i in range(self._n):
                active_channels = np.unique(
                    np.random.randint(0, self.n_channels, size=n_active_channels)
                )
                rates[i, active_channels] += 1
        else:
            raise NotImplementedError("Please use rates='random'.")
        return rates + eps

    def simulate_data(self, time_course: np.ndarray) -> np.ndarray:
        """Simulate data.

        Parameters
        ----------
        time_course : np.ndarray
            State time course. Shape: (n_samples, n_states).

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape: (n_samples, n_channels).
        """
        self._check_states(time_course)
        n_samples = time_course.shape[0]
        data = np.empty([n_samples, self.n_channels])
        for i in range(n_samples):
            state = np.argmax(time_course[i])
            data[i] = np.random.poisson(self.rates[state])
        return data


class Dirichlet(ObservationModel):
    """Dirichlet observation model.

    Each sample is a composition (non-negative and summing to one over
    channels) drawn from a Dirichlet distribution with a state-specific
    concentration vector.

    Parameters
    ----------
    concentration : np.ndarray or str
        Concentration vector for each state, shape (n_states, n_channels).
        Either a numpy array or :code:`'random'`.
    n_channels : int, optional
        Number of channels. Needed if :code:`concentration='random'`.
    """

    def __init__(
        self,
        concentration: Union[np.ndarray, str],
        n_channels: Optional[int] = None,
    ) -> None:
        self.concentration = concentration
        if isinstance(concentration, np.ndarray):
            self._n, self.n_channels = concentration.shape
        elif n_channels is None:
            raise ValueError(
                "If we are generating concentration, n_channels must be passed."
            )
        else:
            self.n_channels = n_channels

    def _create_parameters(self) -> None:
        self.concentration = self.create_concentration(self.concentration)

    def create_concentration(self, option: str, eps: float = 1e-2) -> np.ndarray:
        if option == "random":
            # Randomly sample the concentration from a gamma distribution
            concentration = np.random.gamma(
                shape=1.0, scale=1.1, size=(self._n, self.n_channels)
            )

            #  Add a large concentration to a small number of the channels at random
            n_active_channels = max(1, self.n_channels // self._n)
            for i in range(self._n):
                active_channels = np.unique(
                    np.random.randint(0, self.n_channels, size=n_active_channels)
                )
                concentration[i, active_channels] += 3
        else:
            raise NotImplementedError("Please use concentration='random'.")
        return concentration + eps

    def simulate_data(self, time_course: np.ndarray) -> np.ndarray:
        """Simulate data.

        Parameters
        ----------
        time_course : np.ndarray
            State time course. Shape: (n_samples, n_states).

        Returns
        -------
        data : np.ndarray
            Simulated data. Shape: (n_samples, n_channels).
        """
        self._check_states(time_course)
        n_samples = time_course.shape[0]
        data = np.empty([n_samples, self.n_channels])
        for i in range(n_samples):
            state = np.argmax(time_course[i])
            data[i] = np.random.dirichlet(self.concentration[state])
        return data
