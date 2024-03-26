"""Classes for simulating a soft mixture of modes.

"""

import numpy as np
from scipy.special import softmax

from osl_dynamics.simulation.mvn import MVN, MSess_MVN
from osl_dynamics.simulation.base import Simulation


class MixedSine:
    """Simulates sinusoidal oscilations in mode time courses.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    relative_activation : np.ndarray or list
        Average value for each sine wave. Note, this might not be the
        mean value for each mode time course because there is a softmax
        operation. This argument can use use to change the relative values
        of each mode time course.
    amplitudes : np.ndarray or list
        Amplitude of each sinusoid.
    frequencies : np.ndarray or list
        Frequency of each sinusoid.
    sampling_frequency : float
        Sampling frequency.
    """

    def __init__(
        self,
        n_modes,
        relative_activation,
        amplitudes,
        frequencies,
        sampling_frequency,
    ):
        if len(relative_activation) != n_modes:
            raise ValueError("len(relative_activation) does not match len(n_modes).")

        if len(amplitudes) != n_modes:
            raise ValueError("n_modes amplitudes must be passed.")

        if len(frequencies) != n_modes:
            raise ValueError("n_modes frequencies must be passed.")

        self.n_modes = n_modes
        self.relative_activation = relative_activation
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.sampling_frequency = sampling_frequency

    def generate_modes(self, n_samples):
        # Simulate a random initial phase for each sinusoid
        self.phases = np.random.uniform(0, 2 * np.pi, self.n_modes)

        # Generator mode time courses
        self.logits = np.empty([n_samples, self.n_modes], dtype=np.float32)
        t = np.arange(
            0,
            n_samples / self.sampling_frequency,
            1.0 / self.sampling_frequency,
        )
        for i in range(self.n_modes):
            self.logits[:, i] = self.relative_activation[i] + self.amplitudes[
                i
            ] * np.sin(2 * np.pi * self.frequencies[i] * t + self.phases[i])

        # Ensure mode time courses sum to one at each time point
        modes = softmax(self.logits, axis=1)

        return modes


class MixedSine_MVN(Simulation):
    """Simulates sinusoidal alphas with a multivariable normal observation
    model.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    relative_activation : np.ndarray or list
        Average value for each sine wave. Note, this might not be the
        mean value for each mode time course because there is a softmax
        operation. This argument can use use to change the relative values
        of each mode time course. Shape must be (n_modes,).
    amplitudes : np.ndarray or list
        Amplitude of each sinusoid. Shape must be (n_modes,).
    frequencies : np.ndarray or list
        Frequency of each sinusoid. Shape must be (n_modes,).
    sampling_frequency : float
        Sampling frequency.
    means : np.ndarray or str
        Mean vector for each mode, shape should be (n_modes, n_channels).
        Either a numpy array or :code:`'zero'` or :code:`'random'`.
    covariances : np.ndarray or str
        Covariance matrix for each mode, shape should be (n_modes, n_channels,
        n_channels). Either a numpy array or :code:`'random'`.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    n_modes : int, optional
        Number of modes.
    n_channels : int, optional
        Number of channels.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
    """

    def __init__(
        self,
        n_samples,
        relative_activation,
        amplitudes,
        frequencies,
        sampling_frequency,
        means,
        covariances,
        n_covariances_act=1,
        n_modes=None,
        n_channels=None,
        observation_error=0.0,
    ):
        # Observation model
        self.obs_mod = MVN(
            means=means,
            covariances=covariances,
            n_covariances_act=n_covariances_act,
            n_modes=n_modes,
            n_channels=n_channels,
            observation_error=observation_error,
        )

        self.n_modes = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        # Soft mixed mode time courses class
        self.sm = MixedSine(
            n_modes=self.n_modes,
            relative_activation=relative_activation,
            amplitudes=amplitudes,
            frequencies=frequencies,
            sampling_frequency=sampling_frequency,
        )

        super().__init__(n_samples=n_samples)

        # Simulate data
        self.mode_time_course = self.generate_modes(self.n_samples)
        self.time_series = self.obs_mod.simulate_data(self.mode_time_course)

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        elif attr in dir(self.sm):
            return getattr(self.sm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def standardize(self):
        mu = np.mean(self.time_series, axis=0).astype(np.float64)
        sigma = np.std(self.time_series, axis=0).astype(np.float64)
        super().standardize()
        self.obs_mod.means = (self.obs_mod.means - mu[np.newaxis, ...]) / sigma[
            np.newaxis, ...
        ]
        self.obs_mod.covariances /= np.outer(sigma, sigma)[np.newaxis, ...]


class MSess_MixedSine_MVN(Simulation):
    """Simulates sinusoidal alphas with a multivariable normal observation model
    for each session.

    Parameters
    ----------
    n_samples : int
        Number of samples per session to draw from the model.
    relative_activation : np.ndarray or list
        Average value for each sine wave. Note, this might not be the
        mean value for each mode time course because there is a softmax
        operation. This argument can use use to change the relative values
        of each mode time course. Shape must be (n_modes,).
    amplitudes : np.ndarray or list
        Amplitude of each sinusoid. Shape must be (n_modes,).
    frequencies : np.ndarray or list
        Frequency of each sinusoid. Shape must be (n_modes,).
    sampling_frequency : float
        Sampling frequency.
    session_means : np.ndarray or str
        Session mean vector for each mode, shape should be (n_sessions, n_modes,
        n_channels). Either a numpy array or :code:`'zero'` or
        :code:`'random'`.
    session_covariances : np.ndarray or str
        Session covariance matrix for each mode, shape should be
        (n_sessions, n_modes, n_channels, n_channels). Either a numpy array
        or :code:`'random'`.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    n_modes : int, optional
        Number of modes.
    n_channels : int, optional
        Number of channels.
    n_sessions : int, optional
        Number of sessions.
    embeddings_dim : int, optional
        Number of dimensions for embedding vectors.
    spatial_embeddings_dim : int, optional
        Number of dimensions for spatial embedding vectors.
    embeddings_scale : float, optional
        Scale of variability between session observation parameters.
    n_groups : int, optional
        Number of groups when session means or covariances are
        :code:`'random'`.
    between_group_scale : float, optional
        Scale of variability between groups.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
    """

    def __init__(
        self,
        n_samples,
        relative_activation,
        amplitudes,
        frequencies,
        sampling_frequency,
        session_means,
        session_covariances,
        n_covariances_act=1,
        n_modes=None,
        n_channels=None,
        n_sessions=None,
        embeddings_dim=None,
        spatial_embeddings_dim=None,
        embeddings_scale=None,
        n_groups=None,
        between_group_scale=None,
        observation_error=0.0,
    ):
        # Observation model
        self.obs_mod = MSess_MVN(
            session_means=session_means,
            session_covariances=session_covariances,
            n_covariances_act=n_covariances_act,
            n_modes=n_modes,
            n_channels=n_channels,
            n_sessions=n_sessions,
            embeddings_dim=embeddings_dim,
            spatial_embeddings_dim=spatial_embeddings_dim,
            embeddings_scale=embeddings_scale,
            n_groups=n_groups,
            between_group_scale=between_group_scale,
            observation_error=observation_error,
        )

        self.n_modes = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels
        self.n_sessions = self.obs_mod.n_sessions

        super().__init__(n_samples=n_samples)

        # Simulate mode time courses for all sessions
        self.mode_time_course = []
        self.sm = []

        for _ in range(self.n_sessions):
            sm = MixedSine(
                n_modes=self.n_modes,
                relative_activation=relative_activation,
                amplitudes=amplitudes,
                frequencies=frequencies,
                sampling_frequency=sampling_frequency,
            )
            self.sm.append(sm)
            self.mode_time_course.append(sm.generate_modes(self.n_samples))
        self.mode_time_course = np.array(self.mode_time_course)

        # Simulate data
        self.time_series = self.obs_mod.simulate_multi_session_data(
            self.mode_time_course
        )

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def standardize(self):
        means = np.mean(self.time_series, axis=1).astype(np.float64)
        standard_deviations = np.std(self.time_series, axis=1).astype(np.float64)
        super().standardize(axis=1)
        self.obs_mod.session_means = (
            self.obs_mod.session_means - means[:, None, :]
        ) / standard_deviations[:, None, :]
        self.obs_mod.session_covariances /= np.expand_dims(
            standard_deviations[:, :, None] @ standard_deviations[:, None, :],
            axis=1,
        )
