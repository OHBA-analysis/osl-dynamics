"""Soft mixtures of modes."""

from typing import List, Optional, Union

import numpy as np
from scipy.special import softmax

from osl_dynamics.simulation.base import Simulation
from osl_dynamics.simulation.obs_mod import MVN, MSess_MVN


class MixedSine:
    """Mode time courses from softmax-normalized sinusoids.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    relative_activation : np.ndarray or list
        Average value for each sine wave. Note, this might not be the
        mean value for each mode time course because there is a softmax
        operation. This argument can be used to change the relative values
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
        n_modes: int,
        relative_activation: Union[np.ndarray, List[float]],
        amplitudes: Union[np.ndarray, List[float]],
        frequencies: Union[np.ndarray, List[float]],
        sampling_frequency: float,
    ) -> None:
        if len(relative_activation) != n_modes:
            raise ValueError("len(relative_activation) does not match n_modes.")

        if len(amplitudes) != n_modes:
            raise ValueError("n_modes amplitudes must be passed.")

        if len(frequencies) != n_modes:
            raise ValueError("n_modes frequencies must be passed.")

        self.n_modes = n_modes
        self.relative_activation = relative_activation
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.sampling_frequency = sampling_frequency

    def generate_modes(self, n_samples: int) -> np.ndarray:
        """Generate a mode time course.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        mode_time_course : np.ndarray
            Mixing coefficients. Shape is (n_samples, n_modes).
        """
        # Simulate a random initial phase for each sinusoid
        self.phases = np.random.uniform(0, 2 * np.pi, self.n_modes)

        # Logits of the mode time courses
        self.logits = np.empty([n_samples, self.n_modes], dtype=np.float32)
        t = np.arange(n_samples) / self.sampling_frequency
        for i in range(self.n_modes):
            self.logits[:, i] = self.relative_activation[i] + self.amplitudes[
                i
            ] * np.sin(2 * np.pi * self.frequencies[i] * t + self.phases[i])

        # Ensure mode time courses sum to one at each time point
        modes = softmax(self.logits, axis=1)

        return modes


class MixedSine_MVN(Simulation):
    """Sinusoidal mode mixing with a multivariate normal observation model.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    relative_activation : np.ndarray or list
        Average value for each sine wave. Note, this might not be the
        mean value for each mode time course because there is a softmax
        operation. This argument can be used to change the relative values
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
        n_samples: int,
        relative_activation: Union[np.ndarray, List[float]],
        amplitudes: Union[np.ndarray, List[float]],
        frequencies: Union[np.ndarray, List[float]],
        sampling_frequency: float,
        means: Union[np.ndarray, str],
        covariances: Union[np.ndarray, str],
        n_covariances_act: int = 1,
        n_modes: Optional[int] = None,
        n_channels: Optional[int] = None,
        observation_error: float = 0.0,
    ) -> None:
        obs_model = MVN(
            means=means,
            covariances=covariances,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
            observation_error=observation_error,
        )
        obs_model.initialize(n_modes)
        self.sm = MixedSine(
            n_modes=obs_model._n,
            relative_activation=relative_activation,
            amplitudes=amplitudes,
            frequencies=frequencies,
            sampling_frequency=sampling_frequency,
        )
        super().__init__(self.sm, obs_model, n_samples)


class MSess_MixedSine_MVN(Simulation):
    """Sinusoidal mode mixing with session-specific multivariate normals.

    Parameters
    ----------
    n_samples : int
        Number of samples per session to draw from the model.
    relative_activation : np.ndarray or list
        Average value for each sine wave. Note, this might not be the
        mean value for each mode time course because there is a softmax
        operation. This argument can be used to change the relative values
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

    _time_courses_first = True

    def __init__(
        self,
        n_samples: int,
        relative_activation: Union[np.ndarray, List[float]],
        amplitudes: Union[np.ndarray, List[float]],
        frequencies: Union[np.ndarray, List[float]],
        sampling_frequency: float,
        session_means: Union[np.ndarray, str],
        session_covariances: Union[np.ndarray, str],
        n_covariances_act: int = 1,
        n_modes: Optional[int] = None,
        n_channels: Optional[int] = None,
        n_sessions: Optional[int] = None,
        embeddings_dim: Optional[int] = None,
        spatial_embeddings_dim: Optional[int] = None,
        embeddings_scale: Optional[float] = None,
        n_groups: Optional[int] = None,
        between_group_scale: Optional[float] = None,
        observation_error: float = 0.0,
    ) -> None:
        obs_model = MSess_MVN(
            session_means=session_means,
            session_covariances=session_covariances,
            n_covariances_act=n_covariances_act,
            n_channels=n_channels,
            n_sessions=n_sessions,
            embeddings_dim=embeddings_dim,
            spatial_embeddings_dim=spatial_embeddings_dim,
            embeddings_scale=embeddings_scale,
            n_groups=n_groups,
            between_group_scale=between_group_scale,
            observation_error=observation_error,
        )
        obs_model.initialize(n_modes)
        self.sm = [
            MixedSine(
                n_modes=obs_model._n,
                relative_activation=relative_activation,
                amplitudes=amplitudes,
                frequencies=frequencies,
                sampling_frequency=sampling_frequency,
            )
            for _ in range(obs_model.n_sessions)
        ]
        super().__init__(self.sm, obs_model, n_samples)
