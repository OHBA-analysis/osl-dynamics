"""Classes for simulating a sinusoidal time series.

"""

import numpy as np


class SingleSine:
    """Class that generates sinusoidal time series data.

    The time series for each channel is a singe single wave. The amplitude
    and frequency of the sine wave can be different for different modes and
    channels. A random phase is used each time a mode is activated.

    Parameters
    ----------
    amplitudes : np.ndarray
        Amplitude for the sine wave for each mode and channel.
        Shape must be (n_modes, n_channels).
    frequenices : np.ndarray
        Frequency for the sine wave for each mode and channel.
        Shape must be (n_modes, n_channels).
    sampling_frequency : float
        Sampling frequency in Hertz.
    covariances : np.ndarray
        Covariances for each mode. Shape must be (n_modes, n_channels)
        or (n_modes, n_channels, n_channels).
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for the random number generator.
    """

    def __init__(
        self,
        amplitudes: np.ndarray,
        frequencies: np.ndarray,
        sampling_frequency: float,
        covariances: np.ndarray = None,
        observation_error: float = 0.0,
        random_seed: int = None,
    ):
        if amplitudes.shape != frequencies.shape:
            raise ValueError(
                "Mismatch between the number of amplitudes and frequencies."
            )

        self.n_modes = amplitudes.shape[0]
        self.n_channels = amplitudes.shape[1]

        if covariances is None:
            covariances = np.zeros([self.n_modes, self.n_channels, self.n_channels])

        if amplitudes.shape[0] != covariances.shape[0]:
            raise ValueError(
                "Mismatch between number of amplitude modes and covariance modes."
            )

        if covariances.ndim == 2:
            covariances = np.array([np.diag(c) for c in covariances])

        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.sampling_frequency = sampling_frequency
        self.covariances = covariances
        self.observation_error = observation_error
        self._rng = np.random.default_rng(random_seed)

    def simulate_data(self, mode_time_course):
        # NOTE: We assume mutually exclusive modes when generating the data

        n_samples = mode_time_course.shape[0]
        data = np.empty([n_samples, self.n_channels])

        # Generate data
        for i in range(n_samples):
            if i == 0 or (
                np.argmax(mode_time_course[i]) != np.argmax(mode_time_course[i - 1])
            ):
                phases = self._rng.uniform(
                    0, 2 * np.pi, size=(self.n_modes, self.n_channels)
                )
            mode = np.argmax(mode_time_course[i])
            for j in range(self.n_channels):
                data[i, j] = self.amplitudes[mode, j] * np.sin(
                    2 * np.pi * self.frequencies[mode, j] * i / self.sampling_frequency
                    + phases[mode, j]
                )

        # Add noise from the covariances
        for i in range(self.n_modes):
            time_points_active = mode_time_course[:, i] == 1
            n_time_points_active = np.count_nonzero(time_points_active)
            data[time_points_active] += self._rng.multivariate_normal(
                np.zeros(self.n_channels),
                self.covariances[i],
                size=n_time_points_active,
            )

        # Add an error to the data at all time points
        data += self._rng.normal(scale=self.observation_error, size=data.shape)

        return data.astype(np.float32)
