"""Classes for simulating a sinusoidal time series.

"""

import numpy as np


class SingleSine:
    """Class that generates sinusoidal time series data.

    The time series for each channel is a single single wave. The amplitude
    and frequency of the sine wave can be different for different states and
    channels. A random phase is used each time a state is activated.

    Parameters
    ----------
    amplitudes : np.ndarray
        Amplitude for the sine wave for each state and channel.
        Shape must be (n_states, n_channels).
    frequenices : np.ndarray
        Frequency for the sine wave for each state and channel.
        Shape must be (n_states, n_channels).
    sampling_frequency : float
        Sampling frequency in Hertz.
    covariances : np.ndarray
        Covariances for each state. Shape must be (n_states, n_channels)
        or (n_states, n_channels, n_channels).
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for the random number generator.
    """

    def __init__(
        self,
        amplitudes,
        frequencies,
        sampling_frequency,
        covariances=None,
        observation_error=0.0,
        random_seed=None,
    ):
        self._rng = np.random.default_rng(random_seed)

        if amplitudes.shape != frequencies.shape:
            raise ValueError(
                "Mismatch between the number of amplitudes and frequencies."
            )

        self.n_states = amplitudes.shape[0]
        self.n_channels = amplitudes.shape[1]

        if covariances is None:
            covariances = np.zeros([self.n_states, self.n_channels, self.n_channels])

        elif isinstance(covariances, str):
            covariances = self.create_covariances(covariances)

        if amplitudes.shape[0] != covariances.shape[0]:
            raise ValueError(
                "Mismatch between number of amplitude states and covariance states."
            )

        if covariances.ndim == 2:
            covariances = np.array([np.diag(c) for c in covariances])

        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.sampling_frequency = sampling_frequency
        self.covariances = covariances
        self.observation_error = observation_error

    def create_covariances(self, option, eps=1e-6):
        if option == "random":
            # Randomly sample the elements of W from a normal distribution
            W = self._rng.normal(
                0, 0.1, size=[self.n_states, self.n_channels, self.n_channels]
            )

            # A small value to add to the diagonal to ensure the covariances are
            # invertible
            eps = np.tile(np.eye(self.n_channels), [self.n_states, 1, 1]) * eps

            # Calculate the covariances
            covariances = W @ W.transpose([0, 2, 1]) + eps

            # Add a large activation to a small number of the channels at random
            n_active_channels = max(1, self.n_channels // self.n_states)
            for i in range(self.n_states):
                active_channels = np.unique(
                    self._rng.integers(0, self.n_channels, size=n_active_channels)
                )
                covariances[i, active_channels, active_channels] += 0.25
        else:
            raise NotImplementedError("Please use covariances='random'.")

        return covariances

    def simulate_data(self, state_time_course):
        n_samples = state_time_course.shape[0]
        data = np.empty([n_samples, self.n_channels])

        # Generate data
        for i in range(n_samples):
            if i == 0 or (
                np.argmax(state_time_course[i]) != np.argmax(state_time_course[i - 1])
            ):
                phases = self._rng.uniform(
                    0, 2 * np.pi, size=(self.n_states, self.n_channels)
                )
            state = np.argmax(state_time_course[i])
            for j in range(self.n_channels):
                data[i, j] = self.amplitudes[state, j] * np.sin(
                    2 * np.pi * self.frequencies[state, j] * i / self.sampling_frequency
                    + phases[state, j]
                )

        # Add noise from the covariances
        for i in range(self.n_states):
            time_points_active = state_time_course[:, i] == 1
            n_time_points_active = np.count_nonzero(time_points_active)
            data[time_points_active] += self._rng.multivariate_normal(
                np.zeros(self.n_channels),
                self.covariances[i],
                size=n_time_points_active,
            )

        # Add an error to the data at all time points
        data += self._rng.normal(scale=self.observation_error, size=data.shape)

        return data.astype(np.float32)
