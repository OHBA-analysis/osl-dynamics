"""Classes for simulating a soft mixture of modes.

"""

import numpy as np
from scipy.special import softmax

from osl_dynamics.simulation.mvn import MVN
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
    random_seed : int
        Seed used for the random number generator, which is used to sample
        an initial phase for each sinusoid.
    """

    def __init__(
        self,
        n_modes,
        relative_activation,
        amplitudes,
        frequencies,
        sampling_frequency,
        random_seed=None,
    ):
        if len(relative_activation) != n_modes:
            raise ValueError("n_modes relative_activation must be passed.")

        if len(amplitudes) != n_modes:
            raise ValueError("n_modes amplitudes must be passed.")

        if len(frequencies) != n_modes:
            raise ValueError("n_modes frequencies must be passed.")

        self.n_modes = n_modes
        self.relative_activation = relative_activation
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.sampling_frequency = sampling_frequency
        self._rng = np.random.default_rng(random_seed)

    def generate_modes(self, n_samples):

        # Simulate a random initial phase for each sinusoid
        self.phases = self._rng.uniform(0, 2 * np.pi, self.n_modes)

        # Generator mode time courses
        self.logits = np.empty([n_samples, self.n_modes], dtype=np.float32)
        t = np.arange(
            0, n_samples / self.sampling_frequency, 1.0 / self.sampling_frequency
        )
        for i in range(self.n_modes):
            self.logits[:, i] = self.relative_activation[i] + self.amplitudes[
                i
            ] * np.sin(2 * np.pi * self.frequencies[i] * t + self.phases[i])

        # Ensure mode time courses sum to one at each time point
        modes = softmax(self.logits, axis=1)

        return modes


class MixedSine_MVN(Simulation):
    """Simulates sinusoidal alphas with a multivariable normal observation model.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
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
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for random number generator.
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
        n_modes=None,
        n_channels=None,
        observation_error=0.0,
        random_seed=None,
    ):
        # Observation model
        self.obs_mod = MVN(
            means=means,
            covariances=covariances,
            n_modes=n_modes,
            n_channels=n_channels,
            observation_error=observation_error,
            random_seed=random_seed,
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
            random_seed=random_seed,
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
