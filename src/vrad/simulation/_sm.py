"""Classes for simulation a soft mixture of states.

"""

from typing import Union

import numpy as np
from scipy.special import softmax
from vrad.simulation import MVN, Simulation


class MixedSine:
    """Simulates sinusoidal oscilations in state time courses.

    Parameters
    ----------
    n_states : int
        Number of states.
    relative_activation : np.ndarray or list
        Average value for each sine wave. Note, this might not be the
        mean value for each state time course because there is a softmax
        operation. This argument can use use to change the relative values
        of each state time course.
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
        n_states: int,
        relative_activation: Union[np.ndarray, list],
        amplitudes: Union[np.ndarray, list],
        frequencies: Union[np.ndarray, list],
        sampling_frequency: float,
        random_seed: int = None,
    ):
        if len(relative_activation) != n_states:
            raise ValueError("n_states relative_activation must be passed.")

        if len(amplitudes) != n_states:
            raise ValueError("n_states amplitudes must be passed.")

        if len(frequencies) != n_states:
            raise ValueError("n_states frequencies must be passed.")

        self.n_states = n_states
        self.relative_activation = relative_activation
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.sampling_frequency = sampling_frequency
        self._rng = np.random.default_rng(random_seed)

    def generate_states(self, n_samples):

        # Simulate a random initial phase for each sinusoid
        self.phases = self._rng.uniform(0, 2 * np.pi, self.n_states)

        # Generator state time courses
        states = np.empty([n_samples, self.n_states], dtype=np.float32)
        t = np.arange(
            0, n_samples / self.sampling_frequency, 1.0 / self.sampling_frequency
        )
        for i in range(self.n_states):
            states[:, i] = self.relative_activation[i] + self.amplitudes[i] * np.sin(
                2 * np.pi * self.frequencies[i] * t + self.phases[i]
            )

        # Ensure state time courses sum to one at each time point
        states = softmax(states, axis=1)

        return states


class MixedSine_MVN(Simulation):
    """Simulates sinusoidal alphas with a multivariable normal observation model.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    relative_activation : np.ndarray or list
        Average value for each sine wave. Note, this might not be the
        mean value for each state time course because there is a softmax
        operation. This argument can use use to change the relative values
        of each state time course.
    amplitudes : np.ndarray or list
        Amplitude of each sinusoid.
    frequencies : np.ndarray or list
        Frequency of each sinusoid.
    sampling_frequency : float
        Sampling frequency.
    means : np.ndarray or str
        Mean vector for each state, shape should be (n_states, n_channels).
        Either a numpy array or 'zero' or 'random'.
    covariances : np.ndarray or str
        Covariance matrix for each state, shape should be (n_states,
        n_channels, n_channels). Either a numpy array or 'random'.
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for random number generator. Optional.
    """

    def __init__(
        self,
        n_samples: int,
        relative_activation: Union[np.ndarray, list],
        amplitudes: Union[np.ndarray, list],
        frequencies: Union[np.ndarray, list],
        sampling_frequency: float,
        means: Union[np.ndarray, str],
        covariances: Union[np.ndarray, str],
        n_states: int = None,
        n_channels: int = None,
        observation_error: float = 0.0,
        random_seed: int = None,
    ):
        # Observation model
        self.obs_mod = MVN(
            means=means,
            covariances=covariances,
            n_states=n_states,
            n_channels=n_channels,
            observation_error=observation_error,
            random_seed=random_seed,
        )

        self.n_states = self.obs_mod.n_states
        self.n_channels = self.obs_mod.n_channels

        # Soft mixed state time courses class
        self.sm = MixedSine(
            n_states=self.n_states,
            relative_activation=relative_activation,
            amplitudes=amplitudes,
            frequencies=frequencies,
            sampling_frequency=sampling_frequency,
            random_seed=random_seed,
        )

        super().__init__(n_samples=n_samples)

        # Simulate data
        self.state_time_course = self.generate_states(self.n_samples)
        self.time_series = self.obs_mod.simulate_data(self.state_time_course)

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        elif attr in dir(self.sm):
            return getattr(self.sm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")
