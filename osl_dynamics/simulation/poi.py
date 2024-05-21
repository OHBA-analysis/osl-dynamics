"""Classes for simulating a Poisson time series.

"""

import numpy as np


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
        rates,
        n_states=None,
        n_channels=None,
    ):
        if isinstance(rates, np.ndarray):
            self.n_states = rates.shape[0]
            self.n_channels = rates.shape[1]
            self.rates = rates

        elif not isinstance(rates, np.ndarray):
            if n_states is None or n_channels is None:
                raise ValueError(
                    "If we are generating rates, "
                    + "n_states and n_channels must be passed."
                )
            self.n_states = n_states
            self.n_channels = n_channels
            self.rates = self.create_rates(rates)

    def create_rates(self, option, eps=1e-2):
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

    def simulate_data(self, state_time_course):
        n_samples = state_time_course.shape[0]
        data = np.empty([n_samples, self.n_channels])

        # Generate data
        for i in range(n_samples):
            state = np.argmax(state_time_course[i])
            data[i] = np.random.poisson(self.rates[state])

        return data
