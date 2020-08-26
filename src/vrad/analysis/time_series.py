"""Functions to analyse time series data.

"""

import numpy as np


def get_state_time_series(data, state_probabilities):
    """Returns the data for when a state is on.

    Time series calculated as the raw time series (of preprocessed data) multiplied
    by the state probability.
    """

    # Make sure the data and state time courses have the same length
    n_samples = min(data.shape[0], state_probabilities.shape[0])
    data = data[:n_samples]
    state_probabilities = state_probabilities[:n_samples]

    # Number of states and channels
    n_states = state_probabilities.shape[1]
    n_channels = data.shape[1]

    # Get the corresponding time series for when a state is on
    state_time_series = np.empty([n_states, n_samples, n_channels])
    for i in range(n_states):
        state_time_series[i] = data * state_probabilities[:, i, np.newaxis]

    return state_time_series
