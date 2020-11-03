"""Functions to analyse time series data.

"""

import numpy as np


def get_state_time_series(
    data: np.ndarray, state_mixing_factors: np.ndarray
) -> np.ndarray:
    """Calculate a time series for each state.

    Time series calculated as the raw time series (of preprocessed data) multiplied
    by the state probability.


    Parameters
    ----------
    data : np.ndarray
        Raw data time series with shape (n_samples, n_channels).
    state_mixing_factors : np.ndarray
        State mixing factors alpha_t with shape (n_samples, n_states).

    Returns
    -------
    np.ndarray
        Time series for each state. Shape is (n_states, n_samples, n_channels).

    """
    # Make sure the data and state time courses have the same length
    if data.shape[0] != state_mixing_factors.shape[0]:
        raise ValueError(
            "data and state_mixing_factors have different lengths:"
            + f"data.shape[0]={data.shape[0]},"
            + f"state_mixing_factors.shape[0]={state_mixing_factors.shape[0]}"
        )

    # Number of samples, channels and states
    n_samples = data.shape[0]
    n_channels = data.shape[1]
    n_states = state_mixing_factors.shape[1]

    # Get the corresponding time series for when a state is on
    state_time_series = np.empty([n_states, n_samples, n_channels])
    for i in range(n_states):
        state_time_series[i] = data * state_mixing_factors[:, i, np.newaxis]

    return state_time_series
