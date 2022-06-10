"""Functions to analyse time series data.

"""

import numpy as np


def get_mode_time_series(data, alpha):
    """Calculate a time series for each mode.

    Time series calculated as the raw time series (of preprocessed data) multiplied
    by the mode probability.

    Parameters
    ----------
    data : np.ndarray
        Raw data time series with shape (n_samples, n_channels).
    alpha : np.ndarray
        Mode mixing factors alpha_t with shape (n_samples, n_modes).

    Returns
    -------
    mode_time_series : np.ndarray
        Time series for each mode. Shape is (n_modes, n_samples, n_channels).

    """
    # Make sure the data and mode time courses have the same length
    if data.shape[0] != alpha.shape[0]:
        raise ValueError(
            "data and alpha have different lengths:"
            + f"data.shape[0]={data.shape[0]},"
            + f"alpha.shape[0]={alpha.shape[0]}"
        )

    # Number of samples, channels and modes
    n_samples = data.shape[0]
    n_channels = data.shape[1]
    n_modes = alpha.shape[1]

    # Get the corresponding time series for when a mode is on
    mode_time_series = np.empty([n_modes, n_samples, n_channels])
    for i in range(n_modes):
        mode_time_series[i] = data * alpha[:, i, np.newaxis]

    return mode_time_series
