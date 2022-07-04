"""Functions to analyse time series data.

"""

import numpy as np
from scipy.signal.windows import hann


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


def window_mean(data, window_length, step_size=1, n_sub_windows=1):
    """Applies a windowing function to a time series and takes the mean.

    Parameters
    ----------
    data : np.ndarray
        Time series data. Shape is (n_samples, n_modes).
    window_length : int
        Number of data points in a window.
    step_size : int
        Step size for shifting the window.
    n_sub_windows : int
        Should we split the window into a set of sub-windows and average each sub-window.

    Returns
    -------
    a : np.ndarray
        Mean for each window.
    """

    # Number of samples and modes
    n_samples = data.shape[0]
    n_modes = data.shape[1]

    # First pad the data
    data = np.pad(data, window_length // 2)[
        :, window_length // 2 : window_length // 2 + n_modes
    ]

    # Window to apply to the data
    window = hann(window_length // n_sub_windows)

    # Indices of time points to calculate a periodogram for
    time_indices = range(0, n_samples, step_size)
    n_windows = n_samples // step_size

    # Array to hold mean of data multiplied by the windowing function
    a = np.empty([n_windows, n_modes], dtype=np.float32)
    for i in range(n_windows):

        # Alpha in the window
        j = time_indices[i]
        a_window = data[j : j + window_length]

        # Calculate data for the sub-window by taking the mean
        # over time after applying the windowing function
        a_sub_window = np.empty([n_sub_windows, n_modes], dtype=np.float32)
        for k in range(n_sub_windows):
            a_sub_window[k] = np.mean(
                a_window[
                    k
                    * window_length
                    // n_sub_windows : (k + 1)
                    * window_length
                    // n_sub_windows
                ]
                * window[..., np.newaxis],
                axis=0,
            )

        # Average data for each sub-window
        a[i] = np.mean(a_sub_window, axis=0)

    return a
