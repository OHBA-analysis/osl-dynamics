"""Functions to analyse time series data.

"""

import numpy as np


def get_mode_time_series(data: np.ndarray, alpha: np.ndarray) -> np.ndarray:
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
    np.ndarray
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


def regress(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Find the parameters of a regression.

    If A = B @ C, where A, B and C are 2D matrices. This function calculates
    C using A and B: C = pinv(B) @ A.

    Parameters
    ----------
    A : np.ndarray
        First matrix. If this is not a 2D matrix, the extra dimensions are
        concatenated.
    B : np.ndarray
        Second array. Must be 2D.

    Returns
    -------
    C : np.ndarray
    """
    original_shape = A.shape
    new_shape = [B.shape[1]] + list(original_shape[1:])
    A = A.reshape(original_shape[0], -1)
    C = np.linalg.pinv(B) @ A
    C = C.reshape(new_shape)
    return C
