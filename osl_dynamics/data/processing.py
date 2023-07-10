"""Functions to process data.

"""

import mne
import numpy as np
from scipy import signal

from osl_dynamics import array_ops


def standardize(x, axis=0, create_copy=True):
    """Standardizes a time series.

    Returns a time series with zero mean and unit variance.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    axis : int, optional
        Axis on which to perform the transformation.
    create_copy : bool, optional
        Should we return a new array containing the standardized data or modify
        the original time series array?

    Returns
    -------
    X :  np.ndarray
        Standardized data.
    """
    mean = np.expand_dims(np.mean(x, axis=axis), axis=axis)
    std = np.expand_dims(np.std(x, axis=axis), axis=axis)
    if create_copy:
        X = (np.copy(x) - mean) / std
    else:
        X = (x - mean) / std
    return X


def time_embed(x, n_embeddings):
    """Performs time embedding.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    n_embeddings : int
        Number of samples in which to shift the data.

    Returns
    -------
    X : sliding_window_view
        Time embedded data. Shape is (n_samples - n_embeddings + 1,
        n_channels * n_embeddings).
    """
    if n_embeddings % 2 == 0:
        raise ValueError("n_embeddings must be an odd number.")

    # Shape of time embedded data
    te_shape = (x.shape[0] - (n_embeddings - 1), x.shape[1] * n_embeddings)

    # Perform time embedding
    X = (
        array_ops.sliding_window_view(x=x, window_shape=te_shape[0], axis=0)
        .T[..., ::-1]
        .reshape(te_shape)
    )

    return X


def temporal_filter(x, low_freq, high_freq, sampling_frequency, order=5):
    """Apply temporal filtering.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    low_freq : float
        Frequency in Hz for a high pass filter.
    high_freq : float
        Frequency in Hz for a low pass filter.
    sampling_frequency : float
        Sampling frequency in Hz.
    order : int, optional
        Order for a butterworth filter.

    Returns
    -------
    X : np.ndarray
        Filtered time series. Shape is (n_samples, n_channels).
    """
    if low_freq is None and high_freq is None:
        # No filtering
        return x

    if low_freq is None and high_freq is not None:
        btype = "lowpass"
        Wn = high_freq
    elif low_freq is not None and high_freq is None:
        btype = "highpass"
        Wn = low_freq
    else:
        btype = "bandpass"
        Wn = [low_freq, high_freq]

    # Create the filter
    b, a = signal.butter(order, Wn=Wn, btype=btype, fs=sampling_frequency)

    # Apply the filter
    X = signal.filtfilt(b, a, x, axis=0)

    return X.astype(x.dtype)


def amplitude_envelope(x):
    """Calculate amplitude envelope.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).

    Returns
    -------
    X : np.ndarray
        Amplitude envelope data. Shape is (n_samples, n_channels).
    """
    X = np.abs(signal.hilbert(x, axis=0))
    return X.astype(x.dtype)


def moving_average(x, n_window):
    """Calculate a moving average.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    n_window : int
        Number of data points in the sliding window. Must be odd.

    Returns
    -------
    X : np.ndarray
        Time series with sliding window applied.
        Shape is (n_samples - n_window + 1, n_channels).
    """
    if n_window % 2 == 0:
        raise ValueError("n_window must be odd.")
    X = np.array(
        [
            np.convolve(
                x[:, i],
                np.ones(n_window) / n_window,
                mode="valid",
            )
            for i in range(x.shape[1])
        ],
    ).T
    return X.astype(x.dtype)


def downsample(x, new_freq, old_freq):
    """Downsample.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    old_freq : float
        Old sampling frequency in Hz.
    new_freq : float
        New sampling frequency in Hz.

    Returns
    -------
    X : np.ndarray
        Downsampled time series.
        Shape is (n_samples * new_freq / old_freq, n_channels).
    """
    if old_freq < new_freq:
        raise ValueError(
            f"new frequency ({new_freq} Hz) must be less than old "
            + f"frequency ({old_freq} Hz)."
        )
    ratio = old_freq / new_freq
    X = mne.filter.resample(
        x.astype(np.float64),
        down=ratio,
        axis=0,
        verbose=False,
    )
    return X.astype(x.dtype)
