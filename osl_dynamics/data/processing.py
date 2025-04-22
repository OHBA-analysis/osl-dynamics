"""Functions to process data."""

import mne
import numpy as np
from scipy import signal, stats

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


def remove_bad_segments(
    x, window_length, significance_level=0.05, maximum_fraction=0.1
):
    """Automated bad segment removal using the G-ESD algorithm.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    window_length : int, optional
        Window length to used to calculate statistics.
        Defaults to twice the sampling frequency.
    significance_level : float, optional
        Significance level (p-value) to consider as an outlier.
    maximum_fraction : float, optional
        Maximum fraction of time series to mark as bad.

    Returns
    -------
    x : np.ndarray
        Time series with bad segments removed.
        Shape is (n_samples - n_bad_samples, n_channels).
    bad : np.ndarray
        Times of True (bad) or False (good) to indices whether
        a time point is good or bad. This is the full length of
        the original time series. Shape is (n_samples,).
    """

    def _gesd(X, alpha=significance_level, p_out=maximum_fraction, outlier_side=0):
        # Detect outliers using Generalized ESD test
        #
        # Args:
        # - X : data to detect outliers within
        # - alpha : significance level to detect at
        # - p_out : maximum fraction of time series to set as outliers
        # - outlier_side {-1,0,1} :
        #   - -1 -> outliers are all smaller
        #   -  0 -> outliers could be small/negative or large/positive
        #   -  1 -> outliers are all larger
        #
        # Returns: boolean mask for bad segments
        #
        # See: B. Rosner (1983). Percentage Points for a Generalized ESD
        # Many-Outlier Procedure. Technometrics 25(2), pp. 165-172.
        if outlier_side == 0:
            alpha = alpha / 2
        n_out = int(np.ceil(len(X) * p_out))
        if np.any(np.isnan(X)):
            y = np.where(np.isnan(X))[0]
            idx1, x2 = _gesd(X[np.isfinite(X)], alpha, n_out, outlier_side)
            idx = np.zeros_like(X).astype(bool)
            idx[y[idx1]] = True
        n = len(X)
        temp = X.copy()
        R = np.zeros(n_out)
        rm_idx = np.zeros(n_out, dtype=int)
        lam = np.zeros(n_out)
        for j in range(0, int(n_out)):
            i = j + 1
            if outlier_side == -1:
                rm_idx[j] = np.nanargmin(temp)
                sample = np.nanmin(temp)
                R[j] = np.nanmean(temp) - sample
            elif outlier_side == 0:
                rm_idx[j] = int(np.nanargmax(abs(temp - np.nanmean(temp))))
                R[j] = np.nanmax(abs(temp - np.nanmean(temp)))
            elif outlier_side == 1:
                rm_idx[j] = np.nanargmax(temp)
                sample = np.nanmax(temp)
                R[j] = sample - np.nanmean(temp)
            R[j] = R[j] / np.nanstd(temp)
            temp[int(rm_idx[j])] = np.nan
            p = 1 - alpha / (n - i + 1)
            t = stats.t.ppf(p, n - i - 1)
            lam[j] = ((n - i) * t) / (np.sqrt((n - i - 1 + t**2) * (n - i + 1)))
        mask = np.zeros(n).astype(bool)
        mask[rm_idx[np.where(R > lam)[0]]] = True
        return mask

    # Calculate metric for each window
    metrics = []
    indices = []
    starts = np.arange(0, x.shape[0], window_length)
    for i in range(len(starts)):
        start = starts[i]
        if i == len(starts) - 1:
            stop = None
        else:
            stop = starts[i] + window_length
        m = np.std(x[start:stop])
        metrics.append(m)
        indices += [i] * len(x[start:stop])

    # Detect outliers
    bad_metrics_mask = _gesd(metrics)
    bad_metrics_indices = np.where(bad_metrics_mask)[0]

    # Look up what indices in the original data are bad
    bad = np.isin(indices, bad_metrics_indices)

    # Return good data and bad indices
    return x[~bad], bad
