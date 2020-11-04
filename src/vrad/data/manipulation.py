from typing import Union

import numpy as np


def standardize(time_series: np.ndarray, discontinuities: np.ndarray) -> np.ndarray:
    """Standardizes time series data.

    Returns a time series standardized over continuous segments of data.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data.
    discontinuities : numpy.ndarray
        A set of time points at which the time series is discontinuous (e.g. because
        bad segments were removed in preprocessing).
    """
    for i in range(len(discontinuities)):
        start = sum(discontinuities[:i])
        end = sum(discontinuities[: i + 1])
        time_series[start:end] = scale(time_series[start:end], axis=0)
    return time_series


def scale(time_series: np.ndarray, axis: int = 0) -> np.ndarray:
    """Scales a time series.

    Returns a time series with zero mean and unit variance.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data.
    axis : int
        Axis on which to perform the transformation.
    """
    time_series -= time_series.mean(axis=axis)
    time_series /= time_series.std(axis=axis)
    return time_series


def time_embed(
    time_series: np.ndarray,
    discontinuities: np.ndarray,
    n_embeddings: int,
    output_file: str = None,
) -> np.ndarray:
    """Performs time embedding.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data.
    discontinuities : numpy.ndarray
        A set of time points at which the time series is discontinuous (e.g. because
        bad segments were removed in preprocessing).
    n_embeddings : int
        Number of samples in which to shift the data.
    output_file : str
    """

    if n_embeddings % 2 == 0:
        raise ValueError("n_embeddings must be an odd number.")

    # Unpack shape of the original data
    n_samples, n_channels = time_series.shape

    # If an output file hasn't been passed we create a numpy array for the
    # time embedded data
    if output_file is None:
        time_embedded_series = np.empty(
            [
                n_samples - (n_embeddings + 1) * len(discontinuities),
                n_channels * (n_embeddings + 2),
            ]
        )
    else:
        time_embedded_series = output_file

    # Loop through continuous segments of data
    for i in range(len(discontinuities)):
        n_segment = discontinuities[i]
        start = sum(discontinuities[:i])
        end = sum(discontinuities[: i + 1])
        original_time_series = time_series[start:end]

        # Generate time embedded data
        time_embedded_segment = np.empty(
            [n_segment - (n_embeddings + 1), n_channels * (n_embeddings + 2)]
        )
        for j in range(n_channels):
            for k in range(n_embeddings + 2):
                time_embedded_segment[
                    :, j * (n_embeddings + 2) + k
                ] = original_time_series[n_embeddings + 1 - k : n_segment - k, j]

        # Fill the final time embedded series array
        time_embedded_series[
            start - (n_embeddings + 1) * i : end - (n_embeddings + 1) * (i + 1)
        ] = time_embedded_segment

    return time_embedded_series


def num_batches(arr: np.ndarray, sequence_length: int, step_size: int = None):
    """Calculate the number of batches an array will be split into.

    Parameters
    ----------
    arr : numpy.ndarray
        Time series data.
    sequence_length : int
        Length of sequences which the data will be segmented in to.
    step_size : int
        The number of samples by which to move the sliding window between sequences.

    Returns
    -------
    num_batches : int

    """
    step_size = step_size or sequence_length
    final_slice_start = arr.shape[0] - sequence_length + 1
    index = np.arange(0, final_slice_start, step_size)[:, None] + np.arange(
        sequence_length
    )
    return len(index)


def trim_time_series(
    time_series: np.ndarray, discontinuities: np.ndarray, sequence_length,
):
    """Removes data points lost to separating a time series into sequences.

    Removes data points lost to time embedding and separating a time series
    into sequences.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data.
    discontinuities : numpy.ndarray
        A set of time points at which the time series is discontinuous (e.g. because
        bad segments were removed in preprocessing).
    sequence_length : int
    """

    # Separate the time series for each subject
    subject_data_lengths = [sum(d) for d in discontinuities]
    ts = []
    for i in range(len(subject_data_lengths)):
        start = sum(subject_data_lengths[:i])
        end = sum(subject_data_lengths[: i + 1])
        ts.append(time_series[start:end])

    # Remove data points lost to separating into sequences
    for i in range(len(ts)):
        n_sequences = ts[i].shape[0] // sequence_length
        ts[i] = ts[i][: n_sequences * sequence_length]

    return ts
