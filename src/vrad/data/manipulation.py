import numpy as np


def standardize(time_series: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardizes a time series.

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
    time_series: np.ndarray, n_embeddings: int, output_file: str = None,
) -> np.ndarray:
    """Performs time embedding.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data.
    n_embeddings : int
        Number of samples in which to shift the data.
    output_file : str
    """

    if n_embeddings % 2 == 0:
        raise ValueError("n_embeddings must be an odd number.")

    n_samples, n_channels = time_series.shape

    # If an output file hasn't been passed we create a numpy array for the
    # time embedded data
    if output_file is None:
        time_embedded_series = np.empty(
            [n_samples - (n_embeddings + 1), n_channels * (n_embeddings + 2)],
            dtype=np.float32,
        )
    else:
        time_embedded_series = output_file

    # Generate time embedded series
    for i in range(n_channels):
        for j in range(n_embeddings + 2):
            time_embedded_series[:, i * (n_embeddings + 1) + j] = time_series[
                n_embeddings + 1 - j : n_samples - j, i
            ]

    return time_embedded_series


def n_batches(arr: np.ndarray, sequence_length: int, step_size: int = None) -> int:
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
    int
        Number of batches.
    """
    step_size = step_size or sequence_length
    final_slice_start = arr.shape[0] - sequence_length + 1
    index = np.arange(0, final_slice_start, step_size)[:, None] + np.arange(
        sequence_length
    )
    return len(index)


def trim_time_series(
    time_series: np.ndarray, discontinuities: np.ndarray, sequence_length,
) -> np.ndarray:
    """Trims a time seris.

    Removes data points lost to separating a time series into sequences.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data.
    discontinuities : numpy.ndarray
        A set of time points at which the time series is discontinuous (e.g. because
        bad segments were removed in preprocessing).
    sequence_length : int

    Returns
    -------
    np.ndarray
        Trimmed time series.
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
