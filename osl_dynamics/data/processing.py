"""Functions to process data.

"""

import numpy as np
from tqdm import tqdm
from scipy import signal

from osl_dynamics import array_ops
from osl_dynamics.data.spm import SPM


def standardize(
    time_series,
    axis=0,
    create_copy=True,
):
    """Standardizes a time series.

    Returns a time series with zero mean and unit variance.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data.
    axis : int
        Axis on which to perform the transformation.
    create_copy : bool
        Should we return a new array containing the standardized data or modify
        the original time series array?

    Returns
    -------
    std_time_series :  np.ndarray
        Standardized data.
    """
    mean = np.expand_dims(np.mean(time_series, axis=axis), axis=axis)
    std = np.expand_dims(np.std(time_series, axis=axis), axis=axis)
    if create_copy:
        std_time_series = (np.copy(time_series) - mean) / std
    else:
        std_time_series = (time_series - mean) / std
    return std_time_series


def time_embed(time_series, n_embeddings):
    """Performs time embedding.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data.
    n_embeddings : int
        Number of samples in which to shift the data.

    Returns
    -------
    sliding_window_view
        Time embedded data.
    """

    if n_embeddings % 2 == 0:
        raise ValueError("n_embeddings must be an odd number.")

    te_shape = (
        time_series.shape[0] - (n_embeddings - 1),
        time_series.shape[1] * n_embeddings,
    )
    return (
        array_ops.sliding_window_view(x=time_series, window_shape=te_shape[0], axis=0)
        .T[..., ::-1]
        .reshape(te_shape)
    )


def temporal_filter(time_series, low_freq, high_freq, sampling_frequency, order=5):
    """Apply temporal filtering.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data. Shape is (n_samples, n_channels).
    low_freq : float
        Frequency in Hz for a high pass filter.
        Only used if amplitude_envelope=True.
    high_freq : float
        Frequency in Hz for a low pass filter.
        Only used if amplitude_envelope=True.
    sampling_frequency : float
        Sampling frequency in Hz.
    order : int
        Order for a butterworth filter.

    Returns
    -------
    filtered_time_series : numpy.ndarray
        Filtered time series. Shape is (n_samples, n_channels).
    """
    if low_freq is None and high_freq is None:
        # No filtering
        return time_series

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
    filtered_time_series = signal.filtfilt(b, a, time_series, axis=0).astype(
        time_series.dtype
    )

    return filtered_time_series


def trim_time_series(
    time_series,
    sequence_length,
    discontinuities=None,
    concatenate=False,
):
    """Trims a time seris.

    Removes data points lost to separating a time series into sequences.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data for all subjects concatenated.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    discontinuities : list of int
        Length of each subject's data. If nothing is passed we assume the entire
        time series is continuous.
    concatenate : bool
        Should we concatenate the data for segment?

    Returns
    -------
    trimmed_time_series : list of np.ndarray
        Trimmed time series.
    """
    if isinstance(time_series, np.ndarray):
        if discontinuities is None:
            # Assume entire time series corresponds to a single subject
            ts = [time_series]
        else:
            # Separate the time series for each subject
            ts = []
            for i in range(len(discontinuities)):
                start = sum(discontinuities[:i])
                end = sum(discontinuities[: i + 1])
                ts.append(time_series[start:end])
    elif isinstance(time_series, list):
        ts = time_series
    else:
        raise ValueError("time_series must be a list or numpy array.")

    # Remove data points lost to separating into sequences
    for i in range(len(ts)):
        n_sequences = ts[i].shape[0] // sequence_length
        ts[i] = ts[i][: n_sequences * sequence_length]

    if len(ts) == 1:
        ts = ts[0]

    elif concatenate:
        ts = np.concatenate(ts)

    return ts


def make_channels_consistent(spm_filenames, scanner, output_folder="."):
    """Removes channels that are not present in all subjects.

    Parameters
    ----------
    spm_filenames : list of str
        Path to SPM files containing the preprocessed data.
    scanner : str
        Type of scanner used to record MEG data. Either 'ctf' or 'elekta'.
    output_folder : str
        Path to folder to write preprocessed data to. Optional, default
        is the current working directory.
    """
    if scanner not in ["ctf", "elekta"]:
        raise ValueError("scanner must be 'ctf' or 'elekta'.")

    # Get the channel labels
    channel_labels = []
    for filename in tqdm(spm_filenames, desc="Loading files", ncols=98):
        spm = SPM(filename, load_data=False)
        channel_labels.append(spm.channel_labels)

    # Find channels that are common to all SPM files only keeping the MEG
    # Recordings. N.b. the ordering of this list is random.
    common_channels = set(channel_labels[0]).intersection(*channel_labels)
    if scanner == "ctf":
        common_channels = [channel for channel in common_channels if "M" in channel]
    elif scanner == "elekta":
        common_channels = [channel for channel in common_channels if "MEG" in channel]

    # Write the channel labels to file in the correct order
    with open(output_folder + "/channels.dat", "w") as file:
        for channel in spm.channel_labels:
            if channel in common_channels:
                file.write(channel + "\n")

    # Write data to file only keeping the common channels
    for i in tqdm(range(len(spm_filenames)), desc="Writing files", ncols=98):
        spm = SPM(spm_filenames[i], load_data=True)
        channels = [label in common_channels for label in spm.channel_labels]

        output_filename = output_folder + f"/subject{i}.npy"
        output_data = spm.data[:, channels].astype(np.float32)
        np.save(output_filename, output_data)
