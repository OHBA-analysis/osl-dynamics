"""Class and functions to process data.

"""

import warnings

import numpy as np
import yaml
from tqdm import tqdm
from scipy import signal
from osl_dynamics import array_ops
from osl_dynamics.data.spm import SPM
from osl_dynamics.utils.misc import MockArray


class Processing:
    """Class for manipulating time series in the Data object.

    Parameters
    ----------
    n_embeddings : int
        Number of embeddings.
    keep_memmaps_on_close : bool
        Should we keep the memory maps when we delete the object?
    """

    def __init__(self, n_embeddings, keep_memmaps_on_close=False):
        self.n_embeddings = n_embeddings
        self.prepared = False
        self.prepared_data_filenames = []
        self.keep_memmaps_on_close = keep_memmaps_on_close

    def _process_from_yaml(self, file, **kwargs):
        with open(file) as f:
            settings = yaml.load(f, Loader=yaml.Loader)

        prep_settings = settings.get("prepare", {})
        if prep_settings.get("do", False):
            self.prepare(
                n_embeddings=prep_settings.get("n_embeddings"),
                n_pca_components=prep_settings.get("n_pca_components", None),
                whiten=prep_settings.get("whiten", False),
                load_memmaps=prep_settings.get("load_memmaps", True),
            )

    def prepare(
        self,
        amplitude_envelope=False,
        low_freq=None,
        high_freq=None,
        n_window=1,
        n_embeddings=1,
        n_pca_components=None,
        pca_components=None,
        whiten=False,
        load_memmaps=True,
    ):
        """Prepares data to train the model with.

        If amplitude_envelope=True, first we filter the data then
        calculate a Hilbert transform and take the absolute value.
        We then apply a sliding window moving average. Finally, we
        standardize the data.

        Otherwise, we standardize the data, perform time-delay embedding,
        then PCA, then whiten. Finally, the data is standardized again.

        If no arguments are passed, the data is just standardized.

        Parameters
        ----------
        amplitude_envelope : bool
            Should we prepare amplitude envelope data?
        low_freq : float
            Frequency in Hz for a high pass filter.
            Only used if amplitude_envelope=True.
        high_freq : float
            Frequency in Hz for a low pass filter.
            Only used if amplitude_envelope=True.
        n_window : int
            Number of data points in a sliding window to apply to the amplitude
            envelope data. Only used if amplitude_envelope=True.
        n_embeddings : int
            Number of data points to embed the data.
            Only used if amplitude_envelope=False.
        n_pca_components : int
            Number of PCA components to keep. Default is no PCA.
            Only used if amplitude_envelope=False.
        pca_components : np.ndarray
            PCA components to apply if they have already been calculated.
            Only used if amplitude_envelope=False.
        whiten : bool
            Should we whiten the PCA'ed data?
            Only used if amplitude_envelope=False.
        load_memmaps: bool
            Should we load the data into the memmaps?
        """
        if self.prepared:
            warnings.warn(
                "Previously prepared data will be overwritten.", RuntimeWarning
            )

        self.amplitude_envelope = amplitude_envelope
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.n_window = n_window
        self.n_embeddings = n_embeddings
        self.n_te_channels = self.n_raw_data_channels * n_embeddings
        self.n_pca_components = n_pca_components
        self.pca_components = pca_components
        self.whiten = whiten
        self.load_memmaps = load_memmaps

        # Prepare data (either amplitude envelope or time-delay embedded)
        if amplitude_envelope:
            self.prepare_amp_env(low_freq, high_freq, n_window)
        else:
            self.prepare_tde(n_embeddings, n_pca_components, pca_components, whiten)

        self.prepared = True

    def prepare_amp_env(self, low_freq=None, high_freq=None, n_window=1):
        """Prepare amplitude envelope data."""

        # Validation
        if (
            low_freq is not None or high_freq is not None
        ) and self.sampling_frequency is None:
            raise ValueError(
                "Data.sampling_frequency must be set if we are filtering the data. "
                + "Use Data.set_sampling_frequency() or pass "
                + "Data(..., sampling_frequency=...) when creating the Data object."
            )

        # Create filenames for memmaps (i.e. self.prepared_data_filenames)
        self.prepare_memmap_filenames()

        # Prepare the data
        for raw_data_memmap, prepared_data_file in zip(
            tqdm(self.raw_data_memmaps, desc="Preparing data", ncols=98),
            self.prepared_data_filenames,
        ):
            # Filtering
            prepared_data = temporal_filter(
                raw_data_memmap, low_freq, high_freq, self.sampling_frequency
            )

            # Hilbert transform
            prepared_data = np.abs(signal.hilbert(prepared_data))

            # Moving average filter
            prepared_data = np.array(
                [
                    np.convolve(
                        prepared_data[:, i], np.ones(n_window) / n_window, mode="valid"
                    )
                    for i in range(prepared_data.shape[1])
                ],
                dtype=np.float32,
            ).T

            # Create a memory map for the prepared data
            if self.load_memmaps:
                prepared_data_memmap = MockArray.get_memmap(
                    prepared_data_file, prepared_data.shape, dtype=np.float32
                )

            # Standardise to get the final data
            prepared_data_memmap = standardize(prepared_data, create_copy=False)
            self.prepared_data_memmaps.append(prepared_data_memmap.astype(np.float32))

        # Update subjects to return the prepared data
        self.subjects = self.prepared_data_memmaps

    def prepare_tde(
        self,
        n_embeddings=1,
        n_pca_components=None,
        pca_components=None,
        whiten=False,
    ):
        """Prepares time-delay embedded data to train the model with."""

        if n_pca_components is not None and pca_components is not None:
            raise ValueError("Please only pass n_pca_components or pca_components.")

        if pca_components is not None and not isinstance(pca_components, np.ndarray):
            raise ValueError("pca_components must be a numpy array.")

        # Create filenames for memmaps (i.e. self.prepared_data_filenames)
        self.prepare_memmap_filenames()

        # Principle component analysis (PCA)
        # NOTE: the approach used here only works for zero mean data
        if n_pca_components is not None:

            # Calculate the PCA components by performing SVD on the covariance
            # of the data
            covariance = np.zeros([self.n_te_channels, self.n_te_channels])
            for raw_data_memmap in tqdm(
                self.raw_data_memmaps, desc="Calculating PCA components", ncols=98
            ):
                # Standardise and time embed the data
                std_data = standardize(raw_data_memmap)
                te_std_data = time_embed(std_data, n_embeddings)

                # Calculate the covariance of the entire dataset
                covariance += np.transpose(te_std_data) @ te_std_data

                # Clear data in memory
                del std_data, te_std_data

            # Use SVD to calculate PCA components
            u, s, vh = np.linalg.svd(covariance)
            u = u[:, :n_pca_components].astype(np.float32)
            explained_variance = np.sum(s[:n_pca_components]) / np.sum(s)
            print(f"Explained variance: {100 * explained_variance:.1f}%")
            s = s[:n_pca_components].astype(np.float32)
            if whiten:
                u = u @ np.diag(1.0 / np.sqrt(s))
            self.pca_components = u

        # Prepare the data
        for raw_data_memmap, prepared_data_file in zip(
            tqdm(self.raw_data_memmaps, desc="Preparing data", ncols=98),
            self.prepared_data_filenames,
        ):
            # Standardise and time embed the data
            std_data = standardize(raw_data_memmap)
            te_std_data = time_embed(std_data, n_embeddings)

            # Apply PCA to get the prepared data
            if self.pca_components is not None:
                prepared_data = te_std_data @ self.pca_components

            # Otherwise, the time embedded data is the prepared data
            else:
                prepared_data = te_std_data

            # Create a memory map for the prepared data
            if self.load_memmaps:
                prepared_data_memmap = MockArray.get_memmap(
                    prepared_data_file, prepared_data.shape, dtype=np.float32
                )

            # Standardise to get the final data
            prepared_data_memmap = standardize(prepared_data, create_copy=False)
            self.prepared_data_memmaps.append(prepared_data_memmap)

            # Clear intermediate data
            del std_data, te_std_data, prepared_data

        # Update subjects to return the prepared data
        self.subjects = self.prepared_data_memmaps

    def prepare_memmap_filenames(self):
        prepared_data_pattern = "prepared_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(self.n_subjects)), identifier=self._identifier
        )
        self.prepared_data_filenames = [
            str(self.store_dir / prepared_data_pattern.format(i=i))
            for i in range(self.n_subjects)
        ]

        self.prepared_data_memmaps = []

    def trim_time_series(
        self,
        sequence_length=None,
        n_embeddings=1,
        prepared=True,
        concatenate=False,
    ):
        """Trims the data time series.

        Removes the data points that are lost when the data is prepared,
        i.e. due to time embedding and separating into sequences, but does not
        perform time embedding or batching into sequences on the time series.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        n_embeddings : int
            Number of data points to embed the data.
        prepared : bool
            Should we return the prepared data? If not we return the raw data.
        concatenate : bool
            Should we concatenate the data for each subject?

        Returns
        -------
        list of np.ndarray
            Trimed time series for each subject.
        """
        if self.n_embeddings is None:
            # Data has not been prepared so we can't trim the prepared data
            prepared = False

        if not prepared:
            # We're trimming the raw data, how many time embedding data
            # points do we need to remove?
            n_embeddings = self.n_embeddings or n_embeddings

        # What data should we trim?
        if prepared:
            memmaps = self.subjects
        else:
            memmaps = self.raw_data_memmaps

        trimmed_time_series = []
        for memmap in memmaps:

            # Remove data points lost to time embedding
            if n_embeddings != 1:
                memmap = memmap[n_embeddings // 2 : -(n_embeddings // 2)]

            # Remove data points lost to separating into sequences
            if sequence_length is not None:
                n_sequences = memmap.shape[0] // sequence_length
                memmap = memmap[: n_sequences * sequence_length]

            trimmed_time_series.append(memmap)

        if concatenate or len(trimmed_time_series) == 1:
            trimmed_time_series = np.concatenate(trimmed_time_series)

        return trimmed_time_series


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
    filtered_time_series = signal.filtfilt(b, a, time_series).astype(time_series.dtype)

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
