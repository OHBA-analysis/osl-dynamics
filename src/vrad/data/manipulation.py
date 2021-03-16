from tqdm import tqdm

import numpy as np
from scipy.signal import butter, lfilter

from vrad import array_ops
from vrad.utils.misc import MockArray


class Manipulation:
    """Class for manipulating time series in the Data object.

    Parameters
    ----------
    n_embeddings : int
        Number of embeddings. Optional. Can be passed if data has already been prepared.
    n_pca_components : int
        Number of PCA components. Optional. Can be passed if data has already been
        prepared.
    whiten : bool
        Was whitening applied during the PCA? Optional.
    prepared : bool
        Flag indicating if data has already been prepared. Optional.
    """

    def __init__(
        self,
        n_embeddings: int,
        n_pca_components: int,
        whiten: bool,
        prepared: bool,
    ):
        self.n_embeddings = n_embeddings
        self.n_pca_components = n_pca_components
        self.whiten = whiten
        self.prepared = prepared

    def _process_from_yaml(self, file, **kwargs):
        with open(file) as f:
            settings = yaml.load(f, Loader=yaml.Loader)

        prep_settings = settings.get("prepare", {})
        if prep_settings.get("do", False):
            self.prepare(
                n_embeddings=prep_settings.get("n_embeddings"),
                n_pca_components=prep_settings.get("n_pca_components", None),
                whiten=prep_settings.get("whiten", False),
            )

    def _pre_pca(self, raw_data, filter_range, filter_order, n_embeddings):
        std_data = standardize(raw_data)
        if filter_range is not None:
            f_std_data = bandpass_filter(
                std_data, filter_range, filter_order, self.sampling_frequency
            )
        else:
            f_std_data = std_data
        te_f_std_data = time_embed(f_std_data, n_embeddings)
        return te_f_std_data

    def prepare(
        self,
        filter_range: list = None,
        filter_order: int = None,
        n_embeddings: int = 1,
        n_pca_components: int = None,
        whiten: bool = False,
    ):
        """Prepares data to train the model with.

        Performs standardization, time embedding and principle component analysis.

        Parameters
        ----------
        filter_range : list
            Min and max frequencies to bandpass filter. Optional, default is
            no filtering. A butterworth filter is applied.
        filter_order : int
            Order of the butterworth filter. Optional. Required is filter_range
            is passed.
        n_embeddings : int
            Number of data points to embed the data. Optional, default is 1.
        n_pca_components : int
            Number of PCA components to keep. Optional, default is no PCA.
        whiten : bool
            Should we whiten the PCA'ed data? Optional, default is False.
        """
        if self.prepared:
            _logger.warning("Previously prepared data will be overwritten.")

        if filter_range is not None:
            if filter_order is None:
                raise ValueError(
                    "If we are filtering the data, filter_order must be passed."
                )
            if self.sampling_frequency is None:
                raise ValueError(
                    "If we are filtering the data, sampling_frequency must be passed."
                )

        # Class attributes related to data preparation
        self.filter_range = filter_range
        self.filter_order = filter_order
        self.n_embeddings = n_embeddings
        self.n_te_channels = self.n_raw_data_channels * n_embeddings
        self.n_pca_components = n_pca_components
        self.whiten = whiten
        self.prepared = True

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
                # Standardise, filter and time embed the data, this function
                # returns a copy of the data that is held in memory
                te_f_std_data = self._pre_pca(
                    raw_data_memmap, filter_range, filter_order, n_embeddings
                )

                # Calculate the covariance of the entire dataset
                covariance += np.transpose(te_f_std_data) @ te_f_std_data

                # Clear data in memory
                del te_f_std_data

            # Use SVD to calculate PCA components
            u, s, vh = np.linalg.svd(covariance)
            u = u[:, :n_pca_components].astype(np.float32)
            s = s[:n_pca_components].astype(np.float32)
            if whiten:
                u = u @ np.diag(1.0 / np.sqrt(s))
            self.pca_weights = u
        else:
            self.pca_weights = None

        # Prepare the data
        for raw_data_memmap, prepared_data_file in zip(
            tqdm(self.raw_data_memmaps, desc="Preparing data", ncols=98),
            self.prepared_data_filenames,
        ):
            # Standardise, filter and time embed the data, this function returns a
            # copy of the data that is held in memory
            te_f_std_data = self._pre_pca(
                raw_data_memmap, filter_range, filter_order, n_embeddings
            )

            # Apply PCA to get the prepared data
            if self.pca_weights is not None:
                prepared_data = te_f_std_data @ self.pca_weights

            # Otherwise, the time embedded data is the prepared data
            else:
                prepared_data = te_f_std_data

            # Create a memory map for the prepared data
            prepared_data_memmap = MockArray.get_memmap(
                prepared_data_file, prepared_data.shape, dtype=np.float32
            )

            # Record the mean and standard deviation of the prepared
            # data and standardise to get the final data
            self.prepared_data_mean.append(np.mean(prepared_data, axis=0))
            self.prepared_data_std.append(np.std(prepared_data, axis=0))
            prepared_data_memmap = standardize(prepared_data, create_copy=False)
            self.prepared_data_memmaps.append(prepared_data_memmap)

            # Clear intermediate data
            del te_f_std_data, prepared_data

        # Update subjects to return the prepared data
        self.subjects = self.prepared_data_memmaps

    def prepare_memmap_filenames(self):
        self.prepared_data_pattern = (
            "prepared_data_{{i:0{width}d}}_{identifier}.npy".format(
                width=len(str(self.n_subjects)), identifier=self._identifier
            )
        )

        # Prepared data memory maps (time embedded and pca'ed)
        self.prepared_data_memmaps = []
        self.prepared_data_filenames = [
            str(self.store_dir / self.prepared_data_pattern.format(i=i))
            for i in range(self.n_subjects)
        ]
        self.prepared_data_mean = []
        self.prepared_data_std = []

    def trim_raw_time_series(
        self,
        sequence_length: int = None,
        n_embeddings: int = None,
    ) -> np.ndarray:
        """Trims the raw preprocessed data time series.

        Removes the data points that are removed when the data is prepared,
        i.e. due to time embedding and separating into sequences, but does not
        perform time embedding or batching into sequences on the time series.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        n_embeddings : int
            Number of data points to embed the data.

        Returns
        -------
        np.ndarray
            Trimed time series.
        """
        if self.prepared:
            n_embeddings = self.n_embeddings or n_embeddings

        trimmed_raw_time_series = []
        for memmap in self.raw_data_memmaps:

            # Remove data points which are removed due to time embedding
            if n_embeddings is not None:
                memmap = memmap[n_embeddings // 2 : -n_embeddings // 2]

            # Remove data points which are removed due to separating into sequences
            if sequence_length is not None:
                n_sequences = memmap.shape[0] // sequence_length
                memmap = memmap[: n_sequences * sequence_length]

            trimmed_raw_time_series.append(memmap)

        return trimmed_raw_time_series


def bandpass_filter(
    time_series: np.ndarray,
    filter_range: list,
    filter_order: int,
    sampling_frequency: float,
):
    """Filters a time series.

    Applies a butterworth filter to a time series.

    Parameters
    ----------
    time_series : np.ndarray
        Time series data.
    filter_range : list
        Min and max frequency to keep.
    filter_order : int
        Order of the butterworth filter.
    sampling_frequency : float
        Sampling frequency of the time series.

    Returns
    -------
    np.ndarray
        Filtered time series.
    """
    nyq = 0.5 * sampling_frequency
    filter_range[0] /= nyq
    filter_range[1] /= nyq
    b, a = butter(filter_order, filter_range, btype="band")
    filtered_time_series = lfilter(b, a, time_series)
    return filtered_time_series


def standardize(
    time_series: np.ndarray,
    axis: int = 0,
    create_copy: bool = True,
) -> np.ndarray:
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
        the original time series array? Optional, default is True.
    """
    mean = np.mean(time_series, axis=axis)
    std = np.std(time_series, axis=axis)
    if create_copy:
        std_time_series = (np.copy(time_series) - mean) / std
    else:
        std_time_series = (time_series - mean) / std
    return std_time_series


def time_embed(time_series: np.ndarray, n_embeddings: int):
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


def trim_time_series(
    time_series: np.ndarray,
    sequence_length: int,
    discontinuities: list = None,
) -> np.ndarray:
    """Trims a time seris.

    Removes data points lost to separating a time series into sequences.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data for all subjects concatenated.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    discontinuities : list of int
        Length of each subject's data. Optional. If nothing is passed we
        assume the entire time series is continuous.

    Returns
    -------
    list of np.ndarray
        Trimmed time series.
    """
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

    # Remove data points lost to separating into sequences
    for i in range(len(ts)):
        n_sequences = ts[i].shape[0] // sequence_length
        ts[i] = ts[i][: n_sequences * sequence_length]

    return ts if len(ts) > 1 else ts[0]
