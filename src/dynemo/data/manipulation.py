import logging
import pathlib
from typing import List, Union

import numpy as np
import yaml
from sklearn.cluster import KMeans
from tqdm import tqdm
from dynemo import array_ops
from dynemo.utils import misc
from dynemo.utils.misc import MockArray

_logger = logging.getLogger("DyNeMo")
_rng = np.random.default_rng()


class Manipulation:
    """Class for manipulating time series in the Data object.

    Parameters
    ----------
    n_embeddings : int
        Number of embeddings.
    """

    def __init__(self, n_embeddings: int, keep_memmaps_on_close: bool = False):
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
            )

    def covariance_sample(
        self,
        segment_length: Union[int, List[int]],
        n_segments: Union[int, List[int]],
        n_clusters: int = None,
    ) -> np.ndarray:
        """Get covariances of a random selection of a time series.

        Given a time series, `data`, randomly select a set of samples of length(s)
        `segment_length` with `n_segments` of each selected. If `n_clusters` is not
        specified each of these covariances will be returned. Otherwise, a K-means
        clustering algorithm is run to return that `n_clusters` covariances.

        Lack of overlap between samples is *not* guaranteed.

        Parameters
        ----------
        data: np.ndarray
            The time series to be analyzed.
        segment_length: int or list of int
            Either the integer number of samples for each covariance, or a list with a
            range of values.
        n_segments: int or list of int
            Either the integer number of segments to select,
             or a list specifying the number
            of each segment length to be sampled.
        n_clusters: int
            The number of K-means clusters to find
            (default is not to perform clustering).

        Returns
        -------
        covariances: np.ndarray
            The calculated covariance matrices of the samples.
        """
        segment_lengths = misc.listify(segment_length)
        n_segments = misc.listify(n_segments)

        if len(n_segments) == 1:
            n_segments = n_segments * len(segment_lengths)

        if len(segment_lengths) != len(n_segments):
            raise ValueError(
                "`segment_lengths` and `n_samples` should have the same lengths."
            )

        covariances = []
        for segment_length, n_sample in zip(segment_lengths, n_segments):
            data = self.subjects[_rng.choice(self.n_subjects)]
            starts = _rng.choice(data.shape[0] - segment_length, n_sample)
            samples = data[np.asarray(starts)[:, None] + np.arange(segment_length)]

            transposed = samples.transpose(0, 2, 1)
            m1 = transposed - transposed.sum(2, keepdims=1) / segment_length
            covariances.append(np.einsum("ijk,ilk->ijl", m1, m1) / (segment_length - 1))
        covariances = np.concatenate(covariances)

        if n_clusters is None:
            return covariances

        flat_covariances = covariances.reshape((covariances.shape[0], -1))

        kmeans = KMeans(n_clusters=n_clusters, random_mode=0).fit(flat_covariances)

        kmeans_covariances = kmeans.cluster_centers_.reshape(
            (n_clusters, *covariances.shape[1:])
        )

        return kmeans_covariances

    def prepare(
        self,
        n_embeddings: int = 1,
        n_pca_components: int = None,
        pca_components: np.ndarray = None,
        whiten: bool = False,
    ):
        """Prepares data to train the model with.

        Performs standardization, time embedding and principle component analysis.

        Parameters
        ----------
        n_embeddings : int
            Number of data points to embed the data. Optional, default is 1.
        n_pca_components : int
            Number of PCA components to keep. Optional, default is no PCA.
        pca_components : np.ndarray
            PCA components to apply if they have already been calculated.
            Optional.
        whiten : bool
            Should we whiten the PCA'ed data? Optional, default is False.
        """
        if self.prepared:
            _logger.warning("Previously prepared data will be overwritten.")

        self.n_embeddings = n_embeddings
        self.n_te_channels = self.n_raw_data_channels * n_embeddings
        self.n_pca_components = n_pca_components
        self.pca_components = pca_components
        self.whiten = whiten
        self.prepared = True

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

    def delete_manipulation_memmaps(self):
        """Deletes memmaps and removes store_dir if empty."""
        if self.prepared_data_filenames is not None:
            for filename in self.prepared_data_filenames:
                pathlib.Path(filename).unlink(missing_ok=True)
        if self.store_dir.exists():
            if not any(self.store_dir.iterdir()):
                self.store_dir.rmdir()
        self.prepared_data_memmaps = None
        self.prepared_data_filenames = None

    def trim_raw_time_series(
        self,
        sequence_length: int = None,
        n_embeddings: int = None,
        concatenate: bool = False,
    ) -> list:
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
        concatenate : bool
            Should we concatenate the data for each subject? Optional, default
            is False.

        Returns
        -------
        list of np.ndarray
            Trimed time series for each subject.
        """
        n_embeddings = n_embeddings or self.n_embeddings

        if n_embeddings is None:
            raise ValueError(
                "n_embeddings has not been set. "
                + "Either pass it as an argument or call prepare."
            )

        if hasattr(self, "sequence_length"):
            sequence_length = self.sequence_length

        trimmed_raw_time_series = []
        for memmap in self.raw_data_memmaps:

            # Remove data points lost to time embedding
            if n_embeddings != 1:
                memmap = memmap[n_embeddings // 2 : -(n_embeddings // 2)]

            # Remove data points lost to separating into sequences
            if sequence_length is not None:
                n_sequences = memmap.shape[0] // sequence_length
                memmap = memmap[: n_sequences * sequence_length]

            trimmed_raw_time_series.append(memmap)

        if len(trimmed_raw_time_series) == 1:
            trimmed_raw_time_series = trimmed_raw_time_series[0]

        elif concatenate:
            trimmed_raw_time_series = np.concatenate(trimmed_raw_time_series)

        return trimmed_raw_time_series


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
    time_series: Union[list, np.ndarray],
    sequence_length: int,
    discontinuities: list = None,
    concatenate: bool = False,
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
    concatenate : bool
        Should we concatenate the data for segment? Optional, default
        is False.

    Returns
    -------
    list of np.ndarray
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
