"""Base class for handling data."""

import atexit
import hashlib
import json
import math
import os
import re
import logging
import pathlib
import pickle
import uuid
from contextlib import contextmanager
from functools import partial
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pqdm.threads import pqdm
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from osl_dynamics.data import processing, rw, sign_flipping, tf as dtf
from osl_dynamics.utils import misc

_logger = logging.getLogger("osl-dynamics")


def _raise_exceptions(results: list, what: str = "array") -> None:
    """Log all exceptions in a list of pqdm results and raise the first.

    pqdm returns exceptions raised in worker threads as items in the
    results list rather than raising them.
    """
    first_exception = None
    for i, e in enumerate(results):
        if isinstance(e, Exception):
            e.args = (f"{what} {i}: {e}",)
            _logger.exception(e, exc_info=False)
            if first_exception is None:
                first_exception = e
    if first_exception is not None:
        raise first_exception


class Data:
    """Data Class.

    The Data class enables the input and processing of data. When given a list
    of files, it produces a set of numpy memory maps which contain their raw
    data. It also provides methods for batching data and creating TensorFlow
    Datasets.

    Parameters
    ----------
    inputs : list of str or pathlib.Path or str or pathlib.Path or np.ndarray
        - A path (:code:`str` or :code:`pathlib.Path`) to a directory
          containing :code:`.npy` files. Each :code:`.npy` file should be a
          subject or session.
        - A list of paths (:code:`str` or :code:`pathlib.Path`) to
          :code:`.npy`, :code:`.mat` or :code:`.fif` files. Each file should
          be a subject or session. If a :code:`.fif` file is passed is must
          end with :code:`'raw.fif'` or :code:`'epo.fif'`.
        - A numpy array. The array will be treated as continuous data from the
          same subject.
        - A list of numpy arrays. Each numpy array should be the data for a
          subject or session.

        The data files or numpy arrays should be in the format (n_samples,
        n_channels). If your data is in (n_channels, n_samples) format, use
        :code:`time_axis_first=False`.
    data_field : str, optional
        If a MATLAB (:code:`.mat`) file is passed, this is the field that
        corresponds to the time series data. By default we read the field
        :code:`'X'`. If a numpy (:code:`.npy`) or fif (:code:`.fif`) file is
        passed, this is ignored.
    picks : str or list of str, optional
        Only used if a fif file is passed. We load the data using the
        `mne.io.Raw.get_data <https://mne.tools/stable/generated/mne.io\
        .Raw.html#mne.io.Raw.get_data>`_ method. We pass this argument to the
        :code:`Raw.get_data` method. By default :code:`picks=None` retrieves
        all channel types.
    reject_by_annotation : str, optional
        Only used if a fif file is passed. We load the data using the
        `mne.io.Raw.get_data <https://mne.tools/stable/generated/mne.io\
        .Raw.html#mne.io.Raw.get_data>`_ method. We pass this argument to the
        :code:`Raw.get_data` method. By default
        :code:`reject_by_annotation=None` retrieves all time points. Use
        :code:`reject_by_annotation="omit"` to remove segments marked as bad.
    sampling_frequency : float, optional
        Sampling frequency of the data in Hz.
    parcellation_file : str, optional
        Path to parcellation file used to source reconstruct the data.
    time_axis_first : bool, optional
        Is the input data of shape (n_samples, n_channels)? Default is
        :code:`True`. If your data is in format (n_channels, n_samples), use
        :code:`time_axis_first=False`.
    load_memmaps : bool, optional
        Should we load the data as memory maps (memmaps)? If :code:`True`, we
        will store the data on disk rather than loading it into memory.
    store_dir : str, optional
        Directory to write memory maps to (if :code:`load_memmaps=True`).
        Each Data instance writes to its own unique sub-directory of
        :code:`store_dir`, which is deleted automatically when the script
        ends (or when :code:`delete_dir` is called). Default is
        :code:`./tmp`.
    session_labels : list of SessionLabels, optional
        Extra session labels.
    extra_channels : dict, optional
        Extra channels to add to the data. The keys are the channel names and
        the values are the channel data.
    n_jobs : int, optional
        Number of processes to load the data in parallel.
        Default is 1, which loads data in serial.
    mask_file : str, optional
        Deprecated and ignored: brain maps are plotted on the voxel grid of
        the parcellation file. Will be removed in v3.8.0.
    """

    def __init__(
        self,
        inputs: Union[
            List[Union[str, pathlib.Path]],
            str,
            pathlib.Path,
            np.ndarray,
            List[np.ndarray],
        ],
        data_field: str = "X",
        picks: Optional[Union[str, List[str]]] = None,
        reject_by_annotation: Optional[str] = None,
        sampling_frequency: Optional[float] = None,
        parcellation_file: Optional[str] = None,
        time_axis_first: bool = True,
        load_memmaps: bool = False,
        store_dir: str = "tmp",
        session_labels: Optional[List["SessionLabels"]] = None,
        extra_channels: Optional[Dict] = None,
        n_jobs: int = 1,
        mask_file: Optional[str] = None,
    ) -> None:
        misc.warn_mask_file_unused(mask_file)
        # Unique identifier for this Data instance. We use a UUID rather
        # than id(self) because id(self) is a memory address, which can
        # be identical across different processes running the same script
        # (e.g. cluster array jobs sharing a store_dir)
        self._identifier = uuid.uuid4().hex
        self.data_field = data_field
        self.picks = picks
        self.reject_by_annotation = reject_by_annotation
        self.original_sampling_frequency = sampling_frequency
        self.sampling_frequency = sampling_frequency
        self.parcellation_file = parcellation_file
        self.mask_file = mask_file
        self.time_axis_first = time_axis_first
        self.load_memmaps = load_memmaps
        self.n_jobs = n_jobs

        # Validate inputs
        self.inputs = rw.validate_inputs(inputs)

        if len(self.inputs) == 0:
            raise ValueError("No valid inputs were passed.")

        # Directory to store memory maps created by this class
        # (created when rw.load_data is called).
        # Each Data instance gets its own sub-directory
        # (named by its identifier).
        self.store_dir = pathlib.Path(store_dir) / str(self._identifier)

        # Delete store_dir automatically when the interpreter exits.
        # Note, we must not delete on garbage collection: TensorFlow
        # datasets open files in store_dir lazily (at iteration time)
        # and can outlive this object
        self._finalizer = partial(
            misc.delete_dir, self.store_dir, remove_empty_parent=True
        )
        atexit.register(self._finalizer)

        # Load and validate the raw data
        self.raw_data_arrays, self.raw_data_filenames = self.load_raw_data()
        self.validate_data()

        self.n_raw_data_channels = self.raw_data_arrays[0].shape[-1]

        # Get data preparation attributes if there's a pickle file in the
        # input directory
        if not isinstance(inputs, list) and not isinstance(inputs, np.ndarray):
            self.load_preparation(inputs)

        # Store raw data in the arrays attribute
        self.arrays = self.raw_data_arrays

        # Create filenames for prepared data memmaps
        prepared_data_pattern = "prepared_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(self.n_sessions)), identifier=self._identifier
        )
        self.prepared_data_filenames = [
            str(self.store_dir / prepared_data_pattern.format(i=i))
            for i in range(self.n_sessions)
        ]

        # Arrays to keep when making TensorFlow Datasets
        self.keep = list(range(self.n_sessions))

        # Extra session labels
        if session_labels is None:
            self.session_labels = []
        else:
            self.session_labels = session_labels

        # Extra channels
        if extra_channels is None:
            self.extra_channels = {}
        else:
            self.extra_channels = extra_channels

    def __iter__(self):
        return iter(self.arrays)

    def __getitem__(self, item: int) -> np.ndarray:
        return self.arrays[item]

    def __str__(self) -> str:
        info = [
            f"{self.__class__.__name__}",
            f"id: {self._identifier}",
            f"n_sessions: {self.n_sessions}",
            f"n_samples: {self.n_samples}",
            f"n_channels: {self.n_channels}",
        ]
        return "\n ".join(info)

    @property
    def raw_data(self) -> List[np.ndarray]:
        """Return raw data as a list of arrays."""
        return self.raw_data_arrays

    @property
    def n_channels(self) -> int:
        """Number of channels in the data files."""
        return self.arrays[0].shape[-1]

    @property
    def n_samples(self) -> int:
        """Number of samples across all arrays."""
        return sum([array.shape[-2] for array in self.arrays])

    @property
    def n_sessions(self) -> int:
        """Number of arrays."""
        return len(self.arrays)

    def input_shapes(self, sequence_length: int) -> Dict:
        """Get the input shapes for the model.

        Parameters
        ----------
        sequence_length : int
            Length of the segment of data to feed into the model.

        Returns
        -------
        shapes : dict
            Dictionary of input shapes.
        """
        return dtf.element_shapes(
            sequence_length, self.n_channels, self.session_labels, self.extra_channels
        )

    @contextmanager
    def set_keep(self, keep: Union[int, List[int]]):
        """Context manager to temporarily set the kept arrays.

        Parameters
        ----------
        keep : int or list of int
            Indices to keep in the Data.arrays list.
        """
        # Store the current kept arrays
        current_keep = self.keep
        try:
            if isinstance(keep, int):
                keep = [keep]
            if not isinstance(keep, list):
                raise ValueError("keep must be a list of indices or a single index.")

            # Set the new kept arrays
            self.keep = keep
            yield
        finally:
            self.keep = current_keep

    def set_sampling_frequency(self, sampling_frequency: float) -> None:
        """Sets the :code:`sampling_frequency` attribute.

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency in Hz.
        """
        self.original_sampling_frequency = sampling_frequency
        self.sampling_frequency = sampling_frequency

    def time_series(
        self, prepared: bool = True, concatenate: bool = False
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Time series data for all arrays.

        Parameters
        ----------
        prepared : bool, optional
            Should we return the latest data after we have prepared it or
            the original data we loaded into the Data object?
        concatenate : bool, optional
            Should we return the time series for each array concatenated?

        Returns
        -------
        ts : list or np.ndarray
            Time series data for each array.
        """
        # What data should we return?
        if prepared:
            arrays = self.arrays
        else:
            arrays = self.raw_data_arrays

        # Should we return one long time series?
        if concatenate or self.n_sessions == 1:
            return np.concatenate(arrays)
        else:
            return arrays

    def load_raw_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """Import data into a list of memory maps.

        Returns
        -------
        memmaps : list of np.memmap
            List of memory maps.
        raw_data_filenames : list of str
            List of paths to the raw data memmaps.
        """
        raw_data_pattern = "raw_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(len(self.inputs))), identifier=self._identifier
        )
        raw_data_filenames = [
            str(self.store_dir / raw_data_pattern.format(i=i))
            for i in range(len(self.inputs))
        ]
        # self.raw_data_filenames is not used if self.inputs is a list of
        # strings, where the strings are paths to .npy files

        # Function to save a single memory map
        def _make_memmap(raw_data, mmap_location):
            if not self.load_memmaps:  # do not load into the memory maps
                mmap_location = None
            raw_data_mmap = rw.load_data(
                raw_data,
                self.data_field,
                self.picks,
                self.reject_by_annotation,
                mmap_location,
                mmap_mode="r",
            )
            if not self.time_axis_first:
                raw_data_mmap = raw_data_mmap.T
            return raw_data_mmap

        # Load data
        memmaps = pqdm(
            array=zip(self.inputs, raw_data_filenames),
            function=_make_memmap,
            n_jobs=self.n_jobs,
            desc="Loading files",
            argument_type="args",
            total=len(self.inputs),
        )
        _raise_exceptions(memmaps, "file")

        return memmaps, raw_data_filenames

    def validate_data(self) -> None:
        """Validate data files."""
        n_channels = [array.shape[-1] for array in self.raw_data_arrays]
        if not np.equal(n_channels, n_channels[0]).all():
            raise ValueError("All inputs should have the same number of channels.")

    def validate_extra_channels(
        self, data: List[np.ndarray], extra_channels: Dict
    ) -> Dict:
        """Validate extra channels."""
        n_sessions = len(data)

        # Validate each channel
        for channel_name, channel in extra_channels.items():
            if not isinstance(channel_name, str):
                raise TypeError("Channel name must be a string.")

            if isinstance(channel, np.ndarray):
                channel = [channel]
                extra_channels[channel_name] = channel

            if not isinstance(channel, list):
                raise TypeError(f"Extra channel {channel_name} must be a list.")

            if len(channel) != n_sessions:
                raise ValueError(
                    "Extra channel must have the same number of sessions as the data."
                )

            for i, channel_data in enumerate(channel):
                if channel_data.ndim not in [1, 2]:
                    raise ValueError(
                        f"Extra channel {channel_name} in session {i} must be 1D or 2D."
                    )
                if data[i].shape[0] != channel_data.shape[0]:
                    raise ValueError(
                        f"Extra channel {channel_name} have different number of samples "
                        f"than the data in session {i}."
                    )

                extra_channels[channel_name][i] = channel_data.astype(np.float32)

        return extra_channels

    def _validate_batching(
        self,
        index: np.ndarray,
        batch_size: int,
        drop_last_batch: bool = False,
        concatenate: bool = True,
    ) -> None:
        """Check the data can be split into at least one batch.

        Parameters
        ----------
        index : np.ndarray
            Sequence index from :func:`osl_dynamics.data.tf.sequence_index`.
        batch_size : int
            Number sequences in each mini-batch which is used to train the
            model.
        drop_last_batch : bool, optional
            Should we drop the last batch if it is smaller than the batch size?
            Defaults to False.
        concatenate : bool, optional
            Should we concatenate the datasets for each array?
            Defaults to True.
        """

        def _n_batches(n_sequences):
            if drop_last_batch:
                return n_sequences // batch_size
            return math.ceil(n_sequences / batch_size)

        if concatenate:
            n_batches = _n_batches(len(index))
        else:
            n_batches = sum(_n_batches(np.sum(index[:, 0] == i)) for i in self.keep)

        if n_batches < 1:
            raise ValueError(
                "Number of batches must be greater than or equal to 1. "
                "Please adjust your sequence length and batch size."
            )

    def select(
        self,
        channels: Optional[Union[int, List[int]]] = None,
        sessions: Optional[Union[int, List[int]]] = None,
        use_raw: bool = False,
    ) -> "Data":
        """Select channels.

        This is an in-place operation.

        Parameters
        ----------
        channels : int or list of int, optional
            Channel indices to keep. If None, all channels are retained.
        sessions : int or list of int, optional
            Session indices to keep. If None, all sessions are retained.
        use_raw : bool, optional
            Should we select channel from the original 'raw' data that
            we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if channels is None:
            # Keep all channels
            if use_raw:
                n_channels = self.raw_data_arrays[0].shape[-1]
            else:
                n_channels = self.arrays[0].shape[-1]
            channels = range(n_channels)

        if sessions is None:
            # Keep all sessions
            if use_raw:
                n_sessions = len(self.raw_data_arrays)
            else:
                n_sessions = len(self.arrays)
            sessions = range(n_sessions)

        if isinstance(channels, int):
            channels = [channels]

        if isinstance(sessions, int):
            sessions = [sessions]

        if isinstance(channels, range):
            channels = list(channels)

        if isinstance(sessions, range):
            sessions = list(sessions)

        if not isinstance(channels, list):
            raise TypeError("channels must be an int or list of int.")

        if not isinstance(sessions, list):
            raise TypeError("sessions must be an int or list of int.")

        # What data should we use?
        arrays = self.raw_data_arrays if use_raw else self.arrays

        # Select channels
        new_arrays = []
        for i in tqdm(sessions, desc="Selecting channels/sessions"):
            array = arrays[i][:, channels]
            if self.load_memmaps:
                array = misc.array_to_memmap(self.prepared_data_filenames[i], array)
            new_arrays.append(array)
        self.arrays = new_arrays

        return self

    def filter(
        self,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
        order: int = 5,
        use_raw: bool = False,
    ) -> "Data":
        """Filter the data.

        This is an in-place operation.

        Parameters
        ----------
        low_freq : float, optional
            Frequency in Hz for a high pass filter. If :code:`None`, no high
            pass filtering is applied.
        high_freq : float, optional
            Frequency in Hz for a low pass filter. If :code:`None`, no low pass
            filtering is applied.
        order : int, optional
            Order for a butterworth filter.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if low_freq is None and high_freq is None:
            _logger.warning("No filtering applied.")
            return

        if self.sampling_frequency is None:
            raise ValueError(
                "Data.sampling_frequency must be set if we are filtering the "
                "data. Use Data.set_sampling_frequency() or pass "
                "Data(..., sampling_frequency=...) when creating the Data "
                "object."
            )

        self.low_freq = low_freq
        self.high_freq = high_freq

        # Function to apply filtering to a single array
        def _apply(array, prepared_data_file):
            array = processing.temporal_filter(
                array, low_freq, high_freq, self.sampling_frequency, order
            )
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Prepare the data in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="Filtering",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        _raise_exceptions(self.arrays)

        return self

    def downsample(self, freq: float, use_raw: bool = False) -> "Data":
        """Downsample the data.

        This is an in-place operation.

        Parameters
        ----------
        freq : float
            Frequency in Hz to downsample to.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if self.sampling_frequency is None:
            raise ValueError(
                "Data.sampling_frequency must be set if we are filtering the "
                "data. Use Data.set_sampling_frequency() or pass "
                "Data(..., sampling_frequency=...) when creating the Data "
                "object."
            )

        if use_raw:
            sampling_frequency = self.original_sampling_frequency
        else:
            sampling_frequency = self.sampling_frequency

        # Function to apply downsampling to a single array
        def _apply(array, prepared_data_file):
            array = processing.downsample(array, freq, sampling_frequency)
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Prepare the data in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="Downsampling",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        _raise_exceptions(self.arrays)

        # Update sampling_frequency attributes
        self.sampling_frequency = freq

        return self

    def pca(
        self,
        n_pca_components: Optional[int] = None,
        pca_components: Optional[np.ndarray] = None,
        whiten: bool = False,
        use_raw: bool = False,
    ) -> "Data":
        """Principal component analysis (PCA).

        This function will first standardize the data then perform PCA.
        This is an in-place operation.

        Parameters
        ----------
        n_pca_components : int, optional
            Number of PCA components to keep. If :code:`None`, then
            :code:`pca_components` should be passed.
        pca_components : np.ndarray, optional
            PCA components to apply if they have already been calculated.
            If :code:`None`, then :code:`n_pca_components` should be passed.
        whiten : bool, optional
            Should we whiten the PCA'ed data?
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if (n_pca_components is None and pca_components is None) or (
            n_pca_components is not None and pca_components is not None
        ):
            raise ValueError("Please pass either n_pca_components or pca_components.")

        if pca_components is not None and not isinstance(pca_components, np.ndarray):
            raise TypeError("pca_components must be a numpy array.")

        self.n_pca_components = n_pca_components
        self.pca_components = pca_components
        self.whiten = whiten

        # What data should we apply PCA to?
        arrays = self.raw_data_arrays if use_raw else self.arrays

        # Calculate PCA
        if n_pca_components is not None:
            # Calculate covariance of the data
            n_channels = arrays[0].shape[-1]
            covariance = np.zeros((n_channels, n_channels), dtype=np.float64)
            for array in tqdm(arrays, desc="Calculating PCA components"):
                std_data = processing.standardize(array)
                covariance += std_data.T @ std_data

            # Compute covariance eigenvalues to obtain PCA components
            s, u = misc.top_eig(covariance, n_pca_components)
            idx = np.argsort(s)[::-1]
            s = s[idx]
            u = u[:, idx]
            self.explained_variance = np.sum(s) / np.trace(covariance)
            _logger.info(f"Explained variance: {100 * self.explained_variance:.1f}%")
            s = s.astype(np.float32)
            u = u.astype(np.float32)
            if whiten:
                u = u @ np.diag(1.0 / np.sqrt(s))
            self.pca_components = u

        # Function to apply PCA to a single array
        def _apply(array, prepared_data_file):
            array = processing.standardize(array)
            array = array @ self.pca_components
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Apply PCA in parallel
        args = zip(arrays, self.prepared_data_filenames)
        with threadpool_limits(limits=1 if self.n_jobs > 1 else None):
            self.arrays = pqdm(
                args,
                function=_apply,
                desc="PCA",
                n_jobs=self.n_jobs,
                argument_type="args",
                total=self.n_sessions,
            )
        _raise_exceptions(self.arrays)

        return self

    def tde(self, n_embeddings: int, use_raw: bool = False) -> "Data":
        """Time-delay embedding (TDE).

        This is an in-place operation.

        See the :doc:`Time-Delay Embedding tutorial
        </tutorials_build/1-4_data_time_delay_embedding>` for advice on
        choosing :code:`n_embeddings`.

        Parameters
        ----------
        n_embeddings : int
            Number of data points to embed the data.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        self.n_embeddings = n_embeddings
        self.n_te_channels = self.n_raw_data_channels * n_embeddings

        # Function to apply TDE to a single array
        def _apply(array, prepared_data_file):
            array = processing.time_embed(array, n_embeddings)
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Apply TDE in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="TDE",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        _raise_exceptions(self.arrays)

        return self

    def tde_pca(
        self,
        n_embeddings: int,
        n_pca_components: Optional[int] = None,
        pca_components: Optional[np.ndarray] = None,
        whiten: bool = False,
        use_raw: bool = False,
    ) -> "Data":
        """Time-delay embedding (TDE) and principal component analysis (PCA).

        This function will first standardize the data, then perform TDE then
        PCA. It is useful to do both operations in a single methods because
        it avoids having to save the time-embedded data. This is an in-place
        operation.

        How to choose the parameters: :code:`n_embeddings` should be chosen
        based on the sampling frequency. We recommend
        :code:`n_embeddings=15` for 250 Hz data and :code:`n_embeddings=7`
        for 100 Hz data (both correspond to an embedding window of roughly
        60-70 ms). :code:`n_pca_components` should be at least twice the
        number of original channels to retain the high-frequency content of
        the data. We advise plotting the power spectrum of the prepared data
        for different parameters to check the frequency range you are
        interested in is retained, see the :doc:`Time-Delay Embedding
        tutorial </tutorials_build/1-4_data_time_delay_embedding>` for
        example code and the :doc:`FAQ </faq>` for further discussion.

        Parameters
        ----------
        n_embeddings : int
            Number of data points to embed the data.
        n_pca_components : int, optional
            Number of PCA components to keep. If :code:`None`, then
            :code:`pca_components` should be passed.
        pca_components : np.ndarray, optional
            PCA components to apply if they have already been calculated.
            If :code:`None`, then :code:`n_pca_components` should be passed.
        whiten : bool, optional
            Should we whiten the PCA'ed data?
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if (n_pca_components is None and pca_components is None) or (
            n_pca_components is not None and pca_components is not None
        ):
            raise ValueError("Please pass either n_pca_components or pca_components.")

        if pca_components is not None and not isinstance(pca_components, np.ndarray):
            raise TypeError("pca_components must be a numpy array.")

        self.n_embeddings = n_embeddings
        self.n_pca_components = n_pca_components
        self.pca_components = pca_components
        self.whiten = whiten

        # What data should we use?
        arrays = self.raw_data_arrays if use_raw else self.arrays
        self.n_te_channels = arrays[0].shape[-1] * n_embeddings

        # Calculate PCA on TDE data
        if n_pca_components is not None:
            # Calculate covariance of the data
            covariance = np.zeros(
                (self.n_te_channels, self.n_te_channels), dtype=np.float64
            )
            for array in tqdm(arrays, desc="Calculating PCA components"):
                std_data = processing.standardize(array)
                te_std_data = processing.time_embed(std_data, n_embeddings)
                covariance += te_std_data.T @ te_std_data

            # Compute covariance eigenvalues to obtain PCA components
            s, u = misc.top_eig(covariance, n_pca_components)
            idx = np.argsort(s)[::-1]
            s = s[idx]
            u = u[:, idx]
            self.explained_variance = np.sum(s) / np.trace(covariance)
            _logger.info(f"Explained variance: {100 * self.explained_variance:.1f}%")
            s = s.astype(np.float32)
            u = u.astype(np.float32)
            if whiten:
                u = u @ np.diag(1.0 / np.sqrt(s))
            self.pca_components = u

        # Function to apply TDE-PCA to a single array
        def _apply(array, prepared_data_file):
            array = processing.standardize(array)
            array = processing.time_embed(array, n_embeddings)
            array = array @ self.pca_components
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Apply TDE and PCA in parallel
        args = zip(arrays, self.prepared_data_filenames)
        with threadpool_limits(limits=1 if self.n_jobs > 1 else None):
            self.arrays = pqdm(
                args,
                function=_apply,
                desc="TDE-PCA",
                n_jobs=self.n_jobs,
                argument_type="args",
                total=self.n_sessions,
            )
        _raise_exceptions(self.arrays)

        return self

    def amplitude_envelope(self, use_raw: bool = False) -> "Data":
        """Calculate the amplitude envelope.

        This is an in-place operation.

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?
        """

        # Function to calculate amplitude envelope for a single array
        def _apply(array, prepared_data_file):
            array = processing.amplitude_envelope(array)
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Prepare the data in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="Amplitude envelope",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        _raise_exceptions(self.arrays)

        return self

    def moving_average(self, n_window: int, use_raw: bool = False) -> "Data":
        """Calculate a moving average.

        This is an in-place operation.

        Parameters
        ----------
        n_window : int
            Number of data points in the sliding window. Must be odd.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        self.n_window = n_window

        # Function to apply sliding window to a single array
        def _apply(array, prepared_data_file):
            array = processing.moving_average(array, n_window)
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Prepare the data in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="Sliding window",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        _raise_exceptions(self.arrays)

        return self

    def standardize(self, use_raw: bool = False) -> "Data":
        """Standardize (z-score) the data.

        This is an in-place operation.

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?
        """

        # Function to apply standardisation to a single array
        def _apply(array, prepared_data_file):
            array = processing.standardize(array, create_copy=False)
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Apply standardisation to each array in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            array=args,
            function=_apply,
            desc="Standardize",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        _raise_exceptions(self.arrays)

        return self

    def align_channel_signs(
        self,
        template_data: Optional[Union[np.ndarray, str]] = None,
        template_cov: Optional[Union[np.ndarray, str]] = None,
        n_random_starts: int = 3,
        n_embeddings: int = 1,
        standardize: bool = True,
        use_raw: bool = False,
    ) -> "Data":
        """Align the sign of each channel across sessions.

        If no template data/covariance is passed, we use the median session.

        Parameters
        ----------
        template_data : np.ndarray or str, optional
            Data to align the sign of channels to.
            If :code:`str`, the file will be read in the same way as the
            inputs to the Data object.
        template_cov : np.ndarray or str, optional
            Covariance to align the sign of channels. This must be the
            covariance of the time-delay embedded data.
            If :code:`str`, must be the path to a :code:`.npy` file.
        n_random_starts : int, optional
            Number of random restarts of the search (in addition to the
            deterministic spectral and all-flips-positive starts). The restarts
            use a fixed internal seed, so the search is deterministic.
        n_embeddings : int, optional
            We may want to compare the covariance of time-delay embedded data
            when aligning the signs. This is the number of embeddings. The
            returned data is not time-delay embedded.
        standardize : bool, optional
            Should we standardize the data before comparing across sessions?
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        if template_data is not None and template_cov is not None:
            raise ValueError(
                "Only pass one of the arguments template_data or template_cov, "
                "not both."
            )

        if isinstance(template_data, str):
            template_data = rw.load_data(
                template_data,
                self.data_field,
                self.picks,
                self.reject_by_annotation,
            )
            if not self.time_axis_first:
                template_data = template_data.T

        if isinstance(template_cov, str):
            template_cov = np.load(template_cov)

        # Helper functions
        def _calc_cov(array):
            return sign_flipping.calc_cov(array, n_embeddings, standardize)

        def _calc_metrics(covs):
            metric = np.zeros([self.n_sessions, self.n_sessions])
            for i in tqdm(range(self.n_sessions), desc="Comparing sessions"):
                for j in range(i + 1, self.n_sessions):
                    metric[i, j] = sign_flipping.calc_corr(
                        covs[i], covs[j], n_embeddings, mode="abs"
                    )
                    metric[j, i] = metric[i, j]
            return metric

        def _find_and_apply_flips(cov, tcov, array, ind):
            flips, best_metric = sign_flipping.find_flips(
                cov,
                tcov,
                self.n_channels,
                n_embeddings,
                n_random_starts=n_random_starts,
            )
            _logger.debug(
                f"Session {ind}, best correlation with template: {best_metric:.3f}"
            )
            return array * flips[np.newaxis, ...].astype(np.float32), best_metric

        # What data do we use?
        arrays = self.raw_data_arrays if use_raw else self.arrays

        # Calculate covariance of each session
        with threadpool_limits(limits=1 if self.n_jobs > 1 else None):
            covs = pqdm(
                array=zip(arrays),
                function=_calc_cov,
                desc="Calculating covariances",
                n_jobs=self.n_jobs,
                argument_type="args",
                total=self.n_sessions,
            )
        _raise_exceptions(covs)

        # Calculate/get template covariances
        if template_cov is None and template_data is None:
            metrics = _calc_metrics(covs)
            metrics_sum = np.sum(metrics, axis=1)
            argmedian = np.argsort(metrics_sum)[len(metrics_sum) // 2]
            _logger.info(f"Using session {argmedian} as template")
            template_cov = covs[argmedian]

        if template_data is not None:
            _logger.info(f"Using template data to calculate template covariance")
            template_cov = _calc_cov(template_data)

        # Perform the sign flipping
        tcovs = [template_cov] * self.n_sessions
        indices = range(self.n_sessions)
        with threadpool_limits(limits=1 if self.n_jobs > 1 else None):
            results = pqdm(
                array=zip(covs, tcovs, arrays, indices),
                function=_find_and_apply_flips,
                desc="Sign flipping",
                n_jobs=self.n_jobs,
                argument_type="args",
                total=self.n_sessions,
            )
        _raise_exceptions(results)

        self.arrays = [array for array, _ in results]
        corrs = [metric for _, metric in results]
        _logger.info(
            f"Mean correlation with template: {np.mean(corrs):.3f} "
            f"(min: {np.min(corrs):.3f})"
        )

        return self

    def prepare(self, methods: Dict) -> "Data":
        """Prepare data.

        Wrapper for calling a series of data preparation methods. Any method
        in Data can be called. Note that if the same method is called multiple
        times, the method name should be appended with an underscore and a
        number, e.g. :code:`standardize_1` and :code:`standardize_2`.

        Parameters
        ----------
        methods : dict
            Each key is the name of a method to call. Each value is a
            :code:`dict` containing keyword arguments to pass to the method.

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.

        Examples
        --------
        TDE-PCA data preparation::

            methods = {
                "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
                "standardize": {},
            }
            data.prepare(methods)

        Amplitude envelope data preparation::

            methods = {
                "filter": {"low_freq": 1, "high_freq": 45},
                "amplitude_envelope": {},
                "moving_average": {"n_window": 5},
                "standardize": {},
            }
            data.prepare(methods)
        """
        # Pattern for identifying the method name from "method-name_num"
        pattern = re.compile(r"^(\w+?)(_\d+)?$")

        for method_name, kwargs in methods.items():
            # Remove the "_num" part from the dict key
            method_name = pattern.search(method_name).groups()[0]

            # Apply method
            method = getattr(self, method_name)
            method(**kwargs)

        return self

    def trim_time_series(
        self,
        sequence_length: Optional[int] = None,
        n_embeddings: Optional[int] = None,
        n_window: Optional[int] = None,
        prepared: bool = True,
        concatenate: bool = False,
        verbose: bool = False,
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Trims the data time series.

        Removes the data points that are lost when the data is prepared,
        i.e. due to time embedding and separating into sequences, but does not
        perform time embedding or batching into sequences on the time series.

        Parameters
        ----------
        sequence_length : int, optional
            Length of the segment of data to feed into the model.
            Can be pass to trim the time points that are lost when separating
            into sequences.
        n_embeddings : int, optional
            Number of data points used to embed the data. If :code:`None`,
            then we use :code:`Data.n_embeddings` (if it exists).
        n_window : int, optional
            Number of data points the sliding window applied to the data.
            If :code:`None`, then we use :code:`Data.n_window` (if it exists).
        prepared : bool, optional
            Should we return the prepared data? If not we return the raw data.
        concatenate : bool, optional
            Should we concatenate the data for each array?
        verbose : bool, optional
            Should we print the number of data points we're removing?

        Returns
        -------
        list of np.ndarray
            Trimmed time series for each array.
        """
        # How many time points from the start/end of the time series should
        # we remove?
        n_remove = 0
        if n_embeddings is None:
            if hasattr(self, "n_embeddings"):
                n_remove += self.n_embeddings // 2
        else:
            n_remove += n_embeddings // 2
        if n_window is None:
            if hasattr(self, "n_window"):
                n_remove += self.n_window // 2
        else:
            n_remove += n_window // 2
        if verbose:
            _logger.info(
                f"Removing {n_remove} data points from the start and end"
                " of each array due to time embedding/sliding window."
            )

        # What data should we trim?
        if prepared:
            arrays = self.arrays
        else:
            arrays = self.raw_data_arrays

        trimmed_time_series = []
        for i, array in enumerate(arrays):
            # Remove data points lost to time embedding or sliding window
            if n_remove != 0:
                array = array[n_remove:-n_remove]

            # Remove data points lost to separating into sequences
            if sequence_length is not None:
                n_sequences = array.shape[0] // sequence_length
                n_keep = n_sequences * sequence_length
                if verbose:
                    _logger.info(
                        f"Removing {array.shape[0] - n_keep} data points "
                        f"from the end of array {i} due to sequencing."
                    )
                array = array[:n_keep]

            trimmed_time_series.append(array)

        if concatenate or len(trimmed_time_series) == 1:
            trimmed_time_series = np.concatenate(trimmed_time_series)

        return trimmed_time_series

    def dataset(
        self,
        sequence_length: int,
        batch_size: int,
        shuffle: bool = True,
        validation_split: Optional[float] = None,
        concatenate: bool = True,
        step_size: Optional[int] = None,
        drop_last_batch: bool = False,
        validation_split_seed: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Create a Tensorflow Dataset for training or evaluation.

        Sequences are cut out of the arrays on the fly, so the arrays are
        not copied. If the data was loaded with :code:`load_memmaps=True`
        only the sequences in flight are held in memory. Shuffling is a
        perfect shuffle of the sequences, redone every epoch.

        Parameters
        ----------
        sequence_length : int
            Length of the segment of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the
            model.
        shuffle : bool, optional
            Should we shuffle the sequences?
        validation_split : float, optional
            Ratio to split the dataset into a training and validation set.
        concatenate : bool, optional
            Should we concatenate the datasets for each array?
        step_size : int, optional
            Number of samples to slide the sequence across the dataset.
            Default is no overlap.
        drop_last_batch : bool, optional
            Should we drop the last batch if it is smaller than the batch size?
        validation_split_seed : int, optional
            Seed for the training/validation split, pass to make the split
            reproducible. Only used if :code:`validation_split` is passed.
        seed : int, optional
            Seed for shuffling.

        Returns
        -------
        dataset : tf.data.Dataset or tuple of tf.data.Dataset
            Dataset for training or evaluating the model along with the
            validation set if :code:`validation_split` was passed.
            The number of batches is known (:code:`dataset.cardinality()`)
            and :code:`dataset.osld_metadata` holds the sequence length,
            step size, number of sequences and number of channels.
        """
        step_size = step_size or sequence_length
        if validation_split is not None and step_size != sequence_length:
            raise ValueError(
                "validation_split cannot be used with overlapping sequences "
                f"(step_size={step_size} != sequence_length={sequence_length}). "
                "The split is performed on non-overlapping sequences, so "
                "overlapping windows would leak data between the training "
                "and validation sets."
            )

        # Validate extra channels
        self.extra_channels = self.validate_extra_channels(
            self.arrays, self.extra_channels
        )

        # Index of every sequence (session, start sample)
        index = dtf.sequence_index(
            [array.shape[0] for array in self.arrays],
            sequence_length,
            step_size,
            keep=self.keep,
        )
        self._validate_batching(index, batch_size, drop_last_batch, concatenate)

        # Function to cut a sequence out of the arrays
        slice_fn = dtf.make_slice_fn(
            self.arrays,
            sequence_length,
            self.session_labels,
            self.extra_channels,
        )

        def _create_dataset(index, shuffle=shuffle):
            kwargs = dict(
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last_batch=drop_last_batch,
                seed=seed,
            )
            if concatenate:
                dataset = dtf.dataset_from_index(index, slice_fn, **kwargs)
                dataset.osld_metadata = {
                    "sequence_length": sequence_length,
                    "step_size": step_size,
                    "n_sequences": len(index),
                    "n_channels": self.n_channels,
                }
                return dataset

            # Otherwise create a dataset for each session separately
            datasets = []
            for i in self.keep:
                datasets.append(
                    dtf.dataset_from_index(index[index[:, 0] == i], slice_fn, **kwargs)
                )
            return datasets

        if validation_split is not None:
            train_index, val_index = dtf.split_index(
                index, validation_split, validation_split_seed
            )
            return _create_dataset(train_index), _create_dataset(
                val_index, shuffle=False
            )

        return _create_dataset(index)

    def _fingerprint(self) -> str:
        """Hash identifying the input data and how it was prepared."""
        h = hashlib.sha256()

        def _update(obj):
            if isinstance(obj, dict):
                for key in sorted(obj, key=str):
                    _update(key)
                    _update(obj[key])
            elif isinstance(obj, (list, tuple)):
                for value in obj:
                    _update(value)
            elif isinstance(obj, np.ndarray):
                h.update(str((obj.dtype, obj.shape)).encode())
                h.update(np.ascontiguousarray(obj).tobytes())
            else:
                h.update(repr(obj).encode())

        _update(self.inputs)
        _update([array.shape[0] for array in self.arrays])
        _update(self._preparation_settings())
        _update(self.session_labels)
        _update(self.extra_channels)
        return h.hexdigest()

    def _tfrecords_up_to_date(self, tfrecord_dir: str, config: Dict) -> bool:
        """Can an existing TFRecord dataset be reused?"""
        if not os.path.exists(os.path.join(tfrecord_dir, dtf.TFRECORD_CONFIG_FILE)):
            return False

        existing = dtf.load_tfrecord_config(tfrecord_dir)
        for key, value in config.items():
            if existing.get(key) != value:
                _logger.info(
                    f"{key} has changed. Rewriting the TFRecord dataset "
                    f"in {tfrecord_dir}."
                )
                return False

        splits = ["train"] if config["validation_split"] is None else ["train", "val"]
        for i in range(self.n_sessions):
            for split in splits:
                if not os.path.exists(
                    dtf.tfrecord_filename(tfrecord_dir, split, i, self.n_sessions)
                ):
                    _logger.info(
                        f"TFRecord file missing. Rewriting the TFRecord dataset "
                        f"in {tfrecord_dir}."
                    )
                    return False

        return True

    def save_tfrecord_dataset(
        self,
        tfrecord_dir: str,
        sequence_length: int,
        step_size: Optional[int] = None,
        validation_split: Optional[float] = None,
        validation_split_seed: Optional[int] = None,
        overwrite: bool = False,
    ) -> None:
        """Save the data as a TFRecord dataset.

        Writes one file per session (two if :code:`validation_split` is
        passed) and a :code:`tfrecord_config.json` describing them. Load
        the dataset with :func:`osl_dynamics.data.load_tfrecord_dataset`.

        If :code:`tfrecord_dir` already holds a dataset written from the
        same data with the same settings it is reused, otherwise it is
        rewritten.

        Parameters
        ----------
        tfrecord_dir : str
            Directory to write the TFRecord dataset to. Must not be inside
            :code:`store_dir`, which is deleted when the script ends.
        sequence_length : int
            Length of the segment of data to feed into the model.
        step_size : int, optional
            Number of samples to slide the sequence across the dataset.
            Default is no overlap.
        validation_split : float, optional
            Ratio to split the dataset into a training and validation set.
        validation_split_seed : int, optional
            Seed for the training/validation split, pass to make the split
            reproducible. Only used if :code:`validation_split` is passed.
        overwrite : bool, optional
            Rewrite the dataset even if the existing one is up to date.
        """
        tfrecord_dir = str(tfrecord_dir)
        resolved = pathlib.Path(tfrecord_dir).resolve()
        store_dir = self.store_dir.resolve()
        if resolved == store_dir or store_dir in resolved.parents:
            raise ValueError(
                f"tfrecord_dir={tfrecord_dir} is inside store_dir, which is "
                "deleted when the script ends. Pass a directory outside "
                "store_dir."
            )

        sequence_length = int(sequence_length)
        step_size = int(step_size or sequence_length)
        if validation_split is not None and step_size != sequence_length:
            raise ValueError(
                "validation_split cannot be used with overlapping sequences "
                f"(step_size={step_size} != sequence_length={sequence_length}). "
                "The split is performed on non-overlapping sequences, so "
                "overlapping windows would leak data between the training "
                "and validation sets."
            )

        # Validate extra channels
        self.extra_channels = self.validate_extra_channels(
            self.arrays, self.extra_channels
        )

        # Index of every sequence (session, start sample)
        index = dtf.sequence_index(
            [array.shape[0] for array in self.arrays], sequence_length, step_size
        )
        if len(index) == 0:
            raise ValueError(
                f"No session has {sequence_length} samples. "
                "Please reduce the sequence_length."
            )

        if validation_split is not None:
            train_index, val_index = dtf.split_index(
                index, validation_split, validation_split_seed
            )
        else:
            train_index, val_index = index, None

        def _per_session(index):
            if index is None:
                return None
            return [int(np.sum(index[:, 0] == i)) for i in range(self.n_sessions)]

        config = {
            "format_version": dtf.TFRECORD_FORMAT_VERSION,
            "sequence_length": sequence_length,
            "step_size": step_size,
            "n_channels": self.n_channels,
            "n_sessions": self.n_sessions,
            "input_shapes": self.input_shapes(sequence_length),
            "validation_split": validation_split,
            "validation_split_seed": validation_split_seed,
            "n_sequences": _per_session(train_index),
            "n_val_sequences": _per_session(val_index),
            "fingerprint": self._fingerprint(),
        }

        if not overwrite and self._tfrecords_up_to_date(tfrecord_dir, config):
            _logger.info(f"Reusing the TFRecord dataset in {tfrecord_dir}.")
            return

        # Remove any existing dataset. The config is removed first so a
        # partially written dataset is never mistaken for a complete one
        os.makedirs(tfrecord_dir, exist_ok=True)
        config_path = os.path.join(tfrecord_dir, dtf.TFRECORD_CONFIG_FILE)
        if os.path.exists(config_path):
            os.remove(config_path)
        for old_file in pathlib.Path(tfrecord_dir).glob("*.tfrecord"):
            old_file.unlink()

        # One file per session (and split)
        slice_fn = dtf.make_slice_fn(
            self.arrays, sequence_length, self.session_labels, self.extra_channels
        )
        jobs = []
        for i in range(self.n_sessions):
            jobs.append(
                (
                    dtf.tfrecord_filename(tfrecord_dir, "train", i, self.n_sessions),
                    train_index[train_index[:, 0] == i],
                )
            )
            if val_index is not None:
                jobs.append(
                    (
                        dtf.tfrecord_filename(tfrecord_dir, "val", i, self.n_sessions),
                        val_index[val_index[:, 0] == i],
                    )
                )

        def _save(filepath, index):
            return dtf.save_tfrecord(filepath, index, slice_fn)

        results = pqdm(
            array=jobs,
            function=_save,
            n_jobs=self.n_jobs,
            desc="Creating TFRecord dataset",
            argument_type="args",
            total=len(jobs),
        )
        _raise_exceptions(results, "TFRecord")

        # Range of each channel in the training data
        train_results = results if val_index is None else results[::2]
        amin = np.min([r[0] for r in train_results], axis=0)
        amax = np.max([r[1] for r in train_results], axis=0)
        config["channel_range"] = (amax - amin).tolist()

        with open(config_path, "w") as file:
            json.dump(config, file, indent=2)

    def add_session_labels(
        self, label_name: str, label_values: np.ndarray, label_type: str
    ) -> None:
        """Add session labels as a new channel to the data.

        Parameters
        ----------
        label_name : str
            Name of the new channel.
        label_values : np.ndarray
            Labels for each session.
        label_type : str
            Type of label, either "categorical" or "continuous".
        """
        if len(label_values) != self.n_sessions:
            raise ValueError(
                "label_values must have the same length as the number of sessions."
            )

        self.session_labels.append(SessionLabels(label_name, label_values, label_type))

    def add_extra_channel(
        self, channel_name: str, channel_values: List[np.ndarray]
    ) -> None:
        """Add an extra channel to the data."""
        if len(channel_values) != self.n_sessions:
            raise ValueError(
                "channel_values must have the same length as the number of sessions."
            )
        if channel_name in self.extra_channels:
            raise ValueError(f"Channel name '{channel_name}' already exists.")

        self.extra_channels[channel_name] = channel_values

    def get_session_labels(self) -> List["SessionLabels"]:
        """Get the session labels.

        Returns
        -------
        session_labels : List[SessionLabels]
            List of session labels.
        """
        return self.session_labels

    def recommend_model_config(self) -> None:
        """Recommends arguments for a model config based on the data."""

        # Initial recommendations
        sequence_length = 200
        batch_size = 256
        learning_rate = 0.01
        n_epochs = 20

        # What's the shortest session
        min_n_samples = np.min([x.shape[0] for x in self.arrays])

        # Make sure we have enough samples for one full sequence for
        # each session
        if min_n_samples < 10:
            _logger.warning(
                "Detected session with less than 10 samples, "
                "this means we may have to have very short sequences."
            )
            sequence_length = min_n_samples

        min_n_sequences = min_n_samples // sequence_length

        if min_n_sequences < 1:
            raise ValueError(
                "Not enough samples in at least one session to "
                f"split into sequences of {sequence_length}."
            )

        # How many sequences and batches do we have in total?
        n_sequences = np.sum([x.shape[0] // sequence_length for x in self.arrays])
        n_batches = math.ceil(n_sequences / batch_size)

        # Make sure we have at least 4 batches (updates per epoch)
        if n_batches < 4:
            batch_size = n_sequences // 4

        n_batches = math.ceil(n_sequences / batch_size)

        # We have a really large dataset we can recommend a lower
        # learning rate and less epochs
        if n_batches > 150:
            learning_rate = 0.001
            n_epochs = 15
            print(
                "This is a large dataset. If you run out of memory, load the "
                "data with Data(..., load_memmaps=True) or save it as a "
                "TFRecord dataset with Data.save_tfrecord_dataset."
            )

        # Print info
        print(
            "Recommendation:\n"
            "  config = Config(\n"
            "      ...\n"
            f"      sequence_length={sequence_length},\n"
            f"      batch_size={batch_size},\n"
            f"      learning_rate={learning_rate},\n"
            f"      n_epochs={n_epochs},\n"
            "  )"
        )
        print(f"Total number of sequences: {n_sequences}")
        print(f"Total number of batches/parameter updates: {n_batches}")

    def _preparation_settings(self) -> Dict:
        """Settings and parameters from preparing the data."""
        attributes = list(self.__dict__.keys())
        dont_keep = [
            "_identifier",
            "_finalizer",
            "data_field",
            "picks",
            "reject_by_annotation",
            "original_sampling_frequency",
            "sampling_frequency",
            "parcellation_file",
            "mask_file",
            "time_axis_first",
            "load_memmaps",
            "n_jobs",
            "prepared_data_filenames",
            "inputs",
            "store_dir",
            "raw_data_arrays",
            "raw_data_filenames",
            "n_raw_data_channels",
            "arrays",
            "keep",
            "session_labels",
            "extra_channels",
        ]
        for item in dont_keep:
            if item in attributes:
                attributes.remove(item)
        return {a: getattr(self, a) for a in attributes}

    def save_preparation(self, output_dir: str = ".") -> None:
        """Save a pickle file containing preparation settings.

        Parameters
        ----------
        output_dir : str
            Path to save data files to. Default is the current working
            directory. Created if it does not already exist.
        """
        os.makedirs(output_dir, exist_ok=True)
        preparation = self._preparation_settings()
        pickle.dump(preparation, open(f"{output_dir}/preparation.pkl", "wb"))

    def load_preparation(self, inputs: str) -> None:
        """Loads a pickle file containing preparation settings.

        Parameters
        ----------
        inputs : str
            Path to directory containing the pickle file with preparation
            settings.
        """
        if os.path.isdir(inputs):
            for file in rw.list_dir(inputs):
                if "preparation.pkl" in file:
                    preparation = pickle.load(open(f"{inputs}/preparation.pkl", "rb"))
                    for attr, value in preparation.items():
                        setattr(self, attr, value)
                    break

    def save(
        self,
        output_dir: str = ".",
        outnames: Optional[List[str]] = None,
    ) -> None:
        """Saves (prepared) data.

        The ordering of the saved files matches the order of the input files.

        Parameters
        ----------
        output_dir : str
            Path to save data files to. Default is the current working
            directory.
        outnames : list of str, optional
            Names/IDs for output files (excluding the extension).
        """
        if outnames is not None:
            if len(outnames) != self.n_sessions:
                raise ValueError(
                    "len(outnames) should match the number of arrays. "
                    f"We have {self.n_sessions} arrays and len(outnames)={len(outnames)}."
                )
        else:
            outnames = []
            for i in range(self.n_sessions):
                padded_index = misc.leading_zeros(i, self.n_sessions)
                outnames.append(f"array{padded_index}")

        # Create output directory
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Function to save a single array
        def _save(i):
            filename = f"{output_dir}/{outnames[i]}.npy"
            np.save(filename, self.arrays[i])

        # Save arrays in parallel
        results = pqdm(
            range(self.n_sessions),
            _save,
            desc="Saving data",
            n_jobs=self.n_jobs,
            total=self.n_sessions,
        )
        _raise_exceptions(results)

        # Save preparation settings
        self.save_preparation(output_dir)

    def delete_dir(self) -> None:
        """Deletes :code:`store_dir`."""
        atexit.unregister(self._finalizer)
        self._finalizer()


@dataclass
class SessionLabels:
    """Class for session labels.

    Parameters
    ----------
    name : str
        Name of the session label.
    values : np.ndarray
        Value for each session. Must be a 1D array of numbers.
    label_type : str
        Type of the session label. Options are "categorical" and "continuous".
    """

    name: str
    values: np.ndarray
    label_type: str

    def __post_init__(self):
        if self.label_type not in ["categorical", "continuous"]:
            raise ValueError("label_type must be 'categorical' or 'continuous'.")

        if self.values.ndim != 1:
            raise ValueError("values must be a 1D array.")

        if self.label_type == "categorical":
            self.values = self.values.astype(np.float32)
            self.n_classes = len(np.unique(self.values))
        else:
            self.values = self.values.astype(np.float32)
            self.n_classes = None
