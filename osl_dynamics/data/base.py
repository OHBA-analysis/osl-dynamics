"""Base class for handling data."""

import re
import logging
import os
import pathlib
import pickle
import random
from contextlib import contextmanager
from shutil import rmtree
from dataclasses import dataclass

import numpy as np
from pqdm.threads import pqdm
from tqdm.auto import tqdm

from osl_dynamics.data import processing, rw, tf as dtf
from osl_dynamics.utils import misc

_logger = logging.getLogger("osl-dynamics")


class Data:
    """Data Class.

    The Data class enables the input and processing of data. When given a list
    of files, it produces a set of numpy memory maps which contain their raw
    data. It also provides methods for batching data and creating TensorFlow
    Datasets.

    Parameters
    ----------
    inputs : list of str or str or np.ndarray
        - A path to a directory containing :code:`.npy` files. Each
          :code:`.npy` file should be a subject or session.
        - A list of paths to :code:`.npy`, :code:`.mat` or :code:`.fif` files.
          Each file should be a subject or session. If a :code:`.fif` file is
          passed is must end with :code:`'raw.fif'` or :code:`'epo.fif'`.
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
    mask_file : str, optional
        Path to mask file used to source reconstruct the data.
    parcellation_file : str, optional
        Path to parcellation file used to source reconstruct the data.
    time_axis_first : bool, optional
        Is the input data of shape (n_samples, n_channels)? Default is
        :code:`True`. If your data is in format (n_channels, n_samples), use
        :code:`time_axis_first=False`.
    load_memmaps : bool, optional
        Should we load the data as memory maps (memmaps)? If :code:`True`, we
        will load store the data on disk rather than loading it into memory.
    store_dir : str, optional
        If `load_memmaps=True`, then we save data to disk and load it as
        a memory map. This is the directory to save the memory maps to.
        Default is :code:`./tmp`.
    buffer_size : int, optional
        Buffer size for shuffling a TensorFlow Dataset. Smaller values will lead
        to less random shuffling but will be quicker. Default is 100000.
    use_tfrecord : bool, optional
        Should we save the data as a TensorFlow Record? This is recommended for
        training on large datasets. Default is :code:`False`.
    session_labels : list of SessionLabels, optional
        Extra session labels.
    extra_channels : dict, optional
        Extra channels to add to the data. The keys are the channel names and
        the values are the channel data. 
    n_jobs : int, optional
        Number of processes to load the data in parallel.
        Default is 1, which loads data in serial.
    """

    def __init__(
        self,
        inputs,
        data_field="X",
        picks=None,
        reject_by_annotation=None,
        sampling_frequency=None,
        mask_file=None,
        parcellation_file=None,
        time_axis_first=True,
        load_memmaps=False,
        store_dir="tmp",
        buffer_size=4000,
        use_tfrecord=False,
        session_labels=None,
        extra_channels=None,
        n_jobs=1,
    ):
        self._identifier = id(self)
        self.data_field = data_field
        self.picks = picks
        self.reject_by_annotation = reject_by_annotation
        self.original_sampling_frequency = sampling_frequency
        self.sampling_frequency = sampling_frequency
        self.mask_file = mask_file
        self.parcellation_file = parcellation_file
        self.time_axis_first = time_axis_first
        self.load_memmaps = load_memmaps
        self.buffer_size = buffer_size
        self.use_tfrecord = use_tfrecord
        self.n_jobs = n_jobs

        # Validate inputs
        self.inputs = rw.validate_inputs(inputs)

        if len(self.inputs) == 0:
            raise ValueError("No valid inputs were passed.")

        # Directory to store memory maps created by this class
        self.store_dir = pathlib.Path(store_dir)
        self.store_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

        # Load and validate the raw data
        self.raw_data_arrays, self.raw_data_filenames = self.load_raw_data()
        self.validate_data()

        self.n_raw_data_channels = self.raw_data_arrays[0].shape[-1]

        # Get data preparation attributes if there's a pickle file in the
        # input directory
        if not isinstance(inputs, list):
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

        # Extra channels
        if extra_channels is None:
            self.extra_channels = {}
        else:
            self.extra_channels = extra_channels

    def __iter__(self):
        return iter(self.arrays)

    def __getitem__(self, item):
        return self.arrays[item]

    def __str__(self):
        info = [
            f"{self.__class__.__name__}",
            f"id: {self._identifier}",
            f"n_sessions: {self.n_sessions}",
            f"n_samples: {self.n_samples}",
            f"n_channels: {self.n_channels}",
        ]
        return "\n ".join(info)

    @property
    def raw_data(self):
        """Return raw data as a list of arrays."""
        return self.raw_data_arrays

    @property
    def n_channels(self):
        """Number of channels in the data files."""
        return self.arrays[0].shape[-1]

    @property
    def n_samples(self):
        """Number of samples across all arrays."""
        return sum([array.shape[-2] for array in self.arrays])

    @property
    def n_sessions(self):
        """Number of arrays."""
        return len(self.arrays)

    @property
    def input_shapes(self):
        """Get the input shapes for the model.

        Returns
        -------
        shapes : dict
            Dictionary of input shapes.
        """
        if getattr(self, "sequence_length", None) is None:
            raise ValueError("Data.sequence_length must be set.")

        input_shapes = {"data": [self.sequence_length, self.n_channels]}
        for session_label in self.session_labels:
            input_shapes[session_label.name] = [self.sequence_length]

        for channel_name, channel_values in self.extra_channels.items():
            if channel_values[0].ndim == 1:
                input_shapes[channel_name] = [self.sequence_length]
            else:
                input_shapes[channel_name] = [
                    self.sequence_length,
                    channel_values[0].shape[-1],
                ]

        return input_shapes

    @contextmanager
    def set_keep(self, keep):
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

    def set_sampling_frequency(self, sampling_frequency):
        """Sets the :code:`sampling_frequency` attribute.

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency in Hz.
        """
        self.original_sampling_frequency = sampling_frequency
        self.sampling_frequency = sampling_frequency

    def set_buffer_size(self, buffer_size):
        """Set the :code:`buffer_size` attribute.

        Parameters
        ----------
        buffer_size : int
            Buffer size for shuffling a TensorFlow Dataset. Smaller values will
            lead to less random shuffling but will be quicker.
        """
        self.buffer_size = buffer_size

    def time_series(self, prepared=True, concatenate=False):
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

    def load_raw_data(self):
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

        return memmaps, raw_data_filenames

    def validate_data(self):
        """Validate data files."""
        n_channels = [array.shape[-1] for array in self.raw_data_arrays]
        if not np.equal(n_channels, n_channels[0]).all():
            raise ValueError("All inputs should have the same number of channels.")

    def validate_extra_channels(self, data, extra_channels):
        """Validate extra channels."""
        n_sessions = len(data)

        # Validate each channel
        for channel_name, channel in extra_channels.items():
            if not isinstance(channel_name, str):
                raise ValueError("Channel name must be a string.")

            if isinstance(channel, np.ndarray):
                channel = [channel]
                extra_channels[channel_name] = channel

            if not isinstance(channel, list):
                raise ValueError(f"Extra channel {channel_name} must be a list.")

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
                        f"Extra channel {channel_name} have different number of samples than the data in session {i}."
                    )

                extra_channels[channel_name][i] = channel_data.astype(np.float32)

        return extra_channels

    def _validate_batching(
        self,
        sequence_length,
        batch_size,
        step_size=None,
        drop_last_batch=False,
        concatenate=True,
    ):
        """Validate the batching process.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the
            model.
        step_size : int, optional
            Number of samples to slide the sequence across the dataset.
            Default is no overlap.
        drop_last_batch : bool, optional
            Should we drop the last batch if it is smaller than the batch size?
            Defaults to False.
        concatenate : bool, optional
            Should we concatenate the datasets for each array?
            Defaults to True.
        """

        # Calculate number of sequences per session
        n_sequences_per_session = [
            (array.shape[0] - sequence_length) // step_size + 1 for array in self.arrays
        ]

        # Calculate number of batches
        if concatenate:
            # Calculate total batches across concatenated sequences
            total_n_sequences = sum(n_sequences_per_session)
            n_batches = total_n_sequences // batch_size
            # Add one more batch if the last incomplete batch is not dropped
            if not drop_last_batch and total_n_sequences % batch_size != 0:
                n_batches += 1
        else:
            # Calculate batches per session individually, then sum
            n_batches_per_session = [
                n // batch_size + (0 if drop_last_batch or n % batch_size == 0 else 1)
                for n in n_sequences_per_session
            ]
            n_batches = sum(n_batches_per_session)

        if n_batches < 1:
            raise ValueError(
                "Number of batches must be greater than or equal to 1. "
                + "Please adjust your sequence length and batch size."
            )

    def select(self, channels=None, sessions=None, use_raw=False):
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
            raise ValueError("channels must be an int or list of int.")

        if not isinstance(sessions, list):
            raise ValueError("sessions must be an int or list of int.")

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

    def filter(self, low_freq=None, high_freq=None, use_raw=False):
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
                array, low_freq, high_freq, self.sampling_frequency
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
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def downsample(self, freq, use_raw=False):
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
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        # Update sampling_frequency attributes
        self.sampling_frequency = freq

        return self

    def pca(
        self,
        n_pca_components=None,
        pca_components=None,
        whiten=False,
        use_raw=False,
    ):
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
            raise ValueError("pca_components must be a numpy array.")

        self.n_pca_components = n_pca_components
        self.pca_components = pca_components
        self.whiten = whiten

        # What data should we apply PCA to?
        arrays = self.raw_data_arrays if use_raw else self.arrays

        # Calculate PCA
        if n_pca_components is not None:
            # Calculate covariance of the data
            n_channels = arrays[0].shape[-1]
            covariance = np.zeros([n_channels, n_channels])
            for array in tqdm(arrays, desc="Calculating PCA components"):
                std_data = processing.standardize(array)
                covariance += np.transpose(std_data) @ std_data

            # Use SVD on the covariance to calculate PCA components
            u, s, vh = np.linalg.svd(covariance)
            u = u[:, :n_pca_components].astype(np.float32)
            self.explained_variance = np.sum(s[:n_pca_components]) / np.sum(s)
            _logger.info(f"Explained variance: {100 * self.explained_variance:.1f}%")
            s = s[:n_pca_components].astype(np.float32)
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
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="PCA",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def tde(self, n_embeddings, use_raw=False):
        """Time-delay embedding (TDE).

        This is an in-place operation.

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
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def tde_pca(
        self,
        n_embeddings,
        n_pca_components=None,
        pca_components=None,
        whiten=False,
        use_raw=False,
    ):
        """Time-delay embedding (TDE) and principal component analysis (PCA).

        This function will first standardize the data, then perform TDE then
        PCA. It is useful to do both operations in a single methods because
        it avoids having to save the time-embedded data. This is an in-place
        operation.

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
            raise ValueError("pca_components must be a numpy array.")

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
            covariance = np.zeros([self.n_te_channels, self.n_te_channels])
            for array in tqdm(arrays, desc="Calculating PCA components"):
                std_data = processing.standardize(array)
                te_std_data = processing.time_embed(std_data, n_embeddings)
                covariance += np.transpose(te_std_data) @ te_std_data

            # Use SVD on the covariance to calculate PCA components
            u, s, vh = np.linalg.svd(covariance)
            u = u[:, :n_pca_components].astype(np.float32)
            self.explained_variance = np.sum(s[:n_pca_components]) / np.sum(s)
            _logger.info(f"Explained variance: {100 * self.explained_variance:.1f}%")
            s = s[:n_pca_components].astype(np.float32)
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
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="TDE-PCA",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def amplitude_envelope(self, use_raw=False):
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
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def moving_average(self, n_window, use_raw=False):
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
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def standardize(self, use_raw=False):
        """Standardize (z-transform) the data.

        This is an in-place operation.

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?
        """

        # Function to apply standardisation to a single array
        def _apply(array):
            return processing.standardize(array, create_copy=False)

        # Apply standardisation to each array in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        self.arrays = pqdm(
            array=zip(arrays),
            function=_apply,
            desc="Standardize",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def align_channel_signs(
        self,
        template_data=None,
        template_cov=None,
        n_init=3,
        n_iter=2500,
        max_flips=20,
        n_embeddings=1,
        standardize=True,
        use_raw=False,
    ):
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
        n_init : int, optional
            Number of initializations.
        n_iter : int, optional
            Number of sign flipping iterations per subject to perform.
        max_flips : int, optional
            Maximum number of channels to flip in an iteration.
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

        if self.n_channels < max_flips:
            _logger.warning(
                f"max_flips={max_flips} cannot be greater than "
                f"n_channels={self.n_channels}. "
                f"Setting max_flips={self.n_channels}."
            )
            max_flips = self.n_channels

        if isinstance(template_data, str):
            template_data = rw.load_data(
                template_data,
                self.data_field,
                self.picks,
                self.reject_by_annotation,
                memmap_location=None,
                mmap_mode="r",
            )
            if not self.time_axis_first:
                template_data = template_data.T

        if isinstance(template_cov, str):
            template_cov = np.load(template_cov)

        # Helper functions
        def _calc_cov(array):
            array = processing.time_embed(array, n_embeddings)
            if standardize:
                array = processing.standardize(array, create_copy=False)
            return np.cov(array.T)

        def _calc_corr(M1, M2, mode=None):
            if mode == "abs":
                M1 = np.abs(M1)
                M2 = np.abs(M2)
            m, n = np.triu_indices(M1.shape[0], k=n_embeddings)
            M1 = M1[m, n]
            M2 = M2[m, n]
            return np.corrcoef([M1, M2])[0, 1]

        def _calc_metrics(covs):
            metric = np.zeros([self.n_sessions, self.n_sessions])
            for i in tqdm(range(self.n_sessions), desc="Comparing sessions"):
                for j in range(i + 1, self.n_sessions):
                    metric[i, j] = _calc_corr(covs[i], covs[j], mode="abs")
                    metric[j, i] = metric[i, j]
            return metric

        def _randomly_flip(flips, max_flips):
            n_channels_to_flip = np.random.choice(max_flips, size=1)
            random_channels_to_flip = np.random.choice(
                self.n_channels, size=n_channels_to_flip, replace=False
            )
            new_flips = np.copy(flips)
            new_flips[random_channels_to_flip] *= -1
            return new_flips

        def _apply_flips(cov, flips):
            flips = np.repeat(flips, n_embeddings)[np.newaxis, ...]
            flips = flips.T @ flips
            return cov * flips

        def _find_and_apply_flips(cov, tcov, array, ind):
            best_flips = np.ones(self.n_channels)
            best_metric = 0
            for n in range(n_init):
                flips = np.ones(self.n_channels)
                metric = _calc_corr(cov, tcov)
                for j in range(n_iter):
                    new_flips = _randomly_flip(flips, max_flips)
                    new_cov = _apply_flips(cov, new_flips)
                    new_metric = _calc_corr(new_cov, tcov)
                    if new_metric > metric:
                        flips = new_flips
                        metric = new_metric
                if metric > best_metric:
                    best_metric = metric
                    best_flips = flips
                _logger.info(
                    f"Session {ind}, Init {n}, best correlation with template: "
                    f"{best_metric:.3f}"
                )
            return array * best_flips[np.newaxis, ...].astype(np.float32)

        # What data do we use?
        arrays = self.raw_data_arrays if use_raw else self.arrays

        # Calculate covariance of each session
        covs = pqdm(
            array=zip(arrays),
            function=_calc_cov,
            desc="Calculating covariances",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in covs]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        # Calculate/get template covariances
        if template_cov is None:
            metrics = _calc_metrics(covs)
            metrics_sum = np.sum(metrics, axis=1)
            argmedian = np.argsort(metrics_sum)[len(metrics_sum) // 2]
            _logger.info(f"Using session {argmedian} as template")
            template_cov = covs[argmedian]

        if template_data is not None:
            template_cov = _calc_cov(template_data)

        # Perform the sign flipping
        _logger.info("Aligning channel signs across sessions")
        tcovs = [template_cov] * self.n_sessions
        indices = range(self.n_sessions)
        self.arrays = pqdm(
            array=zip(covs, tcovs, arrays, indices),
            function=_find_and_apply_flips,
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
            disable=True,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def remove_bad_segments(
        self,
        window_length=None,
        significance_level=0.05,
        maximum_fraction=0.1,
        use_raw=False,
    ):
        """Automated bad segment removal using the G-ESD algorithm.

        Parameters
        ----------
        window_length : int, optional
            Window length to used to calculate statistics.
            Defaults to twice the sampling frequency.
        significance_level : float, optional
            Significance level (p-value) to consider as an outlier.
        maximum_fraction : float, optional
            Maximum fraction of time series to mark as bad.
        use_raw : bool, optional
            Should we prepare the original 'raw' data that we loaded?

        Returns
        -------
        data : osl_dynamics.data.Data
            The modified Data object.
        """
        self.gesd_window_length = window_length
        self.gesd_significance_level = significance_level
        self.gesd_maximum_fraction = maximum_fraction

        if window_length is None:
            if self.sampling_frequency is None:
                raise ValueError(
                    "window_length must be passed. Alternatively, set the "
                    "sampling frequency to use "
                    "window_length=2*sampling_frequency. The sampling "
                    "frequency can be set using Data.set_sampling_frequency() "
                    "or pass Data(..., sampling_frequency=...) when creating "
                    "the Data object."
                )
            else:
                window_length = 2 * self.sampling_frequency

        # Function to remove bad segments to a single array
        def _apply(array, prepared_data_file):
            array = processing.remove_bad_segments(
                array, window_length, significance_level, maximum_fraction
            )
            if self.load_memmaps:
                array = misc.array_to_memmap(prepared_data_file, array)
            return array

        # Run in parallel
        arrays = self.raw_data_arrays if use_raw else self.arrays
        args = zip(arrays, self.prepared_data_filenames)
        self.arrays = pqdm(
            args,
            function=_apply,
            desc="Bad segment removal",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        return self

    def prepare(self, methods):
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
        sequence_length=None,
        n_embeddings=None,
        n_window=None,
        prepared=True,
        concatenate=False,
        verbose=False,
    ):
        """Trims the data time series.

        Removes the data points that are lost when the data is prepared,
        i.e. due to time embedding and separating into sequences, but does not
        perform time embedding or batching into sequences on the time series.

        Parameters
        ----------
        sequence_length : int, optional
            Length of the segement of data to feed into the model.
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
            Trimed time series for each array.
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

    def count_sequences(self, sequence_length, step_size=None):
        """Count sequences.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        step_size : int, optional
            The number of samples by which to move the sliding window between
            sequences. Defaults to :code:`sequence_length`.

        Returns
        -------
        n : np.ndarray
            Number of sequences for each session's data.
        """
        return np.array(
            [
                dtf.get_n_sequences(array, sequence_length, step_size)
                for array in self.arrays
            ]
        )

    def _create_data_dict(self, i, array, extra_channels):
        """Create a dictionary of data for a single session.

        Parameters
        ----------
        i : int
            Index of the session.
        array : np.ndarray
            Time series data for a single session.
        extra_channels : dict
            Dictionary of extra channels to add to the data.

        Returns
        -------
        data : dict
            Dictionary of data for a single session.
        """
        data = {"data": array}

        # Add other session labels
        placeholder = np.zeros(array.shape[0], dtype=np.float32)
        for session_label in self.session_labels:
            label_name = session_label.name
            label_values = session_label.values
            data[label_name] = placeholder + label_values[i]

        # Add extra channels
        for k, v in extra_channels.items():
            if k in data:
                raise ValueError(f"Channel name '{k}' already exists.")
            data[k] = v[i]

        return data

    def _trim_data(self, data, sequence_length, n_sequences):
        """Trim data to be an integer multiple of the sequence length."""
        X = []
        for i in range(self.n_sessions):
            X.append(data[i][: n_sequences[i] * sequence_length])
        return X

    def _validation_split(self, X, extra_channels, validation_split):
        """Split the data into training and validation sets."""

        def _split_data(d, val_indx, train_indx):
            if d.ndim == 1:
                d = d[:, None]
            n_channels = d.shape[-1]
            d = d.reshape(-1, self.sequence_length, n_channels)
            d_train = d[train_indx].reshape(-1, n_channels)
            d_val = d[val_indx].reshape(-1, n_channels)
            return np.squeeze(d_train), np.squeeze(d_val)

        n_sequences = self.count_sequences(self.sequence_length)

        # Number of sequences that should be in the validation set
        n_val_sequences = (validation_split * n_sequences).astype(int)

        if np.all(n_val_sequences == 0):
            raise ValueError(
                "No full sequences could be assigned to the validation set. "
                "Consider reducing the sequence_length."
            )

        X_train = []
        X_val = []
        extra_channels_train = {k: [] for k in extra_channels.keys()}
        extra_channels_val = {k: [] for k in extra_channels.keys()}

        for i in range(self.n_sessions):
            # Randomly pick sequences
            val_indx = np.random.choice(
                n_sequences[i], size=n_val_sequences[i], replace=False
            )
            train_indx = np.setdiff1d(np.arange(n_sequences[i]), val_indx)

            # Split data
            x_train, x_val = _split_data(X[i], val_indx, train_indx)
            X_train.append(x_train)
            X_val.append(x_val)

            for k in extra_channels.keys():
                x_train, x_val = _split_data(extra_channels[k][i], val_indx, train_indx)
                extra_channels_train[k].append(x_train)
                extra_channels_val[k].append(x_val)

        return X_train, X_val, extra_channels_train, extra_channels_val

    def dataset(
        self,
        sequence_length,
        batch_size,
        shuffle=True,
        validation_split=None,
        concatenate=True,
        step_size=None,
        drop_last_batch=False,
        repeat_count=1,
    ):
        """Create a Tensorflow Dataset for training or evaluation.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the
            model.
        shuffle : bool, optional
            Should we shuffle sequences (within a batch) and batches.
        validation_split : float, optional
            Ratio to split the dataset into a training and validation set.
        concatenate : bool, optional
            Should we concatenate the datasets for each array?
        step_size : int, optional
            Number of samples to slide the sequence across the dataset.
            Default is no overlap.
        drop_last_batch : bool, optional
            Should we drop the last batch if it is smaller than the batch size?
        repeat_count : int, optional
            Number of times to repeat the dataset. Default is once.

        Returns
        -------
        dataset : tf.data.Dataset or tuple of tf.data.Dataset
            Dataset for training or evaluating the model along with the
            validation set if :code:`validation_split` was passed.
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.step_size = step_size or sequence_length
        self.validation_split = validation_split

        # Validate batching
        self._validate_batching(
            sequence_length,
            batch_size,
            step_size=self.step_size,
            drop_last_batch=drop_last_batch,
            concatenate=concatenate,
        )

        # Validate extra channels
        self.extra_channels = self.validate_extra_channels(
            self.arrays, self.extra_channels
        )

        n_sequences = self.count_sequences(self.sequence_length)

        def _create_dataset(
            X, extra_channels, shuffle=shuffle, repeat_count=repeat_count
        ):
            # X is a list of np.ndarray

            # Create datasets for each array
            datasets = []
            for i in range(self.n_sessions):
                if i not in self.keep:
                    continue
                data = self._create_data_dict(i, X[i], extra_channels)
                dataset = dtf.create_dataset(
                    data,
                    self.sequence_length,
                    self.step_size,
                )
                datasets.append(dataset)

            # Create a dataset from all the arrays concatenated
            if concatenate:
                if shuffle:
                    # Do a perfect shuffle then concatenate across arrays
                    random.shuffle(datasets)
                    full_dataset = dtf.concatenate_datasets(datasets)

                    # Shuffle sequences
                    full_dataset = full_dataset.shuffle(self.buffer_size)

                    # Group into mini-batches
                    full_dataset = full_dataset.batch(
                        self.batch_size, drop_remainder=drop_last_batch
                    )

                    # Shuffle mini-batches
                    full_dataset = full_dataset.shuffle(self.buffer_size)

                else:
                    # Concatenate across arrays
                    full_dataset = dtf.concatenate_datasets(datasets)

                    # Group into mini-batches
                    full_dataset = full_dataset.batch(
                        self.batch_size, drop_remainder=drop_last_batch
                    )

                # Repeat the dataset
                full_dataset = full_dataset.repeat(repeat_count)

                import tensorflow as tf  # moved here to avoid slow imports

                return full_dataset.prefetch(tf.data.AUTOTUNE)

            # Otherwise create a dataset for each array separately
            else:
                full_datasets = []
                for i, ds in enumerate(datasets):
                    if shuffle:
                        # Shuffle sequences
                        ds = ds.shuffle(self.buffer_size)

                    # Group into batches
                    ds = ds.batch(self.batch_size, drop_remainder=drop_last_batch)

                    if shuffle:
                        # Shuffle batches
                        ds = ds.shuffle(self.buffer_size)

                    # Repeat the dataset
                    ds = ds.repeat(repeat_count)

                    import tensorflow as tf  # moved here to avoid slow imports

                    full_datasets.append(ds.prefetch(tf.data.AUTOTUNE))

                return full_datasets

        # Trim data to be an integer multiple of the sequence length
        X = self._trim_data(self.arrays, sequence_length, n_sequences)
        extra_channels = {}
        for k, v in self.extra_channels.items():
            extra_channels[k] = self._trim_data(v, sequence_length, n_sequences)

        if validation_split is not None:
            X_train, X_val, extra_channels_train, extra_channels_val = (
                self._validation_split(X, extra_channels, validation_split)
            )

            return _create_dataset(X_train, extra_channels_train), _create_dataset(
                X_val, extra_channels_val, shuffle=False, repeat_count=1
            )

        else:
            return _create_dataset(X, extra_channels)

    def save_tfrecord_dataset(
        self,
        tfrecord_dir,
        sequence_length,
        step_size=None,
        validation_split=None,
        overwrite=False,
    ):
        """Save the data as TFRecord files.

        Parameters
        ----------
        tfrecord_dir : str
            Directory to save the TFRecord datasets.
        sequence_length : int
            Length of the segement of data to feed into the model.
        step_size : int, optional
            Number of samples to slide the sequence across the dataset.
            Default is no overlap.
        validation_split : float, optional
            Ratio to split the dataset into a training and validation set.
        overwrite : bool, optional
            Should we overwrite the existing TFRecord datasets if there is a need?
        """
        os.makedirs(tfrecord_dir, mode=0o700, exist_ok=True)

        self.sequence_length = sequence_length
        self.step_size = step_size or sequence_length
        self.validation_split = validation_split

        def _check_rewrite():
            if not os.path.exists(f"{tfrecord_dir}/tfrecord_config.pkl"):
                _logger.warning(
                    "No tfrecord_config.pkl file found. Rewriting TFRecords."
                )
                return True

            if not overwrite:
                return False

            # Check if we need to rewrite the TFRecord datasets
            tfrecord_config = misc.load(f"{tfrecord_dir}/tfrecord_config.pkl")

            if tfrecord_config["identifier"] != self._identifier:
                _logger.warning("Identifier has changed. Rewriting TFRecords.")
                return True

            if tfrecord_config["sequence_length"] != self.sequence_length:
                _logger.warning("Sequence length has changed. Rewriting TFRecords.")
                return True

            if tfrecord_config["n_channels"] != self.n_channels:
                _logger.warning("Number of channels has changed. Rewriting TFRecords.")
                return True

            if tfrecord_config["step_size"] != self.step_size:
                _logger.warning("Step size has changed. Rewriting TFRecords.")
                return True

            if tfrecord_config["validation_split"] != self.validation_split:
                _logger.warning("Validation split has changed. Rewriting TFRecords.")
                return True

            if tfrecord_config["n_sessions"] != self.n_sessions:
                _logger.warning("Number of sessions has changed. Rewriting TFRecords.")
                return True

            input_shapes = tfrecord_config["input_shapes"]
            for input_name, input_shape in self.input_shapes.items():
                if input_name not in input_shapes:
                    _logger.warning(
                        f"Input shape {input_name} not found. Rewriting TFRecords."
                    )
                    return True
                if input_shapes[input_name] != input_shape:
                    _logger.warning(
                        f"Input shape {input_name} has changed. Rewriting TFRecords."
                    )
                    return True

            return False

        # Number of sequences
        n_sequences = self.count_sequences(self.sequence_length)

        # Path to TFRecord file
        tfrecord_path = (
            f"{tfrecord_dir}"
            "/dataset-{val}_{array:0{v}d}-of-{n_session:0{v}d}"
            f".{self._identifier}.tfrecord"
        )

        # TFRecords we need to save
        tfrecord_filenames = []
        tfrecords_to_save = []
        rewrite = _check_rewrite()
        for i in self.keep:
            filepath = tfrecord_path.format(
                array=i,
                val="{val}",
                n_session=self.n_sessions - 1,
                v=len(str(self.n_sessions - 1)),
            )
            tfrecord_filenames.append(filepath)

            rewrite_ = rewrite or not os.path.exists(filepath.format(val=0))
            if validation_split is not None:
                rewrite_ = rewrite_ or not os.path.exists(filepath.format(val=1))
            if rewrite_:
                tfrecords_to_save.append((i, filepath))

        # Validate extra channels
        self.extra_channels = self.validate_extra_channels(
            self.arrays, self.extra_channels
        )

        # Trim data to be an integer multiple of the sequence length
        X = self._trim_data(self.arrays, sequence_length, n_sequences)
        extra_channels = {}
        for k, v in self.extra_channels.items():
            extra_channels[k] = self._trim_data(v, sequence_length, n_sequences)

        if validation_split is not None:
            X_train, X_val, extra_channels_train, extra_channels_val = (
                self._validation_split(X, extra_channels, validation_split)
            )

        # Function for saving a single TFRecord
        def _save_tfrecord(i, filepath):
            if validation_split is not None:
                dtf.save_tfrecord(
                    self._create_data_dict(i, X_train[i], extra_channels_train),
                    self.sequence_length,
                    self.step_size,
                    filepath.format(val=0),
                )
                dtf.save_tfrecord(
                    self._create_data_dict(i, X_val[i], extra_channels_val),
                    self.sequence_length,
                    self.step_size,
                    filepath.format(val=1),
                )

            else:
                dtf.save_tfrecord(
                    self._create_data_dict(i, X[i], extra_channels),
                    self.sequence_length,
                    self.step_size,
                    filepath.format(val=0),
                )

        # Save TFRecords
        if len(tfrecords_to_save) > 0:
            pqdm(
                array=tfrecords_to_save,
                function=_save_tfrecord,
                n_jobs=self.n_jobs,
                desc="Creating TFRecord datasets",
                argument_type="args",
                total=len(tfrecords_to_save),
            )

        # Save tfrecords config
        if rewrite:
            tfrecord_config = {
                "identifier": self._identifier,
                "sequence_length": self.sequence_length,
                "n_channels": self.n_channels,
                "step_size": self.step_size,
                "validation_split": self.validation_split,
                "n_sessions": self.n_sessions,
                "input_shapes": self.input_shapes,
            }
            misc.save(f"{tfrecord_dir}/tfrecord_config.pkl", tfrecord_config)

    def tfrecord_dataset(
        self,
        sequence_length,
        batch_size,
        shuffle=True,
        validation_split=None,
        concatenate=True,
        step_size=None,
        drop_last_batch=False,
        repeat_count=1,
        tfrecord_dir=None,
        overwrite=False,
    ):
        """Create a TFRecord Dataset for training or evaluation.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the model.
        shuffle : bool, optional
            Should we shuffle sequences (within a batch) and batches.
        validation_split : float, optional
            Ratio to split the dataset into a training and validation set.
        concatenate : bool, optional
            Should we concatenate the datasets for each array?
        step_size : int, optional
            Number of samples to slide the sequence across the dataset.
            Default is no overlap.
        drop_last_batch : bool, optional
            Should we drop the last batch if it is smaller than the batch size?
        repeat_count : int, optional
            Number of times to repeat the dataset. Default is once.
        tfrecord_dir : str, optional
            Directory to save the TFRecord datasets. If :code:`None`, then
            :code:`Data.store_dir` is used.
        overwrite : bool, optional
            Should we overwrite the existing TFRecord datasets if there is a need?

        Returns
        -------
        dataset : tf.data.TFRecordDataset or tuple of tf.data.TFRecordDataset
            Dataset for training or evaluating the model along with the
            validation set if :code:`validation_split` was passed.
        """
        tfrecord_dir = tfrecord_dir or self.store_dir

        # Validate batching
        self._validate_batching(
            sequence_length,
            batch_size,
            step_size=(step_size or sequence_length),
            drop_last_batch=drop_last_batch,
            concatenate=concatenate,
        )

        # Save and load the TFRecord files
        self.save_tfrecord_dataset(
            tfrecord_dir=tfrecord_dir,
            sequence_length=sequence_length,
            step_size=step_size,
            validation_split=validation_split,
            overwrite=overwrite,
        )
        return dtf.load_tfrecord_dataset(
            tfrecord_dir=tfrecord_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            concatenate=concatenate,
            drop_last_batch=drop_last_batch,
            repeat_count=repeat_count,
            buffer_size=self.buffer_size,
            keep=self.keep,
        )

    def add_session_labels(self, label_name, label_values, label_type):
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

    def add_extra_channel(self, channel_name, channel_values):
        """Add an extra channel to the data."""
        if len(channel_values) != self.n_sessions:
            raise ValueError(
                "channel_values must have the same length as the number of sessions."
            )
        if channel_name in self.extra_channels:
            raise ValueError(f"Channel name '{channel_name}' already exists.")

        self.extra_channels[channel_name] = channel_values

    def get_session_labels(self):
        """Get the session labels.

        Returns
        -------
        session_labels : List[SessionLabels]
            List of session labels.
        """
        return self.session_labels

    def save_preparation(self, output_dir="."):
        """Save a pickle file containing preparation settings.

        Parameters
        ----------
        output_dir : str
            Path to save data files to. Default is the current working
            directory.
        """
        attributes = list(self.__dict__.keys())
        dont_keep = [
            "_identifier",
            "data_field",
            "picks",
            "reject_by_annotation",
            "original_sampling_frequency",
            "sampling_frequency",
            "mask_file",
            "parcellation_file",
            "time_axis_first",
            "load_memmaps",
            "buffer_size",
            "n_jobs",
            "prepared_data_filenames",
            "inputs",
            "store_dir",
            "raw_data_arrays",
            "raw_data_filenames",
            "n_raw_data_channels",
            "arrays",
            "keep",
            "use_tfrecord",
        ]
        for item in dont_keep:
            if item in attributes:
                attributes.remove(item)
        preparation = {a: getattr(self, a) for a in attributes}
        pickle.dump(preparation, open(f"{output_dir}/preparation.pkl", "wb"))

    def load_preparation(self, inputs):
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

    def save(self, output_dir="."):
        """Saves (prepared) data to numpy files.

        Parameters
        ----------
        output_dir : str
            Path to save data files to. Default is the current working
            directory.
        """
        # Create output directory
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Function to save a single array
        def _save(i, arr):
            padded_number = misc.leading_zeros(i, self.n_sessions)
            np.save(f"{output_dir}/array{padded_number}.npy", arr)

        # Save arrays in parallel
        pqdm(
            enumerate(self.arrays),
            _save,
            desc="Saving data",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_sessions,
        )

        # Save preparation settings
        self.save_preparation(output_dir)

    def delete_dir(self):
        """Deletes :code:`store_dir`."""
        if self.store_dir.exists():
            rmtree(self.store_dir)


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
            self.values = self.values.astype(np.int32)
            self.n_classes = len(np.unique(self.values))
        else:
            self.values = self.values.astype(np.float32)
            self.n_classes = None
