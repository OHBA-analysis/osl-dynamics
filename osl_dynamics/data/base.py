"""Base class for handling data.

"""

import logging
import pathlib
import pickle
from contextlib import contextmanager
from shutil import rmtree
from os import path

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
    store_dir : str, optional
        We don't read all the data into memory. Instead we create store them on
        disk and create memmaps (unless :code:`load_memmaps=False` is passed).
        This is the directory to save memmaps to. Default is :code:`./tmp`.
    time_axis_first : bool, optional
        Is the input data of shape (n_samples, n_channels)? Default is
        :code:`True`. If your data is in format (n_channels, n_samples), use
        :code:`time_axis_first=False`.
    load_memmaps : bool, optional
        Should we load the data as memory maps (memmaps)? If :code:`False`, we
        will load data into memory rather than storing it on disk. By default
        we will keep the data on disk and use memmaps.
    buffer_size : int, optional
        Buffer size for shuffling a TensorFlow Dataset. Smaller values will lead
        to less random shuffling but will be quicker. Default is 100000.
    use_tfrecord : bool, optional
        Should we save the data as a TensorFlow Record? This is recommended for
        training on large datasets. Default is :code:`False`.
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
        store_dir="tmp",
        time_axis_first=True,
        load_memmaps=True,
        buffer_size=100000,
        use_tfrecord=False,
        n_jobs=1,
    ):
        self._identifier = id(self)
        self.data_field = data_field
        self.picks = picks
        self.reject_by_annotation = reject_by_annotation
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
        self.store_dir.mkdir(parents=True, exist_ok=True)

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
            width=len(str(self.n_arrays)), identifier=self._identifier
        )
        self.prepared_data_filenames = [
            str(self.store_dir / prepared_data_pattern.format(i=i))
            for i in range(self.n_arrays)
        ]

        # Arrays to keep when making TensorFlow Datasets
        self.keep = list(range(self.n_arrays))

    def __iter__(self):
        return iter(self.arrays)

    def __getitem__(self, item):
        return self.arrays[item]

    def __str__(self):
        info = [
            f"{self.__class__.__name__}",
            f"id: {self._identifier}",
            f"n_arrays: {self.n_arrays}",
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
        """Number of samples for each array."""
        return sum([array.shape[-2] for array in self.arrays])

    @property
    def n_arrays(self):
        """Number of arrays."""
        return len(self.arrays)

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
            # validation
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
        if concatenate or self.n_arrays == 1:
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
            total=self.n_arrays,
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

        if use_raw and hasattr(self, "original_sampling_frequency"):
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
            total=self.n_arrays,
        )
        if any([isinstance(e, Exception) for e in self.arrays]):
            for i, e in enumerate(self.arrays):
                if isinstance(e, Exception):
                    e.args = (f"array {i}: {e}",)
                    _logger.exception(e, exc_info=False)
            raise e

        # Update sampling_frequency attributes
        self.original_sampling_frequency = self.sampling_frequency
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
            explained_variance = np.sum(s[:n_pca_components]) / np.sum(s)
            _logger.info(f"Explained variance: {100 * explained_variance:.1f}%")
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
            total=self.n_arrays,
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
            total=self.n_arrays,
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
            explained_variance = np.sum(s[:n_pca_components]) / np.sum(s)
            _logger.info(f"Explained variance: {100 * explained_variance:.1f}%")
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
            total=self.n_arrays,
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
            total=self.n_arrays,
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
            total=self.n_arrays,
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
            total=self.n_arrays,
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

        Wrapper for calling a series of data preparation methods. Any method in
        Data can be called.

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
        for method_name, kwargs in methods.items():
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
                + " of each array due to time embedding/sliding window."
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
                        f"Removing {array.shape[0] - n_keep} data points"
                        + f" from the end of array {i} due to sequencing."
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
            Number of sequences for each array's data.
        """
        return np.array(
            [
                dtf.get_n_sequences(array, sequence_length, step_size)
                for array in self.arrays
            ]
        )

    def dataset(
        self,
        sequence_length,
        batch_size,
        shuffle=True,
        validation_split=None,
        concatenate=True,
        step_size=None,
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

        Returns
        -------
        dataset : tf.data.Dataset or tuple
            Dataset for training or evaluating the model along with the
            validation set if :code:`validation_split` was passed.
        """
        import tensorflow as tf  # moved here to avoid slow imports

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.step_size = step_size or sequence_length

        n_sequences = self.count_sequences(self.sequence_length)

        datasets = []
        for i in range(self.n_arrays):
            if i not in self.keep:
                # We don't want to include this file in the dataset
                continue

            # Get time series data and ensure an integer multiple of sequence
            # length
            array = self.arrays[i][: n_sequences[i] * sequence_length]

            # Dataset with the time series data and ID
            array_tracker = np.zeros(array.shape[0], dtype=np.float32) + i
            data = {"data": array, "array_id": array_tracker}

            # Create dataset
            dataset = dtf.create_dataset(
                data,
                self.sequence_length,
                self.step_size,
            )
            datasets.append(dataset)

        # Create a dataset from all the arrays concatenated
        if concatenate:
            full_dataset = dtf.concatenate_datasets(datasets)

            if shuffle:
                # Shuffle sequences
                full_dataset = full_dataset.shuffle(self.buffer_size)

                # Group into mini-batches
                full_dataset = full_dataset.batch(self.batch_size)

                # Shuffle mini-batches
                full_dataset = full_dataset.shuffle(self.buffer_size)

            else:
                # Group into mini-batches
                full_dataset = full_dataset.batch(self.batch_size)

            if validation_split is None:
                # Return the full dataset
                return full_dataset.prefetch(tf.data.AUTOTUNE)

            else:
                # Calculate how many batches should be in the training dataset
                dataset_size = len(full_dataset)
                training_dataset_size = round((1.0 - validation_split) * dataset_size)

                # Split the full dataset into a training and validation dataset
                training_dataset = full_dataset.take(training_dataset_size)
                validation_dataset = full_dataset.skip(training_dataset_size)
                _logger.info(
                    f"{len(training_dataset)} batches in training dataset, "
                    + f"{len(validation_dataset)} batches in the validation "
                    + "dataset."
                )

                return training_dataset.prefetch(
                    tf.data.AUTOTUNE
                ), validation_dataset.prefetch(tf.data.AUTOTUNE)

        # Otherwise create a dataset for each array separately
        else:
            full_datasets = []
            for ds in datasets:
                if shuffle:
                    # Shuffle sequences
                    ds = ds.shuffle(self.buffer_size)

                # Group into batches
                ds = ds.batch(self.batch_size)

                if shuffle:
                    # Shuffle batches
                    ds = ds.shuffle(self.buffer_size)

                full_datasets.append(ds.prefetch(tf.data.AUTOTUNE))

            if validation_split is None:
                # Return the full dataset for each array
                return full_datasets

            else:
                # Split the dataset for each array separately
                training_datasets = []
                validation_datasets = []
                for i in range(len(full_datasets)):
                    # Calculate the number of batches in the training dataset
                    dataset_size = len(full_datasets[i])
                    training_dataset_size = round(
                        (1.0 - validation_split) * dataset_size
                    )

                    # Split this array's dataset
                    training_datasets.append(
                        full_datasets[i].take(training_dataset_size)
                    )
                    validation_datasets.append(
                        full_datasets[i].skip(training_dataset_size)
                    )
                    _logger.info(
                        f"Array {i}: "
                        + f"{len(training_datasets[i])} batches in training dataset, "
                        + f"{len(validation_datasets[i])} batches in the validation "
                        + "dataset."
                    )
                return training_datasets, validation_datasets

    def tfrecord_dataset(
        self,
        sequence_length,
        batch_size,
        shuffle=True,
        validation_split=None,
        concatenate=True,
        step_size=None,
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

        Returns
        -------
        dataset : tf.data.Dataset
            Dataset for training or evaluating the model.
        """
        import tensorflow as tf  # moved here to avoid slow imports

        tfrecord_paths = (
            f"{self.store_dir}"
            "/dataset_{array:0{v}d}-of-{n_array:0{v}d}"
            f".{self._identifier}.tfrecord"
        )

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.step_size = step_size or sequence_length

        n_sequences = self.count_sequences(self.sequence_length)

        # TFRecords we need to save
        tfrecord_filenames = []
        tfrecords_to_save = []
        for i in self.keep:
            filepath = tfrecord_paths.format(
                array=i,
                n_array=self.n_arrays - 1,
                v=len(str(self.n_arrays - 1)),
            )
            tfrecord_filenames.append(filepath)
            if not path.exists(filepath):
                tfrecords_to_save.append((i, filepath))

        # Function for saving a single TFRecord
        def _save_tfrecord(i, filepath):
            # Get time series data and ensure an integer multiple of
            # sequence length
            array = self.arrays[i][: n_sequences[i] * sequence_length]

            # Create a dataset with the time series data and ID
            array_tracker = np.zeros(array.shape[0], dtype=np.float32) + i
            data = {"data": array, "array_id": array_tracker}

            # Save the dataset
            dtf.save_tfrecord(
                data,
                self.sequence_length,
                self.step_size,
                filepath,
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

        # Helper function for parsing training examples
        def _parse_example(example):
            feature_names = ["data", "array_id"]
            feature_description = {
                name: tf.io.FixedLenFeature([], tf.string) for name in feature_names
            }
            parsed_example = tf.io.parse_single_example(
                example,
                feature_description,
            )
            return {
                name: tf.io.parse_tensor(tensor, tf.float32)
                for name, tensor in parsed_example.items()
            }

        # Create the TFRecord dataset
        if concatenate:
            tfrecord_filenames = tf.data.Dataset.from_tensor_slices(tfrecord_filenames)

            if shuffle:
                # First shuffle the shards
                tfrecord_filenames = tfrecord_filenames.shuffle(len(tfrecord_filenames))

                # Create the TFRecord dataset
                full_dataset = tfrecord_filenames.interleave(
                    tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE
                )

                # Parse the examples
                full_dataset = full_dataset.map(_parse_example)

                # Shuffle sequences
                full_dataset = full_dataset.shuffle(self.buffer_size)

                # Group into batches
                full_dataset = full_dataset.batch(self.batch_size)

                # Shuffle batches
                full_dataset = full_dataset.shuffle(self.buffer_size)

            else:
                # Create the TFRecord dataset
                full_dataset = tfrecord_filenames.interleave(
                    tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE
                )

                # Parse the examples
                full_dataset = full_dataset.map(_parse_example)

                # Group into batches
                full_dataset = full_dataset.batch(self.batch_size)

            if validation_split is None:
                # Return the dataset
                return full_dataset.prefetch(tf.data.AUTOTUNE)

            else:
                # Calculate how many batches should be in the training dataset
                dataset_size = dtf.get_n_batches(full_dataset)
                training_dataset_size = round((1.0 - validation_split) * dataset_size)

                # Split the dataset into training and validation datasets
                training_dataset = full_dataset.take(training_dataset_size)
                validation_dataset = full_dataset.skip(training_dataset_size)
                _logger.info(
                    f"{training_dataset_size} batches in training dataset, "
                    + f"{dataset_size - training_dataset_size} batches in the validation "
                    + "dataset."
                )
                return training_dataset.prefetch(
                    tf.data.AUTOTUNE
                ), validation_dataset.prefetch(tf.data.AUTOTUNE)

        # Otherwise create a dataset for each array separately
        else:
            full_datasets = []
            for filename in tfrecord_filenames:
                ds = tf.data.TFRecordDataset(filename)

                # Parse the examples
                ds = ds.map(_parse_example)

                if shuffle:
                    # Shuffle sequences
                    ds = ds.shuffle(self.buffer_size)

                # Group into batches
                ds = ds.batch(self.batch_size)

                if shuffle:
                    # Shuffle batches
                    ds = ds.shuffle(self.buffer_size)

                full_datasets.append(ds.prefetch(tf.data.AUTOTUNE))

            if validation_split is None:
                # Return the full dataset for each array
                return full_datasets

            else:
                # Split the dataset for each array separately
                training_datasets = []
                validation_datasets = []
                for i, ds in enumerate(full_datasets):
                    # Calculate how many batches should be in the training dataset
                    dataset_size = dtf.get_n_batches(ds)
                    training_dataset_size = round(
                        (1.0 - validation_split) * dataset_size
                    )

                    # Split the dataset into training and validation datasets
                    training_datasets.append(ds.take(training_dataset_size))
                    validation_datasets.append(ds.skip(training_dataset_size))
                    _logger.info(
                        f"Array {i}: "
                        + f"{training_dataset_size} batches in training dataset, "
                        + f"{dataset_size - training_dataset_size} batches in the validation "
                        + "dataset."
                    )

                return training_datasets, validation_datasets

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
        if path.isdir(inputs):
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
            padded_number = misc.leading_zeros(i, self.n_arrays)
            np.save(f"{output_dir}/array{padded_number}.npy", arr)

        # Save arrays in parallel
        pqdm(
            enumerate(self.arrays),
            _save,
            desc="Saving data",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_arrays,
        )

        # Save preparation settings
        self.save_preparation(output_dir)

    def delete_dir(self):
        """Deletes :code:`store_dir`."""
        if self.store_dir.exists():
            rmtree(self.store_dir)
