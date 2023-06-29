"""Base class for handling data.

"""

import logging
import pathlib
from functools import partial
from shutil import rmtree
from contextlib import contextmanager

import numpy as np
from pqdm.threads import pqdm
from scipy import signal
from tqdm.auto import tqdm

from osl_dynamics.data import processing, rw, tf
from osl_dynamics.utils import misc

_logger = logging.getLogger("osl-dynamics")


class Data:
    """Data Class.

    The Data class enables the input and processing of data. When given a list of
    files, it produces a set of numpy memory maps which contain their raw data.
    It also provides methods for batching data and creating TensorFlow Datasets.

    Parameters
    ----------
    inputs : list of str or str or np.ndarray
        - A path to a directory containing .npy files.
          Each .npy file should be a subject or session.
        - A list of paths to .npy, .mat or .fif files. Each file should be a subject
          or session. If a .fif file is passed is must end with 'raw.fif' or 'epo.fif'.
        - A numpy array. The array will be treated as continuous data from the
          same subject.
        - A list of numpy arrays. Each numpy array should be the data for a subject
          or session.

        The data files or numpy arrays should be in the format (n_samples, n_channels).
        If your data is in (n_channels, n_samples) format, use time_axis_first=False.
    data_field : str
        If a MATLAB (.mat) file is passed, this is the field that corresponds to the
        time series data. By default we read the field 'X'. If a numpy (.npy) file is
        passed, this is ignored. This argument is optional.
    picks : str or list of str
        If a fif file is passed we load the data using the MNE object's get_data()
        method. We pass this argument to the get_data() method. See the
        `MNE docs <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.get_data>`_
        for further details. This argument is optional. By default picks=None retrieves
        all channel types.
    reject_by_annotation : str
        If a fif file is passed we load the data using the MNE object's get_data()
        method. If the fif file contains a mne.Raw object, we pass this argument to
        the get_data() method. This argument is optional. By default
        reject_by_annotation=None retrieves all time points. Use
        reject_by_annotation="omit" to remove segments marked as bad.
    sampling_frequency : float
        Sampling frequency of the data in Hz. This argument is optional.
    mask_file : str
        Path to mask file used to source reconstruct the data. This argument is
        optional.
    parcellation_file : str
        Path to parcellation file used to source reconstruct the data. This argument
        is optional.
    store_dir : str
        We don't read all the data into memory. Instead we create store them on
        disk and create memmaps (unless load_memmaps=False is passed).
        This is the directory to save memmaps to. Default is ./tmp.
        This argument is optional.
    time_axis_first : bool
        Is the input data of shape (n_samples, n_channels)? Default is True.
        If your data is in format (n_channels, n_samples), use
        time_axis_first=False. This argument is optional.
    load_memmaps: bool
        Should we load the data as memory maps (memmaps)? If False, we will load data
        into memory rather than storing it on disk. By default we will keep the data
        on disk and use memmaps. This argument is optional.
    n_jobs : int
        Number of processes to load the data in parallel. This argument is optional.
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
        self.n_jobs = n_jobs
        self.prepared_data_filenames = []

        # Validate inputs
        self.inputs = rw.validate_inputs(inputs)

        if len(self.inputs) == 0:
            raise ValueError("No valid inputs were passed.")

        # Directory to store memory maps created by this class
        self.store_dir = pathlib.Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Load and validate the raw data
        self.raw_data_memmaps, self.raw_data_filenames = self.load_raw_data()
        self.validate_data()

        self.n_raw_data_channels = self.raw_data_memmaps[0].shape[-1]

        # Store raw data in the arrays attribute
        self.arrays = self.raw_data_memmaps

        # Subjects that are kept for making tensorflow datasets
        self.keep = list(range(len(self.arrays)))

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
        return self.raw_data_memmaps

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
        """Sets the sampling_frequency attribute.

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency in Hz.
        """
        self.sampling_frequency = sampling_frequency

    def time_series(self, prepared=True, concatenate=False):
        """Time series data for all arrays.

        Parameters
        ----------
        prepared : bool
            Should we return the latest data after we have prepared it or
            the original data we loaded into the Data object?
        concatenate : bool
            Should we return the time series for each array concatenated?

        Returns
        -------
        ts : list or np.ndarray
            Time series data for each array.
        """
        # What data should we return?
        if prepared:
            memmaps = self.arrays
        else:
            memmaps = self.raw_data_memmaps

        # Should we return one long time series?
        if concatenate or self.n_arrays == 1:
            return np.concatenate(memmaps)
        else:
            return memmaps

    def delete_dir(self):
        """Deletes store_dir."""
        if self.store_dir.exists():
            rmtree(self.store_dir)

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
        # self.raw_data_filenames is not used if self.inputs is a list of strings,
        # where the strings are paths to .npy files

        # Load data
        partial_make_memmap = partial(self.make_memmap)
        args = zip(self.inputs, raw_data_filenames)
        memmaps = pqdm(
            args,
            partial_make_memmap,
            n_jobs=self.n_jobs,
            desc="Loading files",
            argument_type="args",
            total=len(self.inputs),
        )

        return memmaps, raw_data_filenames

    def make_memmap(self, raw_data, mmap_location):
        """Make a memory map for a single file.

        Parameters
        ----------
        raw_data : str
            Path to file.
        mmap_location : str
            Path to save memory map to.

        Returns
        -------
        raw_data_mmap: np.memmap
            Memory map of the raw data.
        """
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

    def save(self, output_dir="."):
        """Saves data to numpy files.

        Parameters
        ----------
        output_dir : str
            Path to save data files to. Default is the current working directory.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, array_data in enumerate(tqdm(self.arrays, desc="Saving data")):
            padded_number = misc.leading_zeros(i, self.n_arrays)
            np.save(f"{output_dir}/array{padded_number}.npy", array_data)

    def validate_data(self):
        """Validate data files."""
        n_channels = [memmap.shape[-1] for memmap in self.raw_data_memmaps]
        if not np.equal(n_channels, n_channels[0]).all():
            raise ValueError("All inputs should have the same number of channels.")

    def filter(self, low_freq=None, high_freq=None):
        """Filter the data.

        This is an in-place operation.

        Parameters
        ----------
        low_freq : float
            Frequency in Hz for a high pass filter.
        high_freq : float
            Frequency in Hz for a low pass filter.
        """
        if (
            low_freq is not None or high_freq is not None
        ) and self.sampling_frequency is None:
            raise ValueError(
                "Data.sampling_frequency must be set if we are filtering the data. "
                + "Use Data.set_sampling_frequency() or pass "
                + "Data(..., sampling_frequency=...) when creating the Data object."
            )

        self.low_freq = low_freq
        self.high_freq = high_freq

        # Create filenames for memmaps (i.e. self.prepared_data_filenames)
        self.prepare_memmap_filenames()

        # Function to apply filtering to a single array
        def _apply(memmap, prepared_data_file):
            prepared_data = processing.temporal_filter(
                memmap, self.low_freq, self.high_freq, self.sampling_frequency
            )
            if self.load_memmaps:
                prepared_data_memmap = misc.array_to_memmap(
                    prepared_data_file, prepared_data
                )
            else:
                prepared_data_memmap = prepared_data
            return prepared_data_memmap

        # Prepare the data in parallel
        args = zip(self.arrays, self.prepared_data_filenames)
        prepared_data_memmaps = pqdm(
            args,
            _apply,
            desc="Filtering",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_arrays,
        )
        self.prepared_data_memmaps.extend(prepared_data_memmaps)

        # Update arrays to return the prepared data
        self.arrays = self.prepared_data_memmaps

    def tde_pca(
        self,
        n_embeddings,
        n_pca_components=None,
        pca_components=None,
        whiten=False,
    ):
        """Time-delay embedding (TDE) and principal component analysis (PCA).

        This function will first standardize the data, then perform TDE then PCA.
        This is an in-place operation.

        Parameters
        ----------
        n_embeddings : int
            Number of data points to embed the data.
        n_pca_components : int
            Number of PCA components to keep. Default is no PCA.
        pca_components : np.ndarray
            PCA components to apply if they have already been calculated.
        whiten : bool
            Should we whiten the PCA'ed data?
        """
        if n_pca_components is not None and pca_components is not None:
            raise ValueError("Please only pass n_pca_components or pca_components.")

        if pca_components is not None and not isinstance(pca_components, np.ndarray):
            raise ValueError("pca_components must be a numpy array.")

        self.n_embeddings = n_embeddings
        self.n_te_channels = self.n_raw_data_channels * n_embeddings
        self.n_pca_components = n_pca_components
        self.pca_components = pca_components
        self.whiten = whiten

        # Create filenames for memmaps (i.e. self.prepared_data_filenames)
        self.prepare_memmap_filenames()

        # Calculate PCA on TDE data
        # NOTE: the approach used here only works for zero mean data
        if n_pca_components is not None:
            # Calculate covariance of the data
            covariance = np.zeros([self.n_te_channels, self.n_te_channels])
            for memmap in tqdm(self.arrays, desc="Calculating PCA components"):
                std_data = processing.standardize(memmap)
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
        def _apply(memmap, prepared_data_file):
            std_data = processing.standardize(memmap)
            te_std_data = processing.time_embed(std_data, self.n_embeddings)
            if self.pca_components is not None:
                prepared_data = te_std_data @ self.pca_components
            else:
                prepared_data = te_std_data
            if self.load_memmaps:
                prepared_data_memmap = misc.array_to_memmap(
                    prepared_data_file, prepared_data
                )
            else:
                prepared_data_memmap = prepared_data
            return prepared_data_memmap

        # Apply TDE and PCA in parallel
        args = zip(self.arrays, self.prepared_data_filenames)
        prepared_data_memmaps = pqdm(
            args,
            _apply,
            desc="TDE-PCA",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_arrays,
        )
        self.prepared_data_memmaps.extend(prepared_data_memmaps)

        # Update arrays to return the prepared data
        self.arrays = self.prepared_data_memmaps

    def amp_env(self):
        """Calculate the amplitude envelope.

        This is an in-place operation.
        """

        # Create filenames for memmaps (i.e. self.prepared_data_filenames)
        self.prepare_memmap_filenames()

        # Function to calculate amplitude envelope for a single array
        def _apply_amp_env(memmap, prepared_data_file):
            prepared_data = np.abs(signal.hilbert(memmap, axis=0))
            prepared_data = prepared_data.astype(np.float32)
            if self.load_memmaps:
                prepared_data_memmap = misc.array_to_memmap(
                    prepared_data_file, prepared_data
                )
            else:
                prepared_data_memmap = prepared_data
            return prepared_data_memmap

        # Prepare the data in parallel
        args = zip(self.arrays, self.prepared_data_filenames)
        prepared_data_memmaps = pqdm(
            args,
            _apply_amp_env,
            desc="Amplitude envelope",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_arrays,
        )
        self.prepared_data_memmaps.extend(prepared_data_memmaps)

        # Update arrays to return the prepared data
        self.arrays = self.prepared_data_memmaps

    def sliding_window(self, n_window):
        """Apply a sliding window.

        This is an in-place operation.

        Parameters
        ----------
        n_window : int
            Number of data points in the sliding window. Must be odd.
        """
        if n_window % 2 == 0:
            raise ValueError("n_window must be odd.")

        self.n_window = n_window

        # Function to apply sliding window to a single array
        def _apply(memmap, prepared_data_file):
            prepared_data = np.array(
                [
                    np.convolve(
                        memmap[:, i],
                        np.ones(self.n_window) / self.n_window,
                        mode="valid",
                    )
                    for i in range(memmap.shape[1])
                ],
            ).T
            prepared_data = prepared_data.astype(np.float32)
            if self.load_memmaps:
                prepared_data_memmap = misc.array_to_memmap(
                    prepared_data_file, prepared_data
                )
            else:
                prepared_data_memmap = prepared_data
            return prepared_data_memmap

        # Prepare the data in parallel
        args = zip(self.arrays, self.prepared_data_filenames)
        prepared_data_memmaps = pqdm(
            args,
            _apply,
            desc="Sliding window",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_arrays,
        )
        self.prepared_data_memmaps.extend(prepared_data_memmaps)

        # Update arrays to return the prepared data
        self.arrays = self.prepared_data_memmaps

    def standardize(self):
        """Standardize (z-transform) the data.

        This is an in-place operation.
        """

        # Function to apply standardisation to a single array
        def _apply(memmap):
            processing.standardize(memmap, create_copy=False)

        # Apply standardisation to each array in parallel
        pqdm(
            zip(self.arrays),
            _apply,
            desc="Standardization",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=self.n_arrays,
        )

    def prepare_memmap_filenames(self):
        prepared_data_pattern = "prepared_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(self.n_arrays)), identifier=self._identifier
        )
        self.prepared_data_filenames = [
            str(self.store_dir / prepared_data_pattern.format(i=i))
            for i in range(self.n_arrays)
        ]

        self.prepared_data_memmaps = []

    def trim_time_series(
        self,
        sequence_length=None,
        n_embeddings=1,
        n_window=1,
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
            Number of data points used to embed the data.
        n_window : int
            Number of data points the sliding window applied to the data.
        prepared : bool
            Should we return the prepared data? If not we return the raw data.
        concatenate : bool
            Should we concatenate the data for each array?

        Returns
        -------
        list of np.ndarray
            Trimed time series for each array.
        """
        if self.n_embeddings is None and self.n_window is None:
            # Data has not been prepared so we can't trim the prepared data
            prepared = False

        n_remove = 0
        if not prepared:
            # We're trimming the raw data, how many data points do we
            # need to remove due to time embedding or moving average?
            n_embeddings = n_embeddings or self.n_embeddings
            n_window = n_window or self.n_window
            if n_embeddings is not None:
                n_remove += n_embeddings
            if n_window is not None:
                n_remove += n_window

        # What data should we trim?
        if prepared:
            memmaps = self.arrays
        else:
            memmaps = self.raw_data_memmaps

        trimmed_time_series = []
        for memmap in memmaps:
            # Remove data points lost to time embedding
            if n_remove != 0:
                memmap = memmap[n_remove // 2 : -(n_remove // 2)]

            # Remove data points lost to separating into sequences
            if sequence_length is not None:
                n_sequences = memmap.shape[0] // sequence_length
                memmap = memmap[: n_sequences * sequence_length]

            trimmed_time_series.append(memmap)

        if concatenate or len(trimmed_time_series) == 1:
            trimmed_time_series = np.concatenate(trimmed_time_series)

        return trimmed_time_series

    def count_batches(self, sequence_length):
        """Count batches.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.

        Returns
        -------
        n : np.ndarray
            Number of batches for each array's data.
        """
        return np.array(
            [tf.n_batches(memmap, sequence_length) for memmap in self.arrays]
        )

    def dataset(
        self,
        sequence_length,
        batch_size,
        shuffle=True,
        validation_split=None,
        concatenate=True,
        subj_id=False,
        step_size=None,
    ):
        """Create a tensorflow dataset for training or evaluation.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the model.
        shuffle : bool
            Should we shuffle sequences (within a batch) and batches.
        validation_split : float
            Ratio to split the dataset into a training and validation set.
        concatenate : bool
            Should we concatenate the datasets for each array? Optional, default
            is True.
        subj_id : bool
            Should we include the subject id in the dataset? Optional, default is
            False. This argument can be used to prepare datasets for subject-specific
            models.
        step_size : int
            Number of samples to slide the sequence across the dataset. Optional.
            Default is no overlap.

        Returns
        -------
        tensorflow.data.Dataset or Tuple
            Dataset for training or evaluating the model along with the validation
            set if validation_split was passed.
        """
        self.n_batches = self.count_batches(sequence_length)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.step_size = step_size or sequence_length

        datasets = []
        for i in range(self.n_arrays):
            if i not in self.keep:
                # We don't want to include this file in the dataset
                continue

            # Get time series data
            array = self.arrays[i]

            if subj_id:
                # Create a dataset with the time series data and ID
                array_tracker = np.zeros(array.shape[0], dtype=np.float32) + i
                dataset = tf.create_dataset(
                    {"data": array, "subj_id": array_tracker},
                    self.sequence_length,
                    self.step_size,
                )
            else:
                # Createa a dataset with just the time series data
                dataset = tf.create_dataset(
                    {"data": array}, self.sequence_length, self.step_size
                )

            datasets.append(dataset)

        # Create a dataset from all the arrays concatenated
        if concatenate:
            full_dataset = tf.concatenate_datasets(datasets, shuffle=False)

            if shuffle:
                # Shuffle sequences
                full_dataset = full_dataset.shuffle(100000)

                # Group into mini-batches
                full_dataset = full_dataset.batch(batch_size)

                # Shuffle mini-batches
                full_dataset = full_dataset.shuffle(100000)

            else:
                # Group into mini-batches
                full_dataset = full_dataset.batch(batch_size)

            if validation_split is None:
                # Return the full dataset
                return full_dataset.prefetch(-1)

            else:
                # Calculate how many batches should be in the training dataset
                dataset_size = len(full_dataset)
                training_dataset_size = round((1.0 - validation_split) * dataset_size)

                # Split the full dataset into a training and validation dataset
                training_dataset = full_dataset.take(training_dataset_size)
                validation_dataset = full_dataset.skip(training_dataset_size)
                _logger.info(
                    f"{len(training_dataset)} batches in training dataset, "
                    + f"{len(validation_dataset)} batches in the validation dataset."
                )

                return training_dataset.prefetch(-1), validation_dataset.prefetch(-1)

        # Otherwise create a dataset for each array separately
        else:
            full_datasets = []
            for ds in datasets:
                if shuffle:
                    # Shuffle sequences
                    ds = ds.shuffle(100000)

                # Group into batches
                ds = ds.batch(batch_size)

                if shuffle:
                    # Shuffle batches
                    ds = ds.shuffle(100000)

                full_datasets.append(ds.prefetch(-1))

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
                        f"Subject {i}: "
                        + f"{len(training_datasets[i])} batches in training dataset, "
                        + f"{len(validation_datasets[i])} batches in the validation dataset."
                    )
                return training_datasets, validation_datasets
