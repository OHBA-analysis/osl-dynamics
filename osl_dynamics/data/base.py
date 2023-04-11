"""Base class for handling data.

"""

import logging
import pathlib
import pickle
import warnings
from functools import partial
from os import path
from shutil import rmtree

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
    n_embeddings : int
        Number of time-delay embeddings that have already been appleid to the data.
        This argument is optional. It is useful to pass this argument if the data has
        already been prepared.
    n_window : int
        Length of sliding window that has already been applied to the data. This
        argument is optional. It is useful to pass this argument if the data has
        already been prepared.
    amplitude_envelope : bool
        Is the data we're loading amplitude envelope data? This argument is optional.
        It is useful to pass this argument if the data has already been prepared.
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
        n_embeddings=None,
        n_window=None,
        amplitude_envelope=None,
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
        self.n_embeddings = n_embeddings
        self.n_window = n_window
        self.amplitude_envelope = amplitude_envelope
        self.time_axis_first = time_axis_first
        self.load_memmaps = load_memmaps
        self.n_jobs = n_jobs
        self.prepared = False
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

        # Get data prepration attributes if the raw data has been prepared
        if not isinstance(inputs, list):
            self.load_preparation(inputs)

        self.n_raw_data_channels = self.raw_data_memmaps[0].shape[-1]

        # Use raw data for the subject data
        self.subjects = self.raw_data_memmaps

    def __iter__(self):
        return iter(self.subjects)

    def __getitem__(self, item):
        return self.subjects[item]

    def __str__(self):
        info = [
            f"{self.__class__.__name__}",
            f"id: {self._identifier}",
            f"n_subjects: {self.n_subjects}",
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
        return self.subjects[0].shape[-1]

    @property
    def n_samples(self):
        """Number of samples for each subject."""
        return sum([subject.shape[-2] for subject in self.subjects])

    @property
    def n_subjects(self):
        """Number of subjects."""
        return len(self.subjects)

    def set_sampling_frequency(self, sampling_frequency):
        """Sets the sampling_frequency attribute.

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency in Hz.
        """
        self.sampling_frequency = sampling_frequency

    def time_series(self, concatenate=False):
        """Time series data for all subjects.

        Parameters
        ----------
        concatenate : bool
            Should we return the time series for each subject concatenated?

        Returns
        -------
        ts : list or np.ndarray
            Time series data for each subject.
        """
        if concatenate or self.n_subjects == 1:
            return np.concatenate(self.subjects)
        else:
            return self.subjects

    def delete_dir(self):
        """Deletes store_dir."""
        if self.store_dir.exists():
            rmtree(self.store_dir)

    def load_preparation(self, inputs):
        """Loads a pickle file containing preparation settings.

        Parameters
        ----------
        inputs : str
            Path to directory containing the pickle file with preparation settings.
        """
        if path.isdir(inputs):
            for file in rw.list_dir(inputs):
                if "preparation.pkl" in file:
                    preparation = pickle.load(open(inputs + "/preparation.pkl", "rb"))
                    self.amplitude_envelope = preparation["amplitude_envelope"]
                    self.low_freq = preparation["low_freq"]
                    self.high_freq = preparation["high_freq"]
                    self.n_window = preparation["n_window"]
                    self.n_embeddings = preparation["n_embeddings"]
                    self.n_te_channels = preparation["n_te_channels"]
                    self.n_pca_components = preparation["n_pca_components"]
                    self.pca_components = preparation["pca_components"]
                    self.whiten = preparation["whiten"]
                    self.prepared = True

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

        # Save time series data
        for i, subject_data in enumerate(tqdm(self.subjects, desc="Saving data")):
            padded_number = misc.leading_zeros(i, self.n_subjects)
            np.save(f"{output_dir}/subject{padded_number}.npy", subject_data)

        # Save preparation info if .prepared has been called
        if self.prepared:
            preparation = {
                "amplitude_envelope": self.amplitude_envelope,
                "low_freq": self.low_freq,
                "high_freq": self.high_freq,
                "n_window": self.n_window,
                "n_embeddings": self.n_embeddings,
                "n_te_channels": self.n_te_channels,
                "n_pca_components": self.n_pca_components,
                "pca_components": self.pca_components,
                "whiten": self.whiten,
            }
            pickle.dump(preparation, open(f"{output_dir}/preparation.pkl", "wb"))

    def validate_data(self):
        """Validate data files."""
        n_channels = [memmap.shape[-1] for memmap in self.raw_data_memmaps]
        if not np.equal(n_channels, n_channels[0]).all():
            raise ValueError("All inputs should have the same number of channels.")

    def filter(self, low_freq=None, high_freq=None):
        """Filter the raw data.

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

        # Save settings
        self.amplitude_envelope = False
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.n_window = 1
        self.n_embeddings = 1
        self.n_te_channels = self.n_raw_data_channels
        self.n_pca_components = None
        self.pca_components = None
        self.whiten = None

        # Create filenames for memmaps (i.e. self.prepared_data_filenames)
        self.prepare_memmap_filenames()

        # Prepare the data
        for raw_data_memmap, prepared_data_file in zip(
            tqdm(self.raw_data_memmaps, desc="Filtering data"),
            self.prepared_data_filenames,
        ):
            # Filtering
            prepared_data = processing.temporal_filter(
                raw_data_memmap, low_freq, high_freq, self.sampling_frequency
            )

            if self.load_memmaps:
                # Save the prepared data as a memmap
                prepared_data_memmap = misc.array_to_memmap(
                    prepared_data_file, prepared_data
                )
            else:
                prepared_data_memmap = prepared_data
            self.prepared_data_memmaps.append(prepared_data_memmap)

        # Update subjects to return the prepared data
        self.subjects = self.prepared_data_memmaps

        self.prepared = True

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
        """
        if self.prepared:
            warnings.warn(
                "Previously prepared data will be overwritten.", RuntimeWarning
            )

        # Prepare data (either amplitude envelope or time-delay embedded)
        if amplitude_envelope:
            self.prepare_amp_env(low_freq, high_freq, n_window)
        else:
            self.prepare_tde(n_embeddings, n_pca_components, pca_components, whiten)

    def prepare_amp_env(self, low_freq=None, high_freq=None, n_window=1):
        """Prepare amplitude envelope data.

        Parameters
        ----------
        low_freq : float
            Frequency in Hz for a high pass filter.
        high_freq : float
            Frequency in Hz for a low pass filter.
        n_window : int
            Number of data points in a sliding window to apply to the amplitude
            envelope data.
        """

        # Validation
        if (
            low_freq is not None or high_freq is not None
        ) and self.sampling_frequency is None:
            raise ValueError(
                "Data.sampling_frequency must be set if we are filtering the data. "
                + "Use Data.set_sampling_frequency() or pass "
                + "Data(..., sampling_frequency=...) when creating the Data object."
            )

        if n_window % 2 == 0:
            raise ValueError("n_window must be an odd number.")

        # Save settings
        self.amplitude_envelope = True
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.n_window = n_window
        self.n_embeddings = 1
        self.n_te_channels = self.n_raw_data_channels
        self.n_pca_components = None
        self.pca_components = None
        self.whiten = None

        # Create filenames for memmaps (i.e. self.prepared_data_filenames)
        self.prepare_memmap_filenames()

        # Prepare the data
        prepare_args = zip(
            self.raw_data_memmaps,
            self.prepared_data_filenames,
        )

        prepared_data_memmaps = pqdm(
            prepare_args,
            self.apply_amp_env,
            desc="Preparing data",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=len(self.raw_data_memmaps),
        )
        self.prepared_data_memmaps.extend(prepared_data_memmaps)

        # Update subjects to return the prepared data
        self.subjects = self.prepared_data_memmaps

        self.prepared = True

    def apply_amp_env(self, raw_data_memmap, prepared_data_file):
        """Applies filtering, a Hilbert transform and standardization to raw data.

        Parameters
        ----------
        raw_data_memmap : np.memmap or np.ndarray
            Raw data.
        prepared_data_file : str
            Name of memory map file to save prepared data to.
            Can be None if we are not using memory maps.

        Returns
        -------
        prepared_data_memmap : np.memmap or np.ndarray
            Prepared data.
        """
        # Filtering
        prepared_data = processing.temporal_filter(
            raw_data_memmap, self.low_freq, self.high_freq, self.sampling_frequency
        )

        # Hilbert transform
        prepared_data = np.abs(signal.hilbert(prepared_data, axis=0))

        # Moving average filter
        prepared_data = np.array(
            [
                np.convolve(
                    prepared_data[:, i],
                    np.ones(self.n_window) / self.n_window,
                    mode="valid",
                )
                for i in range(prepared_data.shape[1])
            ],
        ).T

        # Finally, we standardise
        prepared_data = processing.standardize(prepared_data, create_copy=False)

        # Make sure data is float32
        prepared_data = prepared_data.astype(np.float32)

        if self.load_memmaps:
            # Save the prepared data as a memmap
            prepared_data_memmap = misc.array_to_memmap(
                prepared_data_file, prepared_data
            )
        else:
            prepared_data_memmap = prepared_data
        return prepared_data_memmap

    def prepare_tde(
        self,
        n_embeddings=1,
        n_pca_components=None,
        pca_components=None,
        whiten=False,
    ):
        """Prepares time-delay embedded data to train the model with.

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

        # Save settings
        self.amplitude_envelope = False
        self.low_freq = None
        self.high_freq = None
        self.n_window = 1
        self.n_embeddings = n_embeddings
        self.n_te_channels = self.n_raw_data_channels * n_embeddings
        self.n_pca_components = n_pca_components
        self.pca_components = pca_components
        self.whiten = whiten

        # Create filenames for memmaps (i.e. self.prepared_data_filenames)
        self.prepare_memmap_filenames()

        # Principle component analysis (PCA)
        # NOTE: the approach used here only works for zero mean data
        if n_pca_components is not None:
            # Calculate the PCA components by performing SVD on the covariance
            # of the data
            covariance = np.zeros([self.n_te_channels, self.n_te_channels])
            for raw_data_memmap in tqdm(
                self.raw_data_memmaps, desc="Calculating PCA components"
            ):
                # Standardise and time embed the data
                std_data = processing.standardize(raw_data_memmap)
                te_std_data = processing.time_embed(std_data, n_embeddings)

                # Calculate the covariance of the entire dataset
                covariance += np.transpose(te_std_data) @ te_std_data

            # Use SVD to calculate PCA components
            u, s, vh = np.linalg.svd(covariance)
            u = u[:, :n_pca_components].astype(np.float32)
            explained_variance = np.sum(s[:n_pca_components]) / np.sum(s)
            _logger.info(f"Explained variance: {100 * explained_variance:.1f}%")
            s = s[:n_pca_components].astype(np.float32)
            if whiten:
                u = u @ np.diag(1.0 / np.sqrt(s))
            self.pca_components = u

        prepare_args = zip(
            self.raw_data_memmaps,
            self.prepared_data_filenames,
        )

        prepared_data_memmaps = pqdm(
            prepare_args,
            self.apply_tde_pca,
            desc="Preparing data",
            n_jobs=self.n_jobs,
            argument_type="args",
            total=len(self.raw_data_memmaps),
        )
        self.prepared_data_memmaps.extend(prepared_data_memmaps)

        # Update subjects to return the prepared data
        self.subjects = self.prepared_data_memmaps

        self.prepared = True

    def apply_tde_pca(self, raw_data_memmap, prepared_data_file):
        """Applies time-delay embeddings, principal component analysis and
        standardization to raw data.

        Parameters
        ----------
        raw_data_memmap : np.memmap or np.ndarray
            Raw data.
        prepared_data_file : str
            Name of memory map file to save prepared data to.
            Can be None if we are not using memory maps.

        Returns
        -------
        prepared_data_memmap : np.memmap or np.ndarray
            Prepared data.
        """

        # Standardise and time embed the data
        std_data = processing.standardize(raw_data_memmap)
        te_std_data = processing.time_embed(std_data, self.n_embeddings)

        # Apply PCA to get the prepared data
        if self.pca_components is not None:
            prepared_data = te_std_data @ self.pca_components

        # Otherwise, the time embedded data is the prepared data
        else:
            prepared_data = te_std_data

        # Finally, we standardise
        prepared_data = processing.standardize(prepared_data, create_copy=False)

        if self.load_memmaps:
            # Save the prepared data as a memmap
            prepared_data_memmap = misc.array_to_memmap(
                prepared_data_file, prepared_data
            )
        else:
            prepared_data_memmap = prepared_data

        return prepared_data_memmap

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
            Should we concatenate the data for each subject?

        Returns
        -------
        list of np.ndarray
            Trimed time series for each subject.
        """
        if self.n_embeddings is None and self.n_window is None:
            # Data has not been prepared so we can't trim the prepared data
            prepared = False

        if not prepared:
            # We're trimming the raw data, how many data points do we
            # need to remove due to time embedding or moving average?
            if self.amplitude_envelope:
                n_remove = self.n_window or n_window
            else:
                n_remove = self.n_embeddings or n_embeddings
        else:
            n_remove = 1

        # What data should we trim?
        if prepared:
            memmaps = self.subjects
        else:
            memmaps = self.raw_data_memmaps

        trimmed_time_series = []
        for memmap in memmaps:
            # Remove data points lost to time embedding
            if n_remove != 1:
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
            Number of batches for each subject's data.
        """
        return np.array(
            [tf.n_batches(memmap, sequence_length) for memmap in self.subjects]
        )

    def dataset(
        self,
        sequence_length,
        batch_size,
        shuffle=True,
        validation_split=None,
        alpha=None,
        gamma=None,
        n_alpha_embeddings=1,
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
        alpha : list of np.ndarray
            List of mode mixing factors for each subject.
            If passed, we create a dataset that includes alpha at each time point.
            Optional. Such a dataset is used to train an observation model.
        gamma : list of np.ndarray
            List of mode mixing factors for the functional connectivity.
            Optional. Used with a multi-dynamic model when training the observation
            model only.
        n_alpha_embeddings : int
            Number of embeddings used when inferring alpha. Optional. Only should be
            used if passing alpha (or gamma).
        concatenate : bool
            Should we concatenate the datasets for each subject? Optional, default
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
        n_embeddings = self.n_embeddings or 1

        # Dataset for learning alpha and the observation model
        if alpha is None:
            subject_datasets = []
            for i in range(self.n_subjects):
                subject = self.subjects[i]
                if subj_id:
                    subject_tracker = np.zeros(subject.shape[0], dtype=np.float32) + i
                    dataset = tf.create_dataset(
                        {"data": subject, "subj_id": subject_tracker},
                        self.sequence_length,
                        self.step_size,
                    )
                else:
                    dataset = tf.create_dataset(
                        {"data": subject}, self.sequence_length, self.step_size
                    )
                subject_datasets.append(dataset)

        # Dataset for learning the observation model
        else:
            if not isinstance(alpha, list):
                raise ValueError("alpha must be a list of numpy arrays.")

            subject_datasets = []
            for i in range(self.n_subjects):
                if n_embeddings > n_alpha_embeddings:
                    # We remove data points in alpha that are not in the new time
                    # embedded data
                    alp = alpha[i][(n_embeddings - n_alpha_embeddings) // 2 :]
                    if gamma is not None:
                        gam = gamma[i][(n_embeddings - n_alpha_embeddings) // 2 :]
                    subject = self.subjects[i][: alp.shape[0]]

                else:
                    # We remove the data points that are not in alpha
                    alp = alpha[i]
                    if gamma is not None:
                        gam = gamma[i]
                    subject = self.subjects[i][
                        (n_alpha_embeddings - n_embeddings) // 2 : alp.shape[0]
                    ]

                # Create dataset
                input_data = {"data": subject, "alpha": alp}
                if gamma is not None:
                    input_data["gamma"] = gam
                if subj_id:
                    input_data["subj_id"] = (
                        np.zeros(subject.shape[0], dtype=np.float32) + i
                    )
                dataset = tf.create_dataset(
                    input_data, self.sequence_length, self.step_size
                )
                subject_datasets.append(dataset)

        # Create a dataset from all the subjects concatenated
        if concatenate:
            full_dataset = tf.concatenate_datasets(subject_datasets, shuffle=False)

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

        # Otherwise create a dataset for each subject separately
        else:
            full_datasets = []
            for ds in subject_datasets:
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
                # Return the full dataset for each subject
                return full_datasets

            else:
                # Split the dataset for each subject separately
                training_datasets = []
                validation_datasets = []
                for i in range(len(full_datasets)):
                    # Calculate the number of batches in the training dataset
                    dataset_size = len(full_datasets[i])
                    training_dataset_size = round(
                        (1.0 - validation_split) * dataset_size
                    )

                    # Split this subject's dataset
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
