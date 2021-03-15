import pathlib
import logging
from shutil import rmtree
from typing import List, Union

import numpy as np
import tensorflow
import yaml
from sklearn.cluster import KMeans
from tensorflow.python.data import Dataset
from tqdm import tqdm

from vrad.analysis import spectral
from vrad.data import io, manipulation
from vrad.utils import misc

_logger = logging.getLogger("VRAD")
_rng = np.random.default_rng()


class Data:
    """Data Class.

    The Data class enables the input and processing of data. When given a list of
    files, it produces a set of numpy memory maps which contain their raw data.
    It also provides methods for batching data and creating TensorFlow Datasets.

    Parameters
    ----------
    inputs : list of str or str
        Filenames to be read.
    sampling_frequency : float
        Sampling frequency of the data in Hz. Optional.
    store_dir : str
        Directory to save results and intermediate steps to. Optional, default is /tmp.
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
        inputs: list,
        sampling_frequency: float = None,
        store_dir: str = "tmp",
        n_embeddings: int = 0,
        n_pca_components: int = None,
        whiten: bool = None,
        prepared: bool = False,
    ):
        # Unique identifier for the data object
        self._identifier = id(inputs)

        # Validate inputs
        if isinstance(inputs, str):
            self.inputs = [inputs]
        elif isinstance(inputs, np.ndarray):
            if inputs.ndim == 2:
                self.inputs = [inputs]
            else:
                self.inputs = inputs
        elif isinstance(inputs, list):
            self.inputs = inputs
        else:
            raise ValueError(
                f"inputs must be str, np.ndarray or list, got {type(inputs)}."
            )

        self.store_dir = pathlib.Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_pattern = "raw_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(len(inputs))), identifier=self._identifier
        )

        # Raw data memory maps
        self.raw_data_filenames = [
            str(self.store_dir / self.raw_data_pattern.format(i=i))
            for i, _ in enumerate(inputs)
        ]

        # Load the data
        self.raw_data_memmaps = self.load_data()
        self.subjects = self.raw_data_memmaps

        # Raw data statistics
        self.raw_data_mean = [
            np.mean(raw_data, axis=0) for raw_data in self.raw_data_memmaps
        ]
        self.raw_data_std = [
            np.std(raw_data, axis=0) for raw_data in self.raw_data_memmaps
        ]

        # Other attributes
        self.sampling_frequency = sampling_frequency
        self.n_raw_data_channels = self.n_channels
        self.n_embeddings = n_embeddings
        self.n_pca_components = n_pca_components
        self.whiten = whiten
        self.prepared = prepared

        # Validation
        self.validate_subjects()

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
            f"prepared: {self.prepared}",
        ]
        return "\n ".join(info)

    @property
    def raw_data(self) -> List:
        """Return raw data as a list of arrays."""
        return self.raw_data_memmaps

    @property
    def n_channels(self) -> int:
        """Number of channels in the data files."""
        return self.subjects[0].shape[1]

    @property
    def n_samples(self) -> list:
        """Number of samples for each subject"""
        return [subject.shape[0] for subject in self.subjects]

    @property
    def n_subjects(self) -> int:
        """Number of subjects."""
        return len(self.subjects)

    @classmethod
    def from_yaml(cls, file, **kwargs):
        instance = misc.class_from_yaml(cls, file, kwargs)

        with open(file) as f:
            settings = yaml.load(f, Loader=yaml.Loader)

        if issubclass(cls, Data):
            try:
                cls._process_from_yaml(instance, file, **kwargs)
            except AttributeError:
                pass

        training_dataset = instance.training_dataset(
            sequence_length=settings["sequence_length"],
            batch_size=settings["batch_size"],
        )
        prediction_dataset = instance.prediction_dataset(
            sequence_length=settings["sequence_length"],
            batch_size=settings["batch_size"],
        )

        return {
            "data": instance,
            "training_dataset": training_dataset,
            "prediction_dataset": prediction_dataset,
        }

    def _pre_pca(self, raw_data, filter_range, filter_order, n_embeddings):
        std_data = manipulation.standardize(raw_data)
        if filter_range is not None:
            f_std_data = manipulation.bandpass_filter(
                std_data, filter_range, filter_order, self.sampling_frequency
            )
        else:
            f_std_data = std_data
        te_f_std_data = manipulation.time_embed(f_std_data, n_embeddings)
        return te_f_std_data

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

    def autocorrelation_functions(
        self,
        covariances: Union[list, np.ndarray],
        reverse_standardization: bool = False,
        subject_index: int = None,
    ) -> np.ndarray:
        """Calculates the autocorrelation function the state covariance matrices.

        An autocorrelation function is calculated for each state for each subject.

        Parameters
        ----------
        covariances : np.ndarray
            State covariance matrices.
            Shape is (n_subjects, n_states, n_channels, n_channels).
            These must be subject specific covariances.
        reverse_standardization : bool
            Should we reverse the standardization performed on the dataset?
            Optional, the default is False.
        subject_index : int
            Index for the subject if the covariances corresponds to a single
            subject. Optional. Only used if reverse_standardization is True.

        Returns
        -------
        np.ndarray
            Autocorrelation function.
            Shape is (n_subjects, n_states, n_channels, n_channels, n_acf)
            or (n_states, n_channels, n_channels, n_acf).
        """
        # Validation
        if self.n_embeddings <= 1:
            raise ValueError(
                "To calculate an autocorrelation function we have to train on time "
                + "embedded data."
            )

        if isinstance(covariances, np.ndarray):
            if covariances.ndim != 3:
                raise ValueError(
                    "covariances must be shape (n_states, n_channels, n_channels) or"
                    + " (n_subjects, n_states, n_channels, n_channels)."
                )
            covariances = [covariances]

        if not isinstance(covariances, list):
            raise ValueError(
                "covariances must be a list of numpy arrays or a numpy array."
            )

        n_subjects = len(covariances)
        n_states = covariances[0].shape[0]

        autocorrelation_function = []
        for n in range(n_subjects):
            if reverse_standardization:
                for i in range(n_states):
                    # Get the standard deviation of the prepared data
                    if subject_index is None:
                        prepared_data_std = self.prepared_data_std[n]
                    else:
                        prepared_data_std = self.prepared_data_std[subject_index]

                    # Reverse the standardisation
                    covariances[n][i] = (
                        np.diag(prepared_data_std)
                        @ covariances[n][i]
                        @ np.diag(prepared_data_std)
                    )

            # Reverse the PCA
            te_cov = []
            for i in range(n_states):
                te_cov.append(self.pca_weights @ covariances[n][i] @ self.pca_weights.T)
            te_cov = np.array(te_cov)

            if reverse_standardization:
                for i in range(n_states):
                    # Get the standard deviation of the raw data
                    if subject_index is None:
                        raw_data_std = self.raw_data_std[n]
                    else:
                        raw_data_std = self.raw_data_std[subject_index]

                    # Reverse the standardisation
                    te_cov[i] = (
                        np.diag(np.repeat(raw_data_std, self.n_embeddings))
                        @ te_cov[i]
                        @ np.diag(np.repeat(raw_data_std, self.n_embeddings))
                    )

            # Calculate the autocorrelation function
            autocorrelation_function.append(
                spectral.autocorrelation_function(
                    te_cov, self.n_embeddings, self.n_raw_data_channels
                )
            )

        return np.squeeze(autocorrelation_function)

    def count_batches(self, sequence_length: int) -> np.ndarray:
        """Count batches.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.

        Returns
        -------
        np.ndarray
            Number of batches for each subject's data.
        """
        return np.array(
            [
                manipulation.n_batches(memmap, sequence_length)
                for memmap in self.subjects
            ]
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

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flat_covariances)

        kmeans_covariances = kmeans.cluster_centers_.reshape(
            (n_clusters, *covariances.shape[1:])
        )

        return kmeans_covariances

    def delete_dir(self):
        """Deletes the directory that stores the memory maps."""
        rmtree(self.store_dir, ignore_errors=True)

    def load_data(self):
        """Import data into a list of memory maps."""
        memmaps = []
        for in_file, out_file in zip(
            tqdm(self.inputs, desc="Loading files", ncols=98), self.raw_data_filenames
        ):
            data = io.load_data(in_file, mmap_location=out_file)
            memmaps.append(data)
        return memmaps

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
            prepared_data_memmap = misc.MockArray.get_memmap(
                prepared_data_file, prepared_data.shape, dtype=np.float32
            )

            # Record the mean and standard deviation of the prepared
            # data and standardise to get the final data
            self.prepared_data_mean.append(np.mean(prepared_data, axis=0))
            self.prepared_data_std.append(np.std(prepared_data, axis=0))
            prepared_data_memmap = manipulation.standardize(
                prepared_data, create_copy=False
            )
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
            n_embddings = self.n_embeddings

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

    def raw_covariances(
        self,
        state_covariances: Union[list, np.ndarray],
        reverse_standardization: bool = False,
        subject_index: int = None,
    ) -> np.ndarray:
        """Covariance matrix of the raw channels.

        PCA and standardization is reversed to give you to the covariance
        matrix for the raw channels.

        Parameters
        ----------
        state_covariances : np.ndarray
            State covariance matrices.
            Shape is (n_subjects, n_states, n_channels, n_channels).
            These must be subject specific covariances.
        reverse_standardization : bool
            Should we reverse the standardization performed on the dataset?
            Optional, the default is False.
        subject_index : int
            Index for the subject if the covariances corresponds to a single
            subject. Optional. Only used if reverse_standardization is True.

        Returns
        -------
        np.ndarray
            The variance for each channel, state and subject.
            Shape is (n_subjects, n_states, n_channels, n_channels) or
            (n_states, n_channels, n_channels).
        """
        # Validation
        if self.n_embeddings <= 1:
            raise ValueError(
                "To calculate an autocorrelation function we have to train on "
                + "time embedded data."
            )

        if isinstance(state_covariances, np.ndarray):
            if state_covariances.ndim != 3:
                raise ValueError(
                    "state_covariances must be shape (n_states, n_channels, n_channels)"
                    + " or (n_subjects, n_states, n_channels, n_channels)."
                )
            state_covariances = [state_covariances]

        if not isinstance(state_covariances, list):
            raise ValueError(
                "state_covariances must be a list of numpy arrays or a numpy array."
            )

        n_subjects = len(state_covariances)
        n_states = state_covariances[0].shape[0]

        raw_covariances = []
        for n in range(n_subjects):
            if reverse_standardization:
                for i in range(n_states):
                    # Get the standard deviation of the prepared data
                    if subject_index is None:
                        prepared_data_std = self.prepared_data_std[n]
                    else:
                        prepared_data_std = self.prepared_data_std[subject_index]

                    # Reverse the standardisation
                    state_covariances[n][i] = (
                        np.diag(prepared_data_std)
                        @ state_covariances[n][i]
                        @ np.diag(prepared_data_std)
                    )

            # Reverse the PCA
            te_cov = []
            for i in range(n_states):
                te_cov.append(
                    self.pca_weights @ state_covariances[n][i] @ self.pca_weights.T
                )
            te_cov = np.array(te_cov)

            if reverse_standardization:
                for i in range(n_states):
                    # Get the standard deviation of the raw data
                    if subject_index is None:
                        raw_data_std = self.raw_data_std[n]
                    else:
                        raw_data_std = self.raw_data_std[subject_index]

                    # Reverse the standardisation
                    te_cov[i] = (
                        np.diag(np.repeat(raw_data_std, self.n_embeddings))
                        @ te_cov[i]
                        @ np.diag(np.repeat(raw_data_std, self.n_embeddings))
                    )

            # Get the raw data covariance
            raw_covariances.append(
                te_cov[
                    :,
                    self.n_embeddings // 2 :: self.n_embeddings,
                    self.n_embeddings // 2 :: self.n_embeddings,
                ]
            )

        return np.squeeze(raw_covariances)

    def training_dataset(
        self,
        sequence_length: int,
        batch_size: int,
        alpha: list = None,
        n_alpha_embeddings: int = 0,
        concatenate: bool = True,
    ) -> tensorflow.data.Dataset:
        """Create a tensorflow dataset for training.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the model.
        alpha : list of np.ndarray
            List of state mixing factors for each subject. Optional.
            If passed, we create a dataset that includes alpha at each time point.
            Such a dataset can be used to train the observation model.
        n_alpha_embeddings : int
            Number of embeddings when inferring alpha_t. Optional.
        concatenate : bool
            Should we concatenate the datasets for each subject? Optional, the
            default is True.

        Returns
        -------
        tensorflow.data.Dataset
            Dataset for training the model.
        """
        n_batches = self.count_batches(sequence_length)

        # Dataset for learning alpha and the observation model
        if alpha is None:
            subject_datasets = []
            for i in range(self.n_subjects):
                subject = self.subjects[i]
                subject_data = Dataset.from_tensor_slices(subject).batch(
                    sequence_length, drop_remainder=True
                )
                subject_tracker = Dataset.from_tensor_slices(
                    np.zeros(n_batches[i], dtype=np.float32) + i
                )
                # The dataset must return the input data and target
                # We use the subject id for the target
                subject_datasets.append(Dataset.zip((subject_data, subject_tracker)))

        # Dataset for learning the observation model
        else:
            if not isinstance(alpha, list):
                raise ValueError("alpha must be a list of numpy arrays.")

            subject_datasets = []
            for i in range(self.n_subjects):
                if self.n_embeddings > n_alpha_embeddings:
                    # We remove data points in alpha that are not in the new time
                    # embedded data
                    alp = alpha[i][(self.n_embeddings - n_alpha_embeddings) // 2 :]
                    subject = self.subjects[i][: alp.shape[0]]
                else:
                    # We remove the data points that are not in alpha
                    alp = alpha[i]
                    subject = self.subjects[i][
                        (n_alpha_embeddings - self.n_embeddings) // 2 : alp.shape[0]
                    ]

                # Create datasets
                alpha_data = Dataset.from_tensor_slices(alp).batch(
                    sequence_length, drop_remainder=True
                )
                subject_data = Dataset.from_tensor_slices(subject).batch(
                    sequence_length, drop_remainder=True
                )
                subject_tracker = Dataset.from_tensor_slices(
                    np.zeros(n_batches[i], dtype=np.float32) + i
                )

                # The dataset has returns two inputs to the model: data and alpha_t
                # It also returns the subject id as the target
                subject_datasets.append(
                    Dataset.zip(
                        ({"data": subject_data, "alpha_t": alpha_data}, subject_tracker)
                    )
                )

        # Create a dataset from all the subjects concatenated
        if concatenate:
            full_datasets = subject_datasets[0]
            for dataset in subject_datasets[1:]:
                full_datasets = full_datasets.concatenate(dataset)
            full_datasets = (
                full_datasets.shuffle(100000)
                .batch(batch_size)
                .shuffle(100000)
                .prefetch(-1)
            )

        # Otherwise create a dataset for each subject separately
        else:
            full_datasets = [
                dataset.shuffle(100000).batch(batch_size).shuffle(100000).prefetch(-1)
                for dataset in subject_datasets
            ]

        return full_datasets

    def prediction_dataset(self, sequence_length: int, batch_size: int) -> list:
        """Create a tensorflow dataset for predicting the hidden state time course.

        Parameters
        ----------
        sequence_length : int
            Length of the segment of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the model.

        Returns
        -------
        list of tensorflow.data.Datasets
            Dataset for each subject.
        """
        subject_datasets = [
            Dataset.from_tensor_slices(subject)
            .batch(sequence_length, drop_remainder=True)
            .batch(batch_size)
            .prefetch(-1)
            for subject in self.subjects
        ]

        return subject_datasets

    def validate_subjects(self):
        """Validate data files."""
        n_channels = [subject.shape[1] for subject in self.subjects]
        if not np.equal(n_channels, n_channels[0]).all():
            raise ValueError("All subjects should have the same number of channels.")
