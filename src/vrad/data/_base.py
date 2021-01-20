import pathlib
from shutil import rmtree
from typing import List

import numpy as np
import tensorflow
from tensorflow.python.data import Dataset
from tqdm import tqdm
from vrad.data import io, manipulation


class Data:
    """Base class for data.

    The Data class enables the input and processing of data. When given a list of
    files, it produces a set of numpy memory maps which contain their raw data.
    It also provides methods for batching data and creating TensorFlow Datasets.

    Parameters
    ----------
    inputs : list of str or str
        Filenames to be read.
    sampling_frequency : float
        Sampling frequency of the data in Hz. Optional, default is 1.0.
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
        sampling_frequency: float = 1.0,
        store_dir: str = "tmp",
        n_embeddings=0,
        n_pca_components=None,
        whiten=None,
        prepared=False,
    ):
        # Identifier for the data
        self._identifier = id(inputs)

        # Input files
        self.inputs = [inputs] if isinstance(inputs, str) else inputs
        self.store_dir = pathlib.Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_pattern = "input_data_{{i:0{width}d}}_{identifier}.npy".format(
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
        self.n_raw_data_channels = self.n_channels
        self.sampling_frequency = sampling_frequency
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

    def validate_subjects(self):
        """Validate data files."""
        n_channels = [subject.shape[1] for subject in self.subjects]
        if not np.equal(n_channels, n_channels[0]).all():
            raise ValueError("All subjects should have the same number of channels.")

    @property
    def raw_data(self) -> List:
        """Return raw data as a list of arrays."""
        return self.raw_data_memmaps

    @property
    def n_channels(self) -> int:
        """Number of channels in the data files."""
        return self.subjects[0].shape[1]

    @property
    def n_subjects(self) -> int:
        """Number of subjects."""
        return len(self.subjects)

    def load_data(self):
        """Import data into a list of memory maps."""
        memmaps = []
        for in_file, out_file in zip(
            tqdm(self.inputs, desc="Loading files", ncols=98), self.raw_data_filenames
        ):
            data = io.load_data(in_file, mmap_location=out_file)
            memmaps.append(data)
        return memmaps

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

    def delete_dir(self):
        """Deletes the directory that stores the memory maps."""
        rmtree(self.store_dir, ignore_errors=True)

    def training_dataset(
        self, sequence_length: int, batch_size: int
    ) -> tensorflow.data.Dataset:
        """Create a tensorflow dataset for training.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the model.

        Returns
        -------
        tensorflow.data.Dataset
            Dataset for training the model.
        """
        n_batches = self.count_batches(sequence_length)

        subject_datasets = []
        for i in range(self.n_subjects):
            subject = self.subjects[i]
            subject_data = Dataset.from_tensor_slices(subject).batch(
                sequence_length, drop_remainder=True
            )
            subject_tracker = Dataset.from_tensor_slices(
                np.zeros(n_batches[i], dtype=np.float32) + i
            )
            subject_datasets.append(Dataset.zip((subject_data, subject_tracker)))

        full_dataset = subject_datasets[0]
        for subject_dataset in subject_datasets[1:]:
            full_dataset = full_dataset.concatenate(subject_dataset)

        return (
            full_dataset.shuffle(100000).batch(batch_size).shuffle(100000).prefetch(-1)
        )

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

    def covariance_training_datasets(
        self,
        alpha_t: list,
        sequence_length: int,
        batch_size: int,
        n_alpha_embeddings: int = 0,
    ) -> tensorflow.data.Dataset:
        """Dataset for training covariances with a fixed alpha_t.

        Parameters
        ----------
        alpha_t : list of np.ndarray
            List of state mixing factors for each subject.
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the model.
        n_alpha_embeddings: int
            Number of embeddings when inferring alpha_t. Optional.

        Returns
        -------
        list of tensorflow.data.Dataset
            Subject-specific datasets for training the covariances.
        """
        n_batches = self.count_batches(sequence_length)

        # Validation
        if not isinstance(alpha_t, list):
            raise ValueError("alpha must be a list of numpy arrays.")

        subject_datasets = []
        for i in range(self.n_subjects):

            # We remove data points in alpha that are not in the new time embedded data
            alpha = alpha_t[i][(self.n_embeddings - n_alpha_embeddings) // 2 :]

            # We have more subject data points than alpha values so we trim the data
            subject = self.subjects[i][: alpha.shape[0]]

            # Create datasets
            alpha_data = Dataset.from_tensor_slices(alpha).batch(
                sequence_length, drop_remainder=True
            )
            subject_data = Dataset.from_tensor_slices(subject).batch(
                sequence_length, drop_remainder=True
            )
            subject_tracker = Dataset.from_tensor_slices(
                np.zeros(n_batches[i], dtype=np.float32) + i
            )

            subject_dataset = Dataset.zip(
                ({"data": subject_data, "alpha_t": alpha_data}, subject_tracker)
            )
            subject_dataset = (
                subject_dataset.shuffle(100000)
                .batch(batch_size)
                .shuffle(100000)
                .prefetch(-1)
            )
            subject_datasets.append(subject_dataset)

        return subject_datasets
