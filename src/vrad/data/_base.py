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
    store_dir : str
        Directory to save results and intermediate steps to.
    """

    def __init__(self, inputs, store_dir="tmp"):
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
        self.raw_data_memmaps, self.discontinuities = self.load_data()
        self.subjects = self.raw_data_memmaps

        # Raw data statistics
        self.raw_data_mean = [
            np.mean(raw_data, axis=0) for raw_data in self.raw_data_memmaps
        ]
        self.raw_data_std = [
            np.std(raw_data, axis=0) for raw_data in self.raw_data_memmaps
        ]

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

    def load_data(self):
        """Import data into a list of memory maps."""
        memmaps = []
        discontinuities = []
        for in_file, out_file in zip(
            tqdm(self.inputs, desc="Loading files", ncols=98), self.raw_data_filenames
        ):
            data, discontinuity_indices, sampling_frequency = io.load_data(
                in_file, mmap_location=out_file
            )
            memmaps.append(data)
            discontinuities.append(discontinuity_indices)
        return memmaps, discontinuities

    def count_batches(self, sequence_length: int, step_size: int = None) -> np.ndarray:
        """Count batches.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        step_size : int
            We can produce sequences with overlaping data. step_size=sequence_length
            will give non-overlaping sequences, step_size=sequence_length/2 will give
            sequences with 50% overlap. Default is no overlap.

        Returns
        -------
        np.ndarray
            Number of batches for each subject's data.
        """
        return np.array(
            [
                manipulation.num_batches(memmap, sequence_length, step_size)
                for memmap in self.subjects
            ]
        )

    def delete_dir(self):
        """Deletes the directory that stores the memory maps."""
        rmtree(self.store_dir, ignore_errors=True)

    def training_dataset(
        self, sequence_length: int, batch_size: int, step_size: int = None
    ) -> tensorflow.data.Dataset:
        """Create a tensorflow dataset for training.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the model.
        step_size : int
            We can produce sequences with overlaping data. step_size=sequence_length
            will give non-overlaping sequences, step_size=sequence_length/2 will give
            sequences with 50% overlap. Default is no overlap.

        Returns
        -------
        tensorflow.data.Dataset
            Dataset for training the model.
        """
        num_batches = self.count_batches(sequence_length, step_size)

        subject_datasets = []
        for i in range(len(self.subjects)):
            subject = self.subjects[i]
            subject_data = Dataset.from_tensor_slices(subject).batch(
                sequence_length, drop_remainder=True
            )
            subject_tracker = Dataset.from_tensor_slices(
                np.zeros(num_batches[i], dtype=np.float32) + i
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
        step_size: int = None,
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
        step_size : int
            We can produce sequences with overlaping data. step_size=sequence_length
            will give non-overlaping sequences, step_size=sequence_length/2 will give
            sequences with 50% overlap. Default is no overlap.

        Returns
        -------
        list of tensorflow.data.Dataset
            Subject-specific datasets for training the covariances.
        """
        num_batches = self.count_batches(sequence_length, step_size)

        subject_datasets = []
        for i in range(len(self.subjects)):

            subject = self.subjects[i]
            subject_data = Dataset.from_tensor_slices(subject).batch(sequence_length)

            alpha = alpha_t[i]
            alpha_data = Dataset.from_tensor_slices(alpha).batch(sequence_length)

            subject_tracker = Dataset.from_tensor_slices(
                np.zeros(num_batches[i], dtype=np.float32) + i
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
