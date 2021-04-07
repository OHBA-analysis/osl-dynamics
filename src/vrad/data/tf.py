import numpy as np
import tensorflow
from tensorflow.python.data import Dataset


class TensorFlowDataset:
    """Class for creating TensorFlow datasets."""

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
            [n_batches(memmap, sequence_length) for memmap in self.subjects]
        )

    def training_dataset(
        self,
        sequence_length: int,
        batch_size: int,
        alpha: list = None,
        n_alpha_embeddings: int = 1,
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
            Number of embeddings used when inferring alpha. Optional.
        concatenate : bool
            Should we concatenate the datasets for each subject? Optional, the
            default is True.

        Returns
        -------
        tensorflow.data.Dataset
            Dataset for training the model.
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        n_batches = self.count_batches(sequence_length)
        n_embeddings = self.n_embeddings or 1

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
                if n_embeddings > n_alpha_embeddings:
                    # We remove data points in alpha that are not in the new time
                    # embedded data
                    alp = alpha[i][(n_embeddings - n_alpha_embeddings) // 2 :]
                    subject = self.subjects[i][: alp.shape[0]]
                else:
                    # We remove the data points that are not in alpha
                    alp = alpha[i]
                    subject = self.subjects[i][
                        (n_alpha_embeddings - n_embeddings) // 2 : alp.shape[0]
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

                # The dataset has returns two inputs to the model: data and alpha
                # It also returns the subject id as the target
                subject_datasets.append(
                    Dataset.zip(
                        ({"data": subject_data, "alpha": alpha_data}, subject_tracker)
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


def n_batches(arr: np.ndarray, sequence_length: int, step_size: int = None) -> int:
    """Calculate the number of batches an array will be split into.

    Parameters
    ----------
    arr : numpy.ndarray
        Time series data.
    sequence_length : int
        Length of sequences which the data will be segmented in to.
    step_size : int
        The number of samples by which to move the sliding window between sequences.

    Returns
    -------
    int
        Number of batches.
    """
    step_size = step_size or sequence_length
    final_slice_start = arr.shape[0] - sequence_length + 1
    index = np.arange(0, final_slice_start, step_size)[:, None] + np.arange(
        sequence_length
    )
    return len(index)
