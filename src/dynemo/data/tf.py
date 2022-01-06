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

    def dataset(
        self,
        sequence_length: int,
        batch_size: int,
        shuffle: bool = True,
        validation_split: float = None,
        alpha: list = None,
        n_alpha_embeddings: int = 1,
        concatenate: bool = True,
    ) -> tensorflow.data.Dataset:
        """Create a tensorflow dataset for training or evaluation.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the model.
        shuffle : bool
            Should we shuffle sequences (within a batch) and batches.
            Optional, default is True.
        validation_split : float
            Ratio to split the dataset into a training and validation set.
            Optional, default returns the entire data.
        alpha : list of np.ndarray
            List of mode mixing factors for each subject. Optional.
            If passed, we create a dataset that includes alpha at each time point.
            Such a dataset can be used to train the observation model.
        n_alpha_embeddings : int
            Number of embeddings used when inferring alpha. Optional.
        concatenate : bool
            Should we concatenate the datasets for each subject? Optional, the
            default is True.

        Returns
        -------
        tensorflow.data.Dataset or Tuple
            Dataset for training or evaluating the model along with the validation
            set if validation_split was passed.
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
                dataset = Dataset.from_tensor_slices(subject).batch(
                    sequence_length, drop_remainder=True
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
                    subject = self.subjects[i][: alp.shape[0]]
                else:
                    # We remove the data points that are not in alpha
                    alp = alpha[i]
                    subject = self.subjects[i][
                        (n_alpha_embeddings - n_embeddings) // 2 : alp.shape[0]
                    ]

                # Create dataset
                input_data = {"data": subject, "alpha": alp}
                dataset = Dataset.from_tensor_slices(input_data).batch(
                    sequence_length, drop_remainder=True
                )
                subject_datasets.append(dataset)

        # Create a dataset from all the subjects concatenated
        if concatenate:
            full_dataset = concatenate_datasets(subject_datasets, shuffle=False)

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
                print(
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
                    print(
                        f"Subject {i}: "
                        + f"{len(training_datasets[i])} batches in training dataset, "
                        + f"{len(validation_datasets[i])} batches in the validation dataset."
                    )
                return training_datasets, validation_datasets


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


def concatenate_datasets(
    datasets: list, shuffle: bool = True
) -> tensorflow.data.Dataset:
    """Concatenates a list of Tensorflow datasets.

    Parameters
    ----------
    datasets : list
        List of Tensorflow datasets.
    Shuffle : bool
        Should we shuffle the final concatenated dataset?
        Optional, default is True.

    Returns
    -------
    tensorflow.data.Dataset
        Concatenated dataset.
    """

    full_dataset = datasets[0]
    for ds in datasets[1:]:
        full_dataset = full_dataset.concatenate(ds)

    if shuffle:
        full_dataset = full_dataset.shuffle(100000)

    return full_dataset
