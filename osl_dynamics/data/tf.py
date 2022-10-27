"""Classes and function related to TensorFlow datasets.

"""

import numpy as np


class TensorFlowDataset:
    """Class for creating TensorFlow datasets."""

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
            [n_batches(memmap, sequence_length) for memmap in self.subjects]
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
            Such a dataset can be used to train the observation model.
        gamma : list of np.ndarray
            List of mode mixing factors for the functional connectivity.
            Used with a multi-time-scale model.
        n_alpha_embeddings : int
            Number of embeddings used when inferring alpha.
        concatenate : bool
            Should we concatenate the datasets for each subject?
        subj_id : bool
            Should we include the subject id in the dataset?
        step_size : int
            Number of samples to slide the sequence across the dataset.

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
                    dataset = create_dataset(
                        {"data": subject, "subj_id": subject_tracker},
                        self.sequence_length,
                        self.step_size,
                    )
                else:
                    dataset = create_dataset(
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
                dataset = create_dataset(
                    input_data, self.sequence_length, self.step_size
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


def n_batches(arr, sequence_length, step_size=None):
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
    n : int
        Number of batches.
    """
    step_size = step_size or sequence_length
    final_slice_start = arr.shape[0] - sequence_length + 1
    index = np.arange(0, final_slice_start, step_size)[:, None] + np.arange(
        sequence_length
    )
    return len(index)


def concatenate_datasets(datasets, shuffle=True):
    """Concatenates a list of TensorFlow datasets.

    Parameters
    ----------
    datasets : list
        List of TensorFlow datasets.
    Shuffle : bool
        Should we shuffle the final concatenated dataset?

    Returns
    -------
    full_dataset : tensorflow.data.Dataset
        Concatenated dataset.
    """

    full_dataset = datasets[0]
    for ds in datasets[1:]:
        full_dataset = full_dataset.concatenate(ds)

    if shuffle:
        full_dataset = full_dataset.shuffle(100000)

    return full_dataset


def create_dataset(
    data,
    sequence_length,
    step_size,
):
    """Creates a TensorFlow dataset of batched time series data.

    Parameters
    ----------
    data : dict
        Dictionary containing data to batch. Keys correspond to the input name
        for the model and the value is the data.
    sequence_length : int
        Sequence length to batch the data.
    step_size : int
        Number of samples to slide the sequence across the data.

    Returns
    -------
    dataset : tensorflow.data.Dataset
        TensorFlow dataset.
    """
    from tensorflow.data import Dataset  # moved here to avoid slow imports

    # Generate a non-overlapping sequence dataset
    if step_size == sequence_length:
        dataset = Dataset.from_tensor_slices(data)
        dataset = dataset.batch(sequence_length, drop_remainder=True)

    # Create an overlapping single model input dataset
    elif len(data) == 1:
        dataset = Dataset.from_tensor_slices(list(data.values())[0])
        dataset = dataset.window(sequence_length, step_size, drop_remainder=True)
        dataset = dataset.flat_map(
            lambda window: window.batch(sequence_length, drop_remainder=True)
        )

    # Create an overlapping multiple model input dataset
    else:

        def batch_windows(*windows):
            batched = [w.batch(sequence_length, drop_remainder=True) for w in windows]
            return Dataset.zip(tuple(batched))

        def tuple_to_dict(*d):
            names = list(data.keys())
            inputs = {}
            for i in range(len(data)):
                inputs[names[i]] = d[i]
            return inputs

        dataset = tuple([Dataset.from_tensor_slices(v) for v in data.values()])
        dataset = Dataset.zip(dataset)
        dataset = dataset.window(sequence_length, step_size, drop_remainder=True)
        dataset = dataset.flat_map(batch_windows)
        dataset = dataset.map(tuple_to_dict)

    return dataset


def get_range(dataset):
    """The range (max-min) of values contained in a batched Tensorflow dataset.

    Parameters
    ----------
    dataset : tensorflow.data.Dataset
        TensorFlow dataset.

    Returns
    -------
    range : np.ndarray
        Range of each channel.
    """
    amax = []
    amin = []
    for batch in dataset:
        if isinstance(batch, dict):
            batch = batch["data"]
        batch = batch.numpy()
        n_channels = batch.shape[-1]
        batch = batch.reshape(-1, n_channels)
        amin.append(np.amin(batch, axis=0))
        amax.append(np.amax(batch, axis=0))
    return np.amax(amax, axis=0) - np.amin(amin, axis=0)


def get_n_channels(dataset):
    """Get the number of channels in a batched TensorFlow dataset.

    Parameters
    ----------
    dataset : tensorflow.data.Dataset
        TensorFlow dataset.

    Returns
    -------
    n_channels : int
        Number of channels.
    """
    for batch in dataset:
        if isinstance(batch, dict):
            batch = batch["data"]
        batch = batch.numpy()
        return batch.shape[-1]


def get_n_batches(dataset):
    """Get number of batches in a TensorFlow dataset.

    Parameters
    ----------
    dataset : tensorflow.data.Dataset
        TensorFlow dataset.

    Returns
    -------
    n_batches : int
        Number of batches.
    """
    return dataset.cardinality().numpy()
