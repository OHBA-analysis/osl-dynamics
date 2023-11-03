"""Function related to TensorFlow datasets.

"""

import numpy as np


def get_n_sequences(arr, sequence_length, step_size=None):
    """Calculate the number of sequences an array will be split into.

    Parameters
    ----------
    arr : np.ndarray
        Time series data.
    sequence_length : int
        Length of sequences which the data will be segmented in to.
    step_size : int, optional
        The number of samples by which to move the sliding window between
        sequences.

    Returns
    -------
    n : int
        Number of sequences.
    """
    step_size = step_size or sequence_length
    n_samples = (arr.shape[0] // sequence_length) * sequence_length
    return n_samples // step_size


def concatenate_datasets(datasets):
    """Concatenates a list of TensorFlow datasets.

    Parameters
    ----------
    datasets : list
        List of TensorFlow datasets.

    Returns
    -------
    full_dataset : tf.data.Dataset
        Concatenated dataset.
    """
    full_dataset = datasets[0]
    for ds in datasets[1:]:
        full_dataset = full_dataset.concatenate(ds)
    return full_dataset


def create_dataset(data, sequence_length, step_size):
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
    dataset : tf.data.Dataset
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
        dataset = dataset.window(
            sequence_length,
            step_size,
            drop_remainder=True,
        )
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
        dataset = dataset.window(
            sequence_length,
            step_size,
            drop_remainder=True,
        )
        dataset = dataset.flat_map(batch_windows)
        dataset = dataset.map(tuple_to_dict)

    return dataset


def save_tfrecord(data, sequence_length, step_size, filepath):
    """Save dataset to a TFRecord file.

    Parameters
    ----------
    data : dict
        Dictionary containing data to batch. Keys correspond to the input name
        for the model and the value is the data.
    sequence_length : int
        Sequence length to batch the data.
    step_size : int
        Number of samples to slide the sequence across the data.
    filepath : str
        Path to save the TFRecord file.
    """
    import tensorflow as tf  # moved here to avoid slow imports
    from tensorflow.train import Feature, Features, Example, BytesList

    dataset = create_dataset(data, sequence_length, step_size)

    # Helper function to serialize a sequence to a tensorflow example
    # byte string
    def _make_example(sequence):
        # Note this function assumes all features are tf tensors
        # and can be converted to bytes
        features = Features(
            feature={
                k: Feature(
                    bytes_list=BytesList(
                        value=[tf.io.serialize_tensor(v).numpy()],
                    )
                )
                for k, v in sequence.items()
            }
        )
        return Example(features=features).SerializeToString()

    # Serialize each sequence and write to a TFRecord file
    with tf.io.TFRecordWriter(filepath) as writer:
        for sequence in dataset:
            writer.write(_make_example(sequence))


def get_range(dataset):
    """The range (max-min) of values contained in a batched Tensorflow dataset.

    Parameters
    ----------
    dataset : tf.data.Dataset
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
    dataset : tf.data.Dataset
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
    dataset : tf.data.Dataset
        TensorFlow dataset.

    Returns
    -------
    n_batches : int
        Number of batches.
    """
    import tensorflow as tf  # avoid slow imports

    if isinstance(dataset, list):
        if len(dataset) == 1:
            dataset = dataset[0]
        else:
            dataset = concatenate_datasets(dataset)

    if not isinstance(dataset, tf.data.Dataset):
        raise ValueError("dataset must be a TensorFlow dataset or a list of datasets")

    # Count number of batches
    cardinality = dataset.cardinality()
    if cardinality == tf.data.UNKNOWN_CARDINALITY:
        for i, _ in enumerate(dataset):
            pass
        return i + 1

    return cardinality.numpy()
