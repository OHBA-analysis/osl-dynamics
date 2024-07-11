"""Function related to TensorFlow datasets.

"""

import logging
import numpy as np

from osl_dynamics.utils import misc

_logger = logging.getLogger("osl-dynamics")


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


def load_tfrecord_dataset(
    tfrecord_dir,
    batch_size,
    shuffle=True,
    validation_split=None,
    concatenate=True,
    drop_last_batch=False,
    buffer_size=4000,
    keep=None,
):
    """Load a TFRecord dataset.

    Parameters
    ----------
    tfrecord_dir : str
        Directory containing the TFRecord datasets.
    batch_size : int
        Number sequences in each mini-batch which is used to train the model.
    shuffle : bool, optional
        Should we shuffle sequences (within a batch) and batches.
    validation_split : float, optional
        Ratio to split the dataset into a training and validation set.
    concatenate : bool, optional
        Should we concatenate the datasets for each array?
    drop_last_batch : bool, optional
        Should we drop the last batch if it is smaller than the batch size?
    buffer_size : int, optional
        Buffer size for shuffling a TensorFlow Dataset. Smaller values will lead
        to less random shuffling but will be quicker. Default is 100000.
    keep : list of int, optional
        List of session indices to keep. If :code:`None`, then all sessions
        are kept.

    Returns
    -------
    dataset : tf.data.Dataset or tuple
        Dataset for training or evaluating the model along with the validation
        set if :code:`validation_split` was passed.
    """
    import tensorflow as tf  # moved here to avoid slow imports

    tf_record_config = misc.load(f"{tfrecord_dir}/tfrecord_config.pkl")
    identifier = tf_record_config["identifier"]
    sequence_length = tf_record_config["sequence_length"]
    n_channels = tf_record_config["n_channels"]
    session_labels = tf_record_config["session_labels"]
    n_sessions = tf_record_config["n_sessions"]

    keep = keep or list(range(n_sessions))

    # Helper function for parsing training examples
    def _parse_example(example):
        feature_names = ["data"]
        tensor_shapes = {
            "data": [sequence_length, n_channels],
        }
        # Add session labels if there are any
        for feature_name in session_labels:
            feature_names.append(feature_name)
            tensor_shapes[feature_name] = [sequence_length]

        feature_description = {
            name: tf.io.FixedLenFeature([], tf.string) for name in feature_names
        }
        parsed_example = tf.io.parse_single_example(
            example,
            feature_description,
        )
        return {
            name: tf.ensure_shape(
                tf.io.parse_tensor(tensor, tf.float32), tensor_shapes[name]
            )
            for name, tensor in parsed_example.items()
        }

    tfrecord_paths = (
        f"{tfrecord_dir}"
        "/dataset_{array:0{v}d}-of-{n_session:0{v}d}"
        f".{identifier}.tfrecord"
    )
    tfrecord_filenames = []
    for i in keep:
        filepath = tfrecord_paths.format(
            array=i,
            n_session=n_sessions - 1,
            v=len(str(n_sessions - 1)),
        )
        tfrecord_filenames.append(filepath)

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
            full_dataset = full_dataset.shuffle(buffer_size)

            # Group into batches
            full_dataset = full_dataset.batch(
                batch_size, drop_remainder=drop_last_batch
            )

            # Shuffle batches
            full_dataset = full_dataset.shuffle(buffer_size)

        else:
            # Create the TFRecord dataset
            full_dataset = tfrecord_filenames.interleave(
                tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE
            )

            # Parse the examples
            full_dataset = full_dataset.map(_parse_example)

            # Group into batches
            full_dataset = full_dataset.batch(
                batch_size, drop_remainder=drop_last_batch
            )

        if validation_split is None:
            # Return the dataset
            return full_dataset.prefetch(tf.data.AUTOTUNE)

        else:
            # Split the dataset into training and validation datasets
            training_dataset, validation_dataset = tf.keras.utils.split_dataset(
                full_dataset,
                right_size=validation_split,
            )
            _logger.info(
                f"{len(training_dataset)} batches in training dataset, "
                f"{len(validation_dataset)} batches in the validation "
                "dataset."
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
                ds = ds.shuffle(buffer_size)

            # Group into batches
            ds = ds.batch(batch_size, drop_remainder=drop_last_batch)

            if shuffle:
                # Shuffle batches
                ds = ds.shuffle(buffer_size)

            full_datasets.append(ds.prefetch(tf.data.AUTOTUNE))

        if validation_split is None:
            # Return the full dataset for each array
            return full_datasets

        else:
            # Split the dataset for each array separately
            training_datasets = []
            validation_datasets = []
            for i, ds in enumerate(full_datasets):
                tds, vds = tf.keras.utils.split_dataset(
                    full_datasets[i],
                    right_size=validation_split,
                )
                _logger.info(
                    f"Session {i}: "
                    f"{len(tds)} batches in training dataset, "
                    f"{len(vds)} batches in the validation dataset."
                )

            return training_datasets, validation_datasets


def _validate_tf_dataset(dataset):
    """Check if the input is a valid TensorFlow dataset.

    Parameters
    ----------
    dataset : tf.data.Dataset or list
        TensorFlow dataset or list of datasets.

    Returns
    -------
    dataset : tf.data.Dataset
        TensorFlow dataset.
    """
    import tensorflow as tf  # avoid slow imports

    if isinstance(dataset, list):
        if len(dataset) == 1:
            dataset = dataset[0]
        else:
            dataset = concatenate_datasets(dataset)

    if not isinstance(dataset, tf.data.Dataset):
        raise ValueError("dataset must be a TensorFlow dataset or a list of datasets")

    return dataset


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
    dataset = _validate_tf_dataset(dataset)
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
    dataset = _validate_tf_dataset(dataset)
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

    dataset = _validate_tf_dataset(dataset)

    # Count number of batches
    cardinality = dataset.cardinality()
    if cardinality == tf.data.UNKNOWN_CARDINALITY:
        for i, _ in enumerate(dataset):
            pass
        return i + 1

    return cardinality.numpy()
