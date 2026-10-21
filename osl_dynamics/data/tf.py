"""Functions related to TensorFlow datasets.

Two routes produce the same batched dataset of dictionaries (keyed by model
input name):

- In memory / memory maps: :func:`sequence_index` lists every sequence as a
  (session, start) pair, :func:`make_slice_fn` cuts a sequence out of the
  arrays with numpy and :func:`dataset_from_index` shuffles the index (a
  perfect shuffle, no buffer) and batches.
- TFRecords: :meth:`Data.save_tfrecord_dataset` writes one file per session
  and a JSON config, :func:`load_tfrecord_dataset` streams them back
  (shuffled files, interleaved reads, a bounded shuffle buffer of
  sequences).
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def concatenate_datasets(datasets: List):
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


def sequence_index(
    n_samples: List[int],
    sequence_length: int,
    step_size: Optional[int] = None,
    keep: Optional[List[int]] = None,
) -> np.ndarray:
    """Index of every sequence that can be cut from a set of sessions.

    Parameters
    ----------
    n_samples : list of int
        Number of samples in each session.
    sequence_length : int
        Length of the segment of data to feed into the model.
    step_size : int, optional
        Number of samples to slide the sequence across the data.
        Default is no overlap.
    keep : list of int, optional
        Sessions to include. Default is all sessions.

    Returns
    -------
    index : np.ndarray
        Array of shape (n_sequences, 2). Each row is the session index
        and the sample the sequence starts at. Rows are ordered by session
        then start.
    """
    step_size = step_size or sequence_length
    keep = range(len(n_samples)) if keep is None else keep
    rows = []
    for i in keep:
        n_sequences = (n_samples[i] - sequence_length) // step_size + 1
        if n_sequences > 0:
            starts = np.arange(n_sequences) * step_size
            rows.append(np.stack([np.full(n_sequences, i), starts], axis=1))
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(rows).astype(np.int64)


def split_index(
    index: np.ndarray,
    validation_split: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a sequence index into training and validation sets.

    The split is done within each session, so every session contributes
    sequences to both sets.

    Parameters
    ----------
    index : np.ndarray
        Sequence index from :func:`sequence_index`.
    validation_split : float
        Fraction of each session's sequences to put in the validation set.
    seed : int, optional
        Seed for picking the validation sequences.

    Returns
    -------
    train_index : np.ndarray
        Index of training sequences.
    val_index : np.ndarray
        Index of validation sequences.
    """
    rng = np.random.default_rng(seed)
    train_rows = []
    val_rows = []
    for session in np.unique(index[:, 0]):
        rows = np.where(index[:, 0] == session)[0]
        n_val = int(validation_split * len(rows))
        val = rng.choice(rows, size=n_val, replace=False)
        train_rows.append(np.setdiff1d(rows, val))
        val_rows.append(np.sort(val))
    train_index = index[np.concatenate(train_rows)]
    val_index = index[np.concatenate(val_rows)]
    if len(val_index) == 0:
        raise ValueError(
            "No full sequences could be assigned to the validation set. "
            "Consider reducing the sequence_length."
        )
    return train_index, val_index


def element_shapes(
    sequence_length: int,
    n_channels: int,
    session_labels: Optional[List] = None,
    extra_channels: Optional[Dict] = None,
) -> Dict:
    """Shape of each element in a dataset (before batching).

    Parameters
    ----------
    sequence_length : int
        Length of the segment of data to feed into the model.
    n_channels : int
        Number of channels in the data.
    session_labels : list of SessionLabels, optional
        Session labels to add as inputs.
    extra_channels : dict, optional
        Extra channels to add as inputs. Keys are input names and values
        are lists of arrays (one per session).

    Returns
    -------
    shapes : dict
        Dictionary mapping input names to their shape.
    """
    shapes = {"data": [sequence_length, n_channels]}
    for label in session_labels or []:
        shapes[label.name] = [sequence_length]
    for name, values in (extra_channels or {}).items():
        if values[0].ndim == 1:
            shapes[name] = [sequence_length]
        else:
            shapes[name] = [sequence_length, values[0].shape[-1]]
    return shapes


def make_slice_fn(
    arrays: List[np.ndarray],
    sequence_length: int,
    session_labels: Optional[List] = None,
    extra_channels: Optional[Dict] = None,
):
    """Create a function that cuts one sequence out of the data.

    The function takes a row of a sequence index (see :func:`sequence_index`)
    and returns a dictionary of inputs for the model. The slicing is done
    with numpy, so the arrays can be memory maps and nothing other than the
    sequence itself is loaded into memory.

    Parameters
    ----------
    arrays : list of np.ndarray
        Time series data for each session. Can be memory maps.
    sequence_length : int
        Length of the segment of data to feed into the model.
    session_labels : list of SessionLabels, optional
        Session labels to add as inputs.
    extra_channels : dict, optional
        Extra channels to add as inputs. Keys are input names and values
        are lists of arrays (one per session).

    Returns
    -------
    slice_fn : callable
        Function mapping an index row to a dictionary of tensors. The
        shape of each element is available as :code:`slice_fn.shapes`.
    """
    import tensorflow as tf  # moved here to avoid slow imports

    session_labels = session_labels or []
    extra_channels = extra_channels or {}
    shapes = element_shapes(
        sequence_length, arrays[0].shape[-1], session_labels, extra_channels
    )
    names = list(shapes.keys())

    def _slice(session, start):
        session = int(session)
        start = int(start)
        stop = start + sequence_length
        outputs = [np.ascontiguousarray(arrays[session][start:stop], dtype=np.float32)]
        for label in session_labels:
            outputs.append(np.full(sequence_length, label.values[session], np.float32))
        for values in extra_channels.values():
            outputs.append(
                np.ascontiguousarray(values[session][start:stop], dtype=np.float32)
            )
        return outputs

    def slice_fn(row):
        tensors = tf.numpy_function(
            _slice, [row[0], row[1]], [tf.float32] * len(names), stateful=False
        )
        for tensor, name in zip(tensors, names):
            tensor.set_shape(shapes[name])
        return dict(zip(names, tensors))

    slice_fn.shapes = shapes
    slice_fn.numpy = _slice
    return slice_fn


def dataset_from_index(
    index: np.ndarray,
    slice_fn,
    batch_size: int,
    shuffle: bool = True,
    drop_last_batch: bool = False,
    seed: Optional[int] = None,
):
    """Create a batched dataset from a sequence index.

    Shuffling is done on the index (which is tiny), so it is a perfect
    shuffle without a buffer. The index is reshuffled every epoch.

    Parameters
    ----------
    index : np.ndarray
        Sequence index from :func:`sequence_index`.
    slice_fn : callable
        Function from :func:`make_slice_fn`.
    batch_size : int
        Number sequences in each mini-batch which is used to train the model.
    shuffle : bool, optional
        Should we shuffle the sequences?
    drop_last_batch : bool, optional
        Should we drop the last batch if it is smaller than the batch size?
    seed : int, optional
        Seed for shuffling.

    Returns
    -------
    dataset : tf.data.Dataset
        Batched dataset. The number of batches is known
        (:code:`dataset.cardinality()`).
    """
    import tensorflow as tf  # moved here to avoid slow imports

    dataset = tf.data.Dataset.from_tensor_slices(index)
    if shuffle:
        dataset = dataset.shuffle(len(index), seed=seed)
    dataset = dataset.map(slice_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_last_batch)
    return dataset.prefetch(tf.data.AUTOTUNE)


TFRECORD_CONFIG_FILE = "tfrecord_config.json"

# Bump this when the TFRecord file or config format changes, existing
# directories are then rewritten rather than reused
TFRECORD_FORMAT_VERSION = 1


def tfrecord_filename(tfrecord_dir: str, split: str, session: int, n_sessions: int):
    """Path of the TFRecord file for one session.

    Parameters
    ----------
    tfrecord_dir : str
        Directory containing the TFRecord dataset.
    split : str
        :code:`"train"` or :code:`"val"`.
    session : int
        Session index.
    n_sessions : int
        Total number of sessions (sets the zero padding).

    Returns
    -------
    filename : str
        Path to the TFRecord file.
    """
    width = len(str(n_sessions - 1))
    return os.path.join(tfrecord_dir, f"{split}_{session:0{width}d}.tfrecord")


def load_tfrecord_config(tfrecord_dir: str) -> Dict:
    """Load the config describing a TFRecord dataset.

    Parameters
    ----------
    tfrecord_dir : str
        Directory containing the TFRecord dataset.

    Returns
    -------
    config : dict
        Settings the dataset was written with: :code:`sequence_length`,
        :code:`step_size`, :code:`n_channels`, :code:`n_sessions`,
        :code:`input_shapes`, :code:`validation_split`,
        :code:`validation_split_seed`, :code:`n_sequences` (per session),
        :code:`n_val_sequences` (per session, or :code:`None`) and
        :code:`channel_range` (max - min of each channel in the training
        data).
    """
    path = os.path.join(tfrecord_dir, TFRECORD_CONFIG_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Create the TFRecord dataset with "
            "Data.save_tfrecord_dataset."
        )
    with open(path) as file:
        return json.load(file)


def save_tfrecord(filepath: str, index: np.ndarray, slice_fn) -> Tuple:
    """Write sequences to a TFRecord file.

    Parameters
    ----------
    filepath : str
        Path to the TFRecord file.
    index : np.ndarray
        Rows of a sequence index (see :func:`sequence_index`) to write.
    slice_fn : callable
        Function from :func:`make_slice_fn`.

    Returns
    -------
    amin : np.ndarray
        Minimum of each data channel over the sequences written.
    amax : np.ndarray
        Maximum of each data channel over the sequences written.
    """
    import tensorflow as tf  # moved here to avoid slow imports
    from tensorflow.train import BytesList, Example, Feature, Features

    names = list(slice_fn.shapes.keys())
    n_channels = slice_fn.shapes["data"][-1]
    amin = np.full(n_channels, np.inf, dtype=np.float32)
    amax = np.full(n_channels, -np.inf, dtype=np.float32)

    with tf.io.TFRecordWriter(filepath) as writer:
        for session, start in index:
            arrays = slice_fn.numpy(session, start)
            amin = np.minimum(amin, arrays[0].min(axis=0))
            amax = np.maximum(amax, arrays[0].max(axis=0))
            feature = {
                name: Feature(
                    bytes_list=BytesList(value=[tf.io.serialize_tensor(a).numpy()])
                )
                for name, a in zip(names, arrays)
            }
            writer.write(
                Example(features=Features(feature=feature)).SerializeToString()
            )

    return amin, amax


def load_tfrecord_dataset(
    tfrecord_dir: str,
    batch_size: int,
    shuffle: bool = True,
    concatenate: bool = True,
    drop_last_batch: bool = False,
    buffer_size: int = 4000,
    cycle_length: Optional[int] = None,
    keep: Optional[List[int]] = None,
    seed: Optional[int] = None,
):
    """Load a TFRecord dataset written by :meth:`Data.save_tfrecord_dataset`.

    The data is streamed from disk. When shuffling, the session files are
    read in a random order, :code:`cycle_length` files are read at the same
    time (round robin) and the resulting stream of sequences is shuffled
    with a buffer of :code:`buffer_size` sequences. Memory use is therefore
    roughly :code:`buffer_size` sequences plus prefetched batches, whatever
    the size of the dataset.

    Parameters
    ----------
    tfrecord_dir : str
        Directory containing the TFRecord dataset.
    batch_size : int
        Number sequences in each mini-batch which is used to train the model.
    shuffle : bool, optional
        Should we shuffle the sequences?
    concatenate : bool, optional
        Should we concatenate the datasets for each session?
    drop_last_batch : bool, optional
        Should we drop the last batch if it is smaller than the batch size?
    buffer_size : int, optional
        Number of sequences held in memory for shuffling. Larger values
        mix sessions better at the cost of memory. Default is 4000.
    cycle_length : int, optional
        Number of session files read at the same time. More files gives a
        better mix of sessions in each batch. Default is the number of
        sessions, capped at 64.
    keep : list of int, optional
        Sessions to load. Default is all sessions.
    seed : int, optional
        Seed for shuffling.

    Returns
    -------
    dataset : tf.data.Dataset or tuple of tf.data.Dataset
        Dataset for training or evaluating the model (or a list of datasets,
        one per session, if :code:`concatenate=False`), along with the
        validation set if the dataset was written with a validation split.
        The number of batches is known (:code:`dataset.cardinality()`)
        and :code:`dataset.osld_metadata` holds the sequence length, step
        size, number of sequences, number of channels and channel range.
    """
    import tensorflow as tf  # moved here to avoid slow imports

    config = load_tfrecord_config(tfrecord_dir)
    n_sessions = config["n_sessions"]
    input_shapes = config["input_shapes"]
    keep = list(range(n_sessions)) if keep is None else keep

    feature_description = {
        name: tf.io.FixedLenFeature([], tf.string) for name in input_shapes
    }

    def _parse_example(example):
        parsed = tf.io.parse_single_example(example, feature_description)
        return {
            name: tf.ensure_shape(
                tf.io.parse_tensor(tensor, tf.float32), input_shapes[name]
            )
            for name, tensor in parsed.items()
        }

    def _n_batches(n_sequences):
        if drop_last_batch:
            return n_sequences // batch_size
        return math.ceil(n_sequences / batch_size)

    def _metadata(n_sequences):
        return {
            "sequence_length": config["sequence_length"],
            "step_size": config["step_size"],
            "n_sequences": n_sequences,
            "n_channels": config["n_channels"],
            "range": np.array(config["channel_range"], dtype=np.float32),
        }

    def _finish(dataset, n_sequences, shuffle):
        if shuffle:
            dataset = dataset.shuffle(buffer_size, seed=seed)
        dataset = dataset.batch(batch_size, drop_remainder=drop_last_batch)
        dataset = dataset.apply(
            tf.data.experimental.assert_cardinality(_n_batches(n_sequences))
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset.osld_metadata = _metadata(n_sequences)
        return dataset

    def _create_dataset(split, shuffle):
        filenames = [
            tfrecord_filename(tfrecord_dir, split, i, n_sessions) for i in keep
        ]
        for filename in filenames:
            if not os.path.exists(filename):
                raise FileNotFoundError(
                    f"{filename} not found. Regenerate the TFRecord dataset "
                    "with Data.save_tfrecord_dataset."
                )
        key = "n_sequences" if split == "train" else "n_val_sequences"
        n_sequences = [config[key][i] for i in keep]

        if concatenate:
            files = tf.data.Dataset.from_tensor_slices(filenames)
            if shuffle:
                files = files.shuffle(len(filenames), seed=seed)
                n_files = cycle_length or min(len(filenames), 64)
            else:
                # Read one file at a time so sessions stay in order
                n_files = 1
            dataset = files.interleave(
                tf.data.TFRecordDataset,
                cycle_length=n_files,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=not shuffle,
            )
            dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
            return _finish(dataset, sum(n_sequences), shuffle)

        # Otherwise create a dataset for each session separately
        datasets = []
        for filename, n in zip(filenames, n_sequences):
            dataset = tf.data.TFRecordDataset(filename)
            dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
            datasets.append(_finish(dataset, n, shuffle))
        return datasets

    if config["validation_split"] is None:
        return _create_dataset("train", shuffle)

    return _create_dataset("train", shuffle), _create_dataset("val", False)


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
        raise TypeError("dataset must be a TensorFlow dataset or a list of datasets")

    return dataset


def get_range(dataset) -> np.ndarray:
    """The range (max-min) of values contained in a batched Tensorflow dataset.

    Parameters
    ----------
    dataset : tf.data.Dataset
        TensorFlow dataset.

    Returns
    -------
    range_ : np.ndarray
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


def get_n_channels(dataset) -> int:
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


def get_n_batches(dataset) -> int:
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


def get_n_sequences_and_range(dataset) -> Tuple[int, np.ndarray]:
    """Get number of sequences and range (max-min) of values.

    Parameters
    ----------
    dataset : tf.data.Dataset
        TensorFlow dataset.

    Returns
    -------
    n_sequences : int
        Number of sequences.
    range_ : np.ndarray
        Range of each channel.
    """
    dataset = _validate_tf_dataset(dataset)

    # Datasets loaded from TFRecords know this already
    metadata = getattr(dataset, "osld_metadata", {})
    if "range" in metadata:
        return metadata["n_sequences"], metadata["range"]

    n_sequences = 0
    amax = []
    amin = []
    for batch in dataset:
        if isinstance(batch, dict):
            batch = batch["data"]
        batch = batch.numpy()
        batch_size, _, n_channels = batch.shape
        batch = batch.reshape(-1, n_channels)
        amin.append(np.amin(batch, axis=0))
        amax.append(np.amax(batch, axis=0))
        n_sequences += batch_size
    range_ = np.amax(amax, axis=0) - np.amin(amin, axis=0)
    return n_sequences, range_
