"""Functions for reading and writing data.

"""

import logging
from os import listdir, path

import mne
import mat73
import numpy as np
import scipy.io

from osl_dynamics.utils import misc

_logger = logging.getLogger("osl-dynamics")
_allowed_ext = [".npy", ".mat", ".txt", ".fif"]


def validate_inputs(inputs):
    """Validates inputs.

    Parameters
    ----------
    inputs : list of str or str or np.ndarray
        Inputs files or data.

    Returns
    -------
    validated_inputs : list of str or str
        Validated inputs.
    """
    if isinstance(inputs, str):
        if path.isdir(inputs):
            validated_inputs = list_dir(inputs, keep_ext=_allowed_ext)
        else:
            validated_inputs = [inputs]

    elif isinstance(inputs, np.ndarray):
        if inputs.ndim == 1:
            validated_inputs = [inputs[:, np.newaxis]]
        elif inputs.ndim == 2:
            validated_inputs = [inputs]
        else:
            validated_inputs = inputs

    elif isinstance(inputs, list):
        if len(inputs) == 0:
            raise ValueError("Empty list passed.")
        elif isinstance(inputs[0], str):
            validated_inputs = []
            for inp in inputs:
                if path.isdir(inp):
                    validated_inputs += list_dir(inp, keep_ext=_allowed_ext)
                elif path.exists(inp):
                    validated_inputs.append(inp)
                else:
                    _logger.warn(f"{inp} not found")
        else:
            validated_inputs = inputs

    else:
        raise ValueError("inputs must be str, np.ndarray or list.")

    return validated_inputs


def file_ext(filename):
    """Returns the extension of a file.

    Parameters
    ----------
    filename : str
        Path to file.
    """
    if not isinstance(filename, str):
        return None
    _, ext = path.splitext(filename)
    return ext


def list_dir(path, keep_ext=None):
    """Lists a directory.

    Parameters
    ----------
    path : str
        Directory to list.
    keep_ext : str or list, optional
        Extensions of files to include in the returned list.
        Default is to include all files.

    Returns
    -------
    files : list
        Full path to files with the correct extension.
    """
    files = []
    if keep_ext is None:
        for file in sorted(listdir(path)):
            files.append(path + "/" + file)
    else:
        if isinstance(keep_ext, str):
            keep_ext = [keep_ext]
        for file in sorted(listdir(path)):
            if file_ext(file) in keep_ext:
                files.append(path + "/" + file)
    return files


def load_data(
    data,
    data_field="X",
    picks=None,
    reject_by_annotation=None,
    mmap_location=None,
    mmap_mode="c",
):
    """Loads time series data.

    Checks the data shape is time by channels and that the data is
    :code:`float32`.

    Parameters
    ----------
    data : numpy.ndarray or str or list
        An array or path to a :code:`.npy`, :code:`.mat`, :code:`.txt` or
        :code:`.fif` file containing the data.
    data_field : str, optional
        If a MATLAB filename is passed, this is the field that corresponds to
        the data.
    picks : str or list of str, optional
        Argument passed to `mne.io.Raw.get_data
        <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw\
        .get_data>`_ or `mne.Epochs.get_data <https://mne.tools/stable\
        /generated/mne.Epochs.html#mne.Epochs.get_data>`_.
        Only used if a fif file is passed.
    reject_by_annotation : str, optional
        Argument passed to `mne.io.Raw.get_data <https://mne.tools/stable\
        /generated/mne.io.Raw.html#mne.io.Raw.get_data>`_.
        Only used if a fif file is passed.
    mmap_location : str, optional
        Filename to save the data as a numpy memory map.
    mmap_mode : str, optional
        Mode to load memory maps in. Default is :code:`'c'`.

    Returns
    -------
    data : np.memmap or np.ndarray
        Data.
    """
    if isinstance(data, np.ndarray):
        data = data.astype(np.float32)
        if mmap_location is None:
            return data
        else:
            # Save to a file so we can load data as a memory map
            np.save(mmap_location, data)
            data = mmap_location

    if isinstance(data, str):
        # Check if file/folder exists
        if not path.exists(data):
            raise FileNotFoundError(data)

        # Check extension
        ext = file_ext(data)
        if ext not in _allowed_ext:
            raise ValueError(f"Data file must have extension: {_allowed_ext}.")

        # Load a MATLAB file
        if ext == ".mat":
            data = load_matlab(data, data_field)
            data = data.astype(np.float32)
            if mmap_location is None:
                return data
            else:
                # Save to a file so we can load data as a memory map
                np.save(mmap_location, data)
                data = mmap_location

        # Load a numpy file
        elif ext == ".npy":
            if mmap_location is None:
                data = np.load(data)
                data = data.astype(np.float32)
                return data
            else:
                mmap_location = data

        # Load a text file
        elif ext == ".txt":
            data = np.loadtxt(data)
            data = data.astype(np.float32)
            if mmap_location is None:
                return data
            else:
                np.save(mmap_location, data)
                data = mmap_location

        # Load a fif file
        elif ext == ".fif":
            data = load_fif(data, picks, reject_by_annotation)
            data = data.astype(np.float32)
            if mmap_location is None:
                return data
            else:
                np.save(mmap_location, data)
                data = mmap_location

    # Load data as memmap
    data = np.load(mmap_location, mmap_mode=mmap_mode)
    data = data.astype(np.float32)

    return data


def load_fif(filename, picks=None, reject_by_annotation=None):
    """Load a fif file.

    Parameters
    ----------
    filename : str
        Path to fif file. Must end with :code:`'raw.fif'` or :code:`'epo.fif'`.
    picks : str or list of str, optional
        Argument passed to `mne.io.Raw.get_data
        <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw\
        .get_data>`_ or `mne.Epochs.get_data <https://mne.tools/stable\
        /generated/mne.Epochs.html#mne.Epochs.get_data>`_.
    reject_by_annotation : str, optional
        Argument passed to `mne.io.Raw.get_data <https://mne.tools/stable\
        /generated/mne.io.Raw.html#mne.io.Raw.get_data>`_.

    Returns
    -------
    data : np.ndarray
        Time series data in format (n_samples, n_channels).
        If an :code:`mne.Epochs` fif file is pass (:code:`'epo.fif'`) the we
        concatenate the epochs in the first axis.
    """
    if "raw.fif" in filename:
        raw = mne.io.read_raw_fif(filename, verbose=False)
        data = raw.get_data(
            picks=picks,
            reject_by_annotation=reject_by_annotation,
            verbose=False,
        ).T
    elif "epo.fif" in filename:
        epochs = mne.read_epochs(filename, verbose=False)
        data = epochs.get_data(picks=picks)
        data = np.swapaxes(data, 1, 2).reshape(-1, data.shape[1])
    else:
        raise ValueError("a fif file must end with 'raw.fif' or 'epo.fif'.")
    return data


def load_matlab(filename, field):
    """Loads a MATLAB file.

    Parameters
    ----------
    filename : str
        Filename of MATLAB file to read.
    field : str
        Field that corresponds to the data.

    Returns
    -------
    data : np.ndarray
        Data in the MATLAB file.
    """
    mat = loadmat(filename, return_dict=True)
    if field not in mat:
        raise KeyError(f"field '{field}' missing from MATLAB file.")
    return mat[field]


def loadmat(filename, return_dict=False):
    """Wrapper for scipy.io.loadmat or mat73.loadmat.

    Parameters
    ----------
    filename : str
        Filename of MATLAB file to read.
    return_dict : bool, optional
        If there's only one field should we still return a :code:`dict`?
        Default is to return a numpy array if there is only one field.
        If there are multiple fields, a :code:`dict` is always returned.

    Returns
    -------
    mat : dict or np.ndarray
        Data in the MATLAB file.
    """
    try:
        mat = scipy.io.loadmat(filename, simplify_cells=True)
    except NotImplementedError:
        mat = mat73.loadmat(filename)

    if not return_dict:
        # Check if there's only one key in the MATLAB file
        fields = [field for field in mat if "__" not in field]
        if len(fields) == 1:
            mat = mat[fields[0]]

    return mat


def load_tfrecord_dataset(
    tfrecord_dir,
    batch_size,
    shuffle=True,
    validation_split=None,
    concatenate=True,
    drop_last_batch=False,
    buffer_size=10000,
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
        to less random shuffling but will be quicker. Default is 10000.
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
            # Calculate how many batches should be in the training dataset
            dataset_size = dtf.get_n_batches(full_dataset)
            training_dataset_size = round((1.0 - validation_split) * dataset_size)

            # Split the dataset into training and validation datasets
            training_dataset = full_dataset.take(training_dataset_size)
            validation_dataset = full_dataset.skip(training_dataset_size)
            _logger.info(
                f"{training_dataset_size} batches in training dataset, "
                + f"{dataset_size - training_dataset_size} batches in the validation "
                + "dataset."
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
                # Calculate how many batches should be in the training dataset
                dataset_size = dtf.get_n_batches(ds)
                training_dataset_size = round((1.0 - validation_split) * dataset_size)

                # Split the dataset into training and validation datasets
                training_datasets.append(ds.take(training_dataset_size))
                validation_datasets.append(ds.skip(training_dataset_size))
                _logger.info(
                    f"Session {i}: "
                    + f"{training_dataset_size} batches in training dataset, "
                    + f"{dataset_size - training_dataset_size} batches in the validation "
                    + "dataset."
                )

            return training_datasets, validation_datasets
