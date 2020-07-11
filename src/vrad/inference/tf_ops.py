"""Helper functions for TensorFlow

"""

import os
import tensorflow as tf
import tensorflow.keras.models as models
from vrad.inference.models import _create_kl_loss_fn, _ll_loss_fn


def gpu_growth():
    """Only allocate the amount of memory required on the GPU.

    If resources are shared between multiple users, it's polite not to hog the GPUs!

    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def select_gpu(gpu_number):
    """Allows the user to pick a GPU to use."""
    if isinstance(gpu_number, int):
        gpu_number = str(gpu_number)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    print(f"Using GPU {gpu_number}")


def train_predict_dataset(
    time_series, sequence_length, batch_size=32, window_shift=None
):
    window_shift = window_shift or sequence_length
    dataset = tf.data.Dataset.from_tensor_slices(time_series)
    training_dataset = dataset.window(
        size=sequence_length, shift=window_shift, drop_remainder=True
    )  # shift=sequence_length means 0% overlap
    training_dataset = training_dataset.flat_map(
        lambda chunk: chunk.batch(sequence_length, drop_remainder=True)
    )

    training_dataset = training_dataset.shuffle(time_series.shape[0])
    training_dataset = training_dataset.batch(batch_size, drop_remainder=True)
    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    training_dataset = tf.data.Dataset.zip(
        (training_dataset, training_dataset)
    )  # dataset must return input and target

    prediction_dataset = tf.data.Dataset.from_tensor_slices(time_series)
    prediction_dataset = prediction_dataset.batch(
        sequence_length, drop_remainder=True
    ).batch(batch_size, drop_remainder=True)

    if not len(list(training_dataset)) or not len(list(prediction_dataset)):
        max_batch_size = int(time_series.shape[0] / sequence_length)
        raise ValueError(
            f"For a non-windowed time series, the maximum batch size is "
            f"int(time_series.shape[0] / sequence_length) = {max_batch_size}"
        )

    return training_dataset, prediction_dataset


def load_model(filename):
    """Loads a model saved with its .save(filename) method.
    """
    model = models.load_model(
        filename,
        custom_objects={
            "_ll_loss_fn": _ll_loss_fn,
            "_kl_loss_fn": _create_kl_loss_fn(1.0),
        },
    )
    return model
