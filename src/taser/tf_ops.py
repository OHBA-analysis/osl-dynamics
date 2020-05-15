"""Helper functions for TensorFlow

"""
import logging
from typing import Tuple

import numpy as np
import tensorflow as tf


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


def train_predict_dataset(
    time_series: np.ndarray,
    mini_batch_length: int,
    batch_size: int,
    buffer_size: int = 10000,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    if time_series.shape[0] < time_series.shape[1]:
        logging.warning("Assuming longer axis to be time and transposing.")
        time_series = time_series.T
    dataset = tf.data.Dataset.from_tensor_slices(time_series)

    train_dataset = (
        dataset.batch(mini_batch_length, drop_remainder=True)
        .shuffle(buffer_size)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .cache()
    )
    predict_dataset = (
        dataset.batch(mini_batch_length, drop_remainder=True)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .cache()
    )

    return train_dataset, predict_dataset
