"""Helper functions for TensorFlow

"""
import logging
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from vrad.data import Data


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


def train_predict_dataset(time_series, sequence_length, batch_size, window_shift=None):
    window_shift = window_shift or sequence_length
    dataset = tf.data.Dataset.from_tensor_slices(time_series)
    training_dataset = dataset.window(
        size=sequence_length, shift=window_shift, drop_remainder=True
    )  # shift=sequence_length means 0% overlap
    training_dataset = training_dataset.flat_map(
        lambda chunk: chunk.batch(sequence_length, drop_remainder=True)
    )

    training_dataset = training_dataset.shuffle(time_series.shape[0])
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    training_dataset = tf.data.Dataset.zip(
        (training_dataset, training_dataset)
    )  # dataset must return input and target

    prediction_dataset = tf.data.Dataset.from_tensor_slices(time_series)
    prediction_dataset = prediction_dataset.batch(sequence_length).batch(batch_size)

    return training_dataset, prediction_dataset


# def train_predict_dataset(
#     time_series: Union[np.ndarray, Data],
#     mini_batch_length: int,
#     batch_size: int,
#     buffer_size: int = 10000,
# ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
#
#     # If data is in a Data class, get the time_series first
#     time_series = time_series[:]
#     time_series = time_series.astype(np.float32)
#
#     if time_series.shape[0] < time_series.shape[1]:
#         logging.warning("Assuming longer axis to be time and transposing.")
#         time_series = time_series.T
#     dataset = tf.data.Dataset.from_tensor_slices(time_series)
#
#     train_dataset = (
#         dataset.batch(mini_batch_length, drop_remainder=True)
#         .shuffle(buffer_size)
#         .batch(batch_size)
#         .prefetch(tf.data.experimental.AUTOTUNE)
#         .cache()
#     )
#     predict_dataset = (
#         dataset.batch(mini_batch_length, drop_remainder=True)
#         .batch(batch_size)
#         .prefetch(tf.data.experimental.AUTOTUNE)
#         .cache()
#     )
#
#     return train_dataset, predict_dataset
