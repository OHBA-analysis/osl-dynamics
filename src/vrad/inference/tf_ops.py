"""Helper functions for TensorFlow

"""

import os
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


def select_gpu(gpu_number):
    """Allows the user to pick a GPU to use."""
    if isinstance(gpu_number, int):
        gpu_number = str(gpu_number)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    print(f"Using GPU {gpu_number}")


def suppress_messages(level=3):
    """Suppress messages from tensorflow.

    Must be called before gpu_growth() and select_gpu().
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(level)


def unzip_dataset(zipped_dataset):
    num_datasets = len(zipped_dataset.element_spec)
    datasets = [
        zipped_dataset.map(lambda *x: x[index]) for index in range(num_datasets)
    ]
    return datasets
