"""Helper functions for TensorFlow operations.

"""

import os


def gpu_growth():
    """Only allocate the amount of memory required on the GPU.

    If resources are shared between multiple users, it's polite not to hog the GPUs!
    """
    import tensorflow as tf  # moved here to avoid slow imports

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


def select_gpu(gpu_numbers):
    """Allows the user to pick a GPU to use.

    Parameters
    ----------
    gpu_number : list or int
        ID numbers for the GPU to use.
    """
    if isinstance(gpu_numbers, int):
        gpu_numbers = str(gpu_numbers)
    else:
        gpu_numbers = ",".join([str(gn) for gn in gpu_numbers])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_numbers
    print(f"Using GPU {gpu_numbers}")


def suppress_messages(level=3):
    """Suppress messages from tensorflow.

    Must be called before gpu_growth() and select_gpu().

    Parameters
    ----------
    level : int
        The level for the messages to suppress.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(level)
