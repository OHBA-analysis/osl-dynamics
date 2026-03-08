"""Helper functions for TensorFlow operations."""

import logging
import os
from typing import List, Union

_logger = logging.getLogger("osl-dynamics")


def gpu_growth() -> None:
    """Only allocate the amount of memory required on the GPU."""
    import tensorflow as tf  # moved here to avoid slow imports

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            _logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            _logger.error(e)


def select_gpu(gpu_numbers: Union[List[int], int]) -> None:
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
    _logger.info(f"Using GPU {gpu_numbers}")


def suppress_messages(level: int = 3) -> None:
    """Suppress messages from TensorFlow.

    Must be called before :func:`osl_dynamics.inference.tf_ops.gpu_growth`
    and :func:`osl_dynamics.inference.tf_ops.select_gpu`.

    Parameters
    ----------
    level : int, optional
        The level for the messages to suppress.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(level)
