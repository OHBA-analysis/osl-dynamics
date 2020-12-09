"""Inference-specific functions which are too broad to include in a model.

"""

import numpy as np
import tensorflow as tf


def is_symmetric(matrix: np.ndarray) -> bool:
    """Checks if a matrix is symmetric.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to be checked.

    Returns
    -------
    bool
        Flag to indicate whether or not the matrix is symmetric.
    """
    return np.all(np.abs(matrix - np.transpose(matrix, (0, 2, 1))) < 1e-8)


@tf.function
def cholesky_factor_to_full_matrix(cholesky_factor: tf.Tensor):
    """Convert a cholesky factor into a full matrix.

    Parameters
    ----------
    cholesky_factor : tf.Tensor
        Cholesky factor of the matrix. Only the lower triangle of this tensor is used.
        Shape is (None, n_states, n_channels, n_channels).

    Returns
    -------
    tf.Tensor
        The full matrix calculated from the cholesky factor. Shape is
        (None, n_states, n_channels, n_channels).
    """
    # The upper triangle is trainable but we should zero it because the array
    # is the cholesky factor of the full covariance
    cholesky_factor = tf.linalg.band_part(cholesky_factor, -1, 0)

    # Calculate the full matrix
    full_matrix = tf.matmul(cholesky_factor, tf.transpose(cholesky_factor, (0, 2, 1)))

    return full_matrix


def cholesky_factor(full_matrix: np.ndarray):
    """Calculate the cholesky decomposition of a matrix.

    Parameters
    ----------
    full_matrix : np.ndarray
        Matrix to calculate the cholesky decomposition for. Shape is (n_states,
        n_channels, n_channels).

    Returns
    -------
    np.ndarray
        Cholesky factor of the matrix. Shape is (n_states, n_channels, n_channels).
    """
    cholesky_factor = np.empty(full_matrix.shape)
    for i in range(full_matrix.shape[0]):
        cholesky_factor[i] = np.linalg.cholesky(full_matrix[i])
    return cholesky_factor


def trace_normalize(matrices):
    """Normalise a matrix based on its trace.

    The trace of each matrix in 'matrices' is taken. All values are then
    divided by it.

    Parameters
    ----------
    covariances : tf.Tensor
        Tensor of shape (None, n_states, n_channels, n_channels).

    Returns
    -------
    tf.Tensor
        Tensor of shape (None, n_states, n_channels, n_channels).
    """
    normalization = tf.reduce_sum(tf.linalg.diag_part(matrices), axis=1)[
        ..., tf.newaxis, tf.newaxis
    ]
    return matrices.shape[1] * matrices / normalization
