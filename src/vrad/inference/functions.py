"""A series of inference-specific functions which are too broad to include in a model.

"""

import numpy as np
import scipy
import tensorflow as tf


def is_symmetric(matrix):
    """Checks if a matrix is symmetric"""
    return np.all(np.abs(matrix - np.transpose(matrix, (0, 2, 1))) < 1e-8)


@tf.function
def cholesky_factor_to_full_matrix(cholesky_factor):
    """Returns a legal non-singular matrix from a cholesky_factor tensor."""
    # The upper triangle is trainable but we should zero it because the array
    # is the cholesky factor of the full covariance
    cholesky_factor = tf.linalg.band_part(cholesky_factor, -1, 0)

    # Calculate the full matrix
    full_matrix = tf.matmul(cholesky_factor, tf.transpose(cholesky_factor, (0, 2, 1)))
    full_matrix = tf.add(
        full_matrix,
        1e-6
        * tf.eye(
            full_matrix.shape[1],
            batch_shape=[cholesky_factor.shape[0]],
            dtype=full_matrix.dtype,
        ),
    )
    return full_matrix


def cholesky_factor(full_matrix):
    """Calculate the cholesky decomposition of a matrix"""
    cholesky_factor = np.empty(full_matrix.shape)
    for i in range(full_matrix.shape[0]):
        cholesky_factor[i] = np.linalg.cholesky(full_matrix[i])
    return cholesky_factor


@tf.function
def trace_normalize(matrices):
    """Normalise a matrix based on its trace.

    The trace of each matrix in 'matrices' is taken. All values are then
    divided by it.

    Parameters
    ----------
    covariances : tf.Tensor
        Tensor of the form [M x N x N]

    Returns
    -------
    normalized_covariances : tf.Tensor
        Tensor of the form [M x N x N]
    """
    normalization = tf.reduce_sum(tf.linalg.diag_part(matrices), axis=1)[
        ..., tf.newaxis, tf.newaxis
    ]
    return matrices.shape[1] * matrices / normalization
