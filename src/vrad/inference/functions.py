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


def matrix_sqrt_3d(matrix):
    """A wrapper function for `scipy.linalg.sqrtm`.

    SciPy's matrix square root function only works on [N x N] 2D matrices. This
    function provides a simple solution for performing this operation on a stack of
    [N x N] 2D arrays.

    Parameters
    ----------
    matrix : numpy.ndarray
        [M x N x N] matrix.

    Returns
    -------
    matrix_sqrt : numpy.ndarray
        A stack of matrix square roots of the same dimensions as `matrix` ([M x N x N])
    """
    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise ValueError("Only accepts matrices with dimensions M x N x N")
    return_matrix = np.empty_like(matrix)
    for index, layer in enumerate(matrix):
        return_matrix[index] = scipy.linalg.sqrtm(layer)
    return return_matrix


@tf.function
def normalize_covariances(covariances):
    """Normalise covariance matrix based on its trace.

    The trace of `covariance` is taken. All values are then divided by it.

    Parameters
    ----------
    covariances : tf.Tensor
        Tensor of the form [M x N x N]

    Returns
    -------
    normalized_covariances : tf.Tensor
        Tensor of the form [M x N x N]
    """
    normalization = tf.reduce_sum(tf.linalg.diag_part(covariances), axis=1)[
        ..., tf.newaxis, tf.newaxis
    ]
    normalized_covariances = covariances / normalization
    return normalized_covariances


def full_covariances(means, covariances):
    """Calculate full covariance.

    Given a set of covariances and means, calculate the full covariance matrices.

    Parameters
    ----------
    means : numpy.ndarray
        Matrix of means of dimensions [n_states x n_channels].
    covariances : numpy.ndarray
        Matrix of covariances of dimensions [n_states x n_channels x n_channels].

    Returns
    -------
    full_covariances : numpy.ndarray
        Normalised full covariance matrices of dimensions [n_states x n_channels
        x n_channels].
    """
    means = means[:, np.newaxis, :]
    full_covariances = (means @ means.transpose(0, 2, 1)) + covariances
    return full_covariances


def softplus(x):
    """Softplus function"""
    return np.log(1 + np.exp(x))
