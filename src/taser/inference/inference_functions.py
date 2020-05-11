"""A series of inference-specific functions which are too broad to include in a model.

"""

import tensorflow as tf


def pseudo_sigma_to_sigma(pseudo_sigma):
    """Ensure covariance matrices are positive semi-definite.

    Parameters
    ----------
    pseudo_sigma : tf.Tensor
        An arbitrary [M x N x N] matrix.

    Returns
    -------
    sigma : tf.Tensor
        A positive semi-definite matrix
    """
    upper_triangle = tf.linalg.band_part(pseudo_sigma, 0, -1)
    positive_definite = upper_triangle @ tf.transpose(upper_triangle, perm=[0, 2, 1])
    sigma = positive_definite + tf.eye(positive_definite.shape[1]) * 1e-6
    return sigma


def normalise_covariance(covariance):
    """Normalise covariance matrix based on its trace.

    The trace of `covariance` is taken. All values are then divided by it.

    Parameters
    ----------
    covariance : tf.Tensor
        Tensor of the form [M x N x N]

    Returns
    -------
    normalised_covariance : tf.Tensor
        Tensor of the form [M x N x N]
    """
    normalisation = tf.reduce_sum(tf.linalg.diag_part(covariance), axis=1)[
        ..., tf.newaxis, tf.newaxis
    ]
    normalised_covariance = covariance / normalisation
    return normalised_covariance
