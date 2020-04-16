import tensorflow as tf
import numpy as np


def _pseudo_sigma_to_sigma(pseudo_sigma):
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


def _normalise_covariance(covariance):
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


def get_alpha_order(real_alpha, est_alpha):
    """Correlate covariance matrices to match known and inferred states.

    Parameters
    ----------
    real_alpha : array-like
    est_alpha : array-like

    Returns
    -------
    alpha_res_order : numpy.array
        The order of inferred states as determined by known states.

    """
    # establish ordering of factors so that they match real alphas
    ccs = np.zeros((real_alpha.shape[1], real_alpha.shape[1]))
    alpha_res_order = np.ones((real_alpha.shape[1]), int)

    for kk in range(real_alpha.shape[1]):
        for jj in range(real_alpha.shape[1]):
            if jj is not kk:
                cc = np.corrcoef(real_alpha[:, kk], est_alpha[:, jj])
                ccs[kk, jj] = cc[0, 1]
        alpha_res_order[kk] = int(np.argmax(ccs[kk, :]))

    return alpha_res_order
