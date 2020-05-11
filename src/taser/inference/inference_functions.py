"""A series of inference-specific functions which are too broad to include in a model.

"""
import logging
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from taser.helpers.array_ops import trials_to_continuous
from taser.helpers.decorators import transpose


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


def pca(time_series: np.ndarray, n_components: Union[int, float] = None,) -> np.ndarray:

    if time_series.ndim == 3:
        logging.warning("Assuming 3D array is [channels x time x trials]")
        time_series = trials_to_continuous(time_series)
    if time_series.ndim != 2:
        raise ValueError("time_series must be a 2D array")
    if time_series.shape[0] < time_series.shape[1]:
        logging.warning("Assuming longer axis to be time and transposing.")
        time_series = time_series.T

    standard_scaler = StandardScaler()
    data_std = standard_scaler.fit_transform(time_series)

    pca_from_variance = PCA(n_components=n_components)
    data_pca = pca_from_variance.fit_transform(data_std)
    if 0 < n_components < 1:
        print(
            f"{pca_from_variance.n_components_} components are required to "
            f"explain {n_components * 100}% of the variance "
        )

    return data_pca


@transpose(0, "time_series")
def scale(time_series: np.ndarray) -> np.ndarray:
    scaled = StandardScaler().fit_transform(time_series)
    return scaled


def scale_pca(time_series: np.ndarray, n_components: Union[int, float]):
    return scale(pca(time_series=time_series, n_components=n_components))


def scale_pca_scale(time_series: np.ndarray, n_components: Union[int, float]):
    return scale(pca(scale(time_series), n_components=n_components))
