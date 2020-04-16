from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.mixture import BayesianGaussianMixture


def learn_mu_sigma(
    data: np.ndarray, n_states: int, n_channels: int, learn_means: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    data
    n_states
    n_channels
    learn_means

    Returns
    -------

    """
    if learn_means:
        # use sklearn learn to do GMM
        gmm = BayesianGaussianMixture(
            n_components=n_states, covariance_type="full"
        ).fit(data)
    else:
        # make sure we force means to be zero:
        gmm = BayesianGaussianMixture(
            n_components=n_states,
            covariance_type="full",
            mean_prior=np.zeros(n_channels),
            mean_precision_prior=1e12,
        ).fit(data)
    return gmm.covariances_, gmm.means_


def process_covariance(
    covariances: np.ndarray, means: np.ndarray, n_states: int, learn_means: bool
) -> np.ndarray:
    """

    Parameters
    ----------
    covariances
    means
    n_states
    learn_means

    Returns
    -------

    """
    if learn_means:
        full_covariances = covariances
    else:
        means = means[:, np.newaxis, :]
        full_covariances = (means @ means.transpose(0, 2, 1)) + covariances

    norms = np.trace(full_covariances, axis1=1, axis2=2)
    full_covariances /= norms[:, np.newaxis, np.newaxis]

    return full_covariances


def matrix_sqrt_3d(matrix):
    """

    Parameters
    ----------
    matrix

    Returns
    -------

    """
    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise ValueError("Only accepts matrices with dimensions M x N x N")
    return_matrix = np.empty_like(matrix)
    for index, layer in enumerate(matrix):
        return_matrix[index] = scipy.linalg.sqrtm(layer)
    return return_matrix


def find_cholesky_decompositions(
    covariances: np.ndarray, means: np.ndarray, learn_means: bool,
):
    """

    Parameters
    ----------
    covariances
    means
    learn_means

    Returns
    -------

    """
    n_states, n_channels = covariances.shape[:2]
    w = np.identity(n_channels)
    full_cov = process_covariance(covariances, means, n_states, learn_means)
    b_k = np.linalg.pinv(w) @ full_cov
    matrix_sqrt = matrix_sqrt_3d(b_k @ b_k.transpose(0, 2, 1))
    cholesky_djs = np.linalg.cholesky(matrix_sqrt)

    return cholesky_djs


def plot_covariances(cholesky_djs: np.ndarray, fig_kwargs=None):
    """

    Parameters
    ----------
    fig_kwargs
    cholesky_djs
    """
    if fig_kwargs is None:
        fig_kwargs = {}
    c_i = cholesky_djs @ cholesky_djs.transpose((0, 2, 1))
    fig, axes = plt.subplots(ncols=len(c_i), **fig_kwargs)
    for covariance, axis in zip(c_i, axes):
        pl = axis.matshow(covariance)
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pl, cax=cax)
    plt.show()
