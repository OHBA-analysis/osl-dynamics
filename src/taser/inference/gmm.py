"""Model data using a `BayesianGaussianMixture` model from Scikit-Learn

"""
from typing import Tuple

import numpy as np
import scipy.linalg
from sklearn.mixture import BayesianGaussianMixture
import warnings


def learn_mu_sigma(
    data: np.ndarray,
    n_states: int,
    n_channels: int,
    learn_means: bool = False,
    retry_attempts: int = 5,
    gmm_kwargs: dict = None,
    normalize_covariance: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find a mixture of Gaussian distributions which characterises a dataset.

    Using scikit-learn's `BayesianGaussianMixture` class, find a set of Gaussian
    distributions which when linearly combined can explain a dataset.

    Parameters
    ----------
    data : numpy.ndarray
        Input data of dimensions [n_channels x n_time_points]
    n_states : int
        Number of states (Gaussian distributions) to try to find.
    n_channels : int
        The number of channels in the dataset.
    learn_means : bool
        If False (default), means will be assumed to be zero and given a strong
        weight as a prior.
    retry_attempts : int
        Number of times to retry fitting if `.fit()` doesn't converge.
    gmm_kwargs : dict
        Keyword arguments for the `BayesianGaussianMixture`

    Returns
    -------
    covariances : numpy.ndarray
        Covariances of the states (Gaussian distributions) found with dimensions
        [n_states x n_channels x n_channels]
    means : numpy.ndarray
        Means of the states (Gaussian distributions) found with dimensions [n_states x
        n_channels]
    """
    if retry_attempts < 1:
        raise ValueError("retry_attempts cannot be less than 1")
    if gmm_kwargs is None:
        gmm_kwargs = {}
    if learn_means:
        # use sklearn learn to do GMM
        gmm = BayesianGaussianMixture(
            n_components=n_states, covariance_type="full", **gmm_kwargs,
        )
    else:
        # make sure we force means to be zero:
        gmm = BayesianGaussianMixture(
            n_components=n_states,
            covariance_type="full",
            mean_prior=np.zeros(n_channels),
            mean_precision_prior=1e12,
            **gmm_kwargs,
        )
    for attempt in range(retry_attempts):
        warnings.filterwarnings('ignore')
        gmm.fit(data)
        if gmm.converged_:
            print(f"Converged on iteration {attempt}")
            break
        print(f"Failed to converge on iteration {attempt}")
    return gmm.covariances_, gmm.means_


def process_covariance(
    covariances: np.ndarray, means: np.ndarray, learn_means: bool
) -> np.ndarray:
    """Calculate normalised full covariance.

    Given a set of covariances and means, calculate the full covariance matrices and
    normalise them by their traces.

    Parameters
    ----------
    covariances : numpy.ndarray
        Matrix of covariances of dimensions [n_states x n_channels x n_channels].
    means : numpy.ndarray
        Matrix of means of dimensions [n_states x n_channels].
    learn_means : bool
        If True, include means in calculation (i.e. calculate full covariance).

    Returns
    -------
    full_covariance : numpy.ndarray
        Normalised full covariance matrices of dimensions [n_states x n_channels
        x n_channels].
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
    """A wrapper function for `scipy.linalg.sqrtm`.

    SciPy's matrix square root function only works on [N x N] 2D matrices. This
    function provides a simple solution for performing this operation on a stack of
    [N x N] 2D arrays.

    TensorFlow has a function to do this (`tensorflow.linalg.sqrtm`) but I won't
    move to that until the whole of gmm.py has moved to TensorFlow code.

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


def find_cholesky_decompositions(
    covariances: np.ndarray, means: np.ndarray, learn_means: bool,
):
    """Calculate the Cholesky decomposition of the full covariance of a distribution.

    Given a set of covariances [n_states x n_channels x n_channels] and means
    [n_states x n_channels], calculate the full covariance (i.e. with means included).
    We then take the matrix square root (sqrtm(M) * sqrtm(M) = M) and project it onto
    a set of principle components W. For now, this is implemented only as the identity
    matrix. A Cholesky decomposition is then performed on the result.

    Parameters
    ----------
    covariances : numpy.ndarray
        Covariance matrices for states [n_states x n_channels x n_channels]
    means : numpy.ndarray
        Means for states [n_states x n_channels]
    learn_means : bool
        If True, means will be incorporated to form full covariance matrices.

    Returns
    -------
    cholesky_decompositions : numpy.array
        [n_states x n_channels x n_channels] array containing the Cholesky
        decompositions of full covariance matrices.
    """
    n_states, n_channels = covariances.shape[:2]
    w = np.identity(n_channels)
    if learn_means:
        full_cov = process_covariance(covariances, means, learn_means)
    else:
        full_cov = covariances
    b_k = np.linalg.pinv(w) @ full_cov
    matrix_sqrt = matrix_sqrt_3d(b_k @ b_k.transpose(0, 2, 1))
    cholesky_djs = np.linalg.cholesky(matrix_sqrt)

    return cholesky_djs
