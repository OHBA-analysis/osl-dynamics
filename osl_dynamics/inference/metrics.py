"""Metrics for evaluating model performance.

"""

import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from scipy.linalg import eigvalsh
from tqdm import trange


def alpha_correlation(alpha_1, alpha_2):
    """Calculates the correlation between modes of two alpha time series.

    Parameters
    ----------
    alpha_1 : np.ndarray
        First alpha time series. Shape is (n_samples, n_modes).
    alpha_2 : np.ndarray
        Second alpha time series. Shape is (n_samples, n_modes).

    Returns
    -------
    corr : np.ndarray
        Correlation of each mode in the corresponding alphas.
        Shape is (n_modes,).
    """
    if alpha_1.shape[1] != alpha_2.shape[1]:
        raise ValueError(
            "alpha_1 and alpha_2 shapes are incomptible. "
            + f"alpha_1.shape={alpha_1.shape}, alpha_2.shape={alpha_2.shape}."
        )
    n_modes = alpha_1.shape[1]
    corr = np.corrcoef(alpha_1, alpha_2, rowvar=False)
    corr = np.diagonal(corr[:n_modes, n_modes:])
    return corr


def confusion_matrix(mode_time_course_1, mode_time_course_2):
    """Calculate the confusion matrix of two mode time courses.

    For two mode-time-courses, calculate the confusion matrix (i.e. the
    disagreement between the mode selection for each sample). If either sequence is
    two dimensional, it will first have argmax(axis=1) applied to it. The produces the
    expected result for a one-hot encoded sequence but other inputs are not guaranteed
    to behave.

    This function is a wrapper for sklearn.metrics.confusion_matrix.

    Parameters
    ----------
    mode_time_course_1 : np.ndarray
    mode_time_course_2 : np.ndarray

    Returns
    -------
    cm : nd.ndarray
        Confusion matrix
    """
    if mode_time_course_1.ndim == 2:
        mode_time_course_1 = mode_time_course_1.argmax(axis=1)
    if mode_time_course_2.ndim == 2:
        mode_time_course_2 = mode_time_course_2.argmax(axis=1)
    if not ((mode_time_course_1.ndim == 1) and (mode_time_course_2.ndim == 1)):
        raise ValueError("Both mode time courses must be either 1D or 2D.")

    return sklearn_confusion_matrix(mode_time_course_1, mode_time_course_2)


def dice_coefficient_1d(sequence_1, sequence_2):
    """Calculate the Dice coefficient of a discrete array

    Given two sequences containing a number of discrete elements (i.e. a
    categorical variable), calculate the Dice coefficient of those sequences.

    The Dice coefficient is 2 times the number of equal elements (equivalent to
    true-positives) divided by the sum of the total number of elements.

    Parameters
    ----------
    sequence_1 : nd.ndarray
        A sequence containing discrete elements.
    sequence_2 : nd.ndarray
        A sequence containing discrete elements.

    Returns
    -------
    dice : float
        The Dice coefficient of the two sequences.
    """
    if (sequence_1.ndim, sequence_2.ndim) != (1, 1):
        raise ValueError(
            f"sequences must be 1D: {(sequence_1.ndim, sequence_2.ndim)} != (1, 1)."
        )
    if (sequence_1.dtype, sequence_2.dtype) != (int, int):
        raise TypeError("Both sequences must be integer (categorical).")

    return 2 * ((sequence_1 == sequence_2).sum()) / (len(sequence_1) + len(sequence_2))


def dice_coefficient(sequence_1, sequence_2):
    """Wrapper method for `dice_coefficient`.

    If passed a one-dimensional array, it will be sent straight to `dice_coefficient`.
    Given a two-dimensional array, it will perform an argmax calculation on each sample.
    The default axis for this is zero, i.e. each row represents a sample.

    Parameters
    ----------
    sequence_1 : nd.ndarray
        A sequence containing either 1D discrete or 2D continuous data.
    sequence_2 : nd.ndarray
        A sequence containing either 1D discrete or 2D continuous data.

    Returns
    -------
    dice : float
        The Dice coefficient of the two sequences.
    """
    if (len(sequence_1.shape) not in [1, 2]) or (len(sequence_2.shape) not in [1, 2]):
        raise ValueError("Both sequences must be either 1D or 2D")
    if (len(sequence_1.shape) == 1) and (len(sequence_2.shape) == 1):
        return dice_coefficient_1d(sequence_1, sequence_2)
    if len(sequence_1.shape) == 2:
        sequence_1 = sequence_1.argmax(axis=1)
    if len(sequence_2.shape) == 2:
        sequence_2 = sequence_2.argmax(axis=1)
    return dice_coefficient_1d(sequence_1, sequence_2)


def frobenius_norm(A, B):
    """Calculates the frobenius norm of the difference of two matrices.

    The Frobenius norm is calculated as sqrt( Sum_ij abs(a_ij - b_ij)^2 ).

    Parameters
    ----------
    A : np.ndarray
        First matrix. Shape must be (n_modes, n_channels, n_channels) or
        (n_channels, n_channels).
    B : np.ndarray
        Second matrix. Shape must be (n_modes, n_channels, n_channels) or
        (n_channels, n_channels).

    Returns
    -------
    norm : float
        The Frobenius norm of the difference of A and B. If A and B are
        stacked matrices, we sum the Frobenius norm of each sub-matrix.
    """
    if A.ndim == 2 and B.ndim == 2:
        norm = np.linalg.norm(A - B, ord="fro")
    elif A.ndim == 3 and B.ndim == 3:
        norm = np.linalg.norm(A - B, ord="fro", axis=(1, 2))
        norm = np.sum(norm)
    else:
        raise ValueError(
            f"Shape of A and/or B is incorrect. A.shape={A.shape}, B.shape={B.shape}."
        )
    return norm


def pairwise_frobenius_distance(matrices):
    """Calculates the pairwise frobenius distance of a set of matrices.

    Parameters
    ----------
    matrices : np.ndarray
        The set of matrices. Shape must be (n_matrices, n_channels, n_channels)

    Returns
    -------
    pairwise_distance : np.ndarray
        Matrix of pairwise Frobenius distance. Shape is (n_matrices, n_matrices)
    """
    return np.sqrt(
        np.sum(
            np.square(np.expand_dims(matrices, 0) - np.expand_dims(matrices, 1)),
            axis=(-2, -1),
        )
    )


def pairwise_matrix_correlations(matrices, remove_diagonal=False):
    """Calculate the correlation between elements of covariance matrices.

    Parameters
    ----------
    matrices : np.ndarray
        Shape must be (n_matrices, N, N).

    Returns
    -------
    correlations : np.ndarray
        Pairwise Pearson correlation between elements of each matrix.
        Shape is (n_matrices, n_matrices).
    """
    n_matrices = matrices.shape[0]
    matrices = matrices.reshape(n_matrices, -1)
    correlations = np.corrcoef(matrices)
    if remove_diagonal:
        correlations -= np.eye(n_matrices)
    return correlations


def riemannian_distance(M1, M2, threshold=1e-3):
    """Calculate the Riemannian distance between two matrices.

    The Riemannian distance is defined as: d = (sum log(eig(M_1 * M_2))) ^ 0.5

    Parameters
    ----------
    M1 : np.ndarray
    M2 : np.ndarray
    threshold : float
        Threshold to apply when there are negative eigenvalues. Must be positive.

    Returns
    -------
    d : np.ndarray
    """
    eigvals = eigvalsh(M1, M2, driver="gv")
    if np.any(eigvals < 0):
        eigvals = np.maximum(eigvals, threshold)

    d = np.sqrt((np.log(eigvals) ** 2).sum())
    return d


def pairwise_riemannian_distances(matrices, threshold=1e-3):
    """Calculate the Riemannian distance between matrices.

    Parameters
    ----------
    matrices : np.ndarray
        Shape must be (n_matrices, N, N).
    threshold : float
        Threshold to apply when there are negative eigenvalues. Must be positive.

    Returns
    -------
    riemannian_distances : np.ndarray
        Matrix containing the pairwise Riemannian distances between matrices.
        Shape is (n_matrices, n_matrices).
    """
    matrices.astype(np.float64)
    n_matrices = matrices.shape[0]
    riemannian_distances = np.zeros([n_matrices, n_matrices])
    for i in trange(n_matrices, desc="Computing Riemannian distances", ncols=98):
        for j in range(i + 1, n_matrices):
            riemannian_distances[i][j] = riemannian_distance(
                matrices[i], matrices[j], threshold=threshold
            )

    # Only the upper triangular entries are filled,
    # the diagonal entries are zeros
    riemannian_distances = riemannian_distances + riemannian_distances.T

    return riemannian_distances


def pairwise_rv_coefficient(M, remove_diagonal=False):
    """Calculate the RV coefficient for two matrices.

    Parameters
    ----------
    M : np.ndarray
        Set of matrices. Shape is (n_matrices, N, N)

    Returns
    -------
    C : np.ndarray
        Matrix of pairwise RV coefficients. Shape is (n_matrices, n_matrices)
    """
    n_matrices = M.shape[0]
    # First compute the scalar product matrices for each data set X
    scal_arr_list = []

    for arr in M:
        scal_arr = np.dot(arr, np.transpose(arr))
        scal_arr_list.append(scal_arr)

    # Now compute the 'between study cosine matrix' C
    C = np.zeros([n_matrices, n_matrices])

    for index, element in np.ndenumerate(C):
        nom = np.trace(
            np.dot(np.transpose(scal_arr_list[index[0]]), scal_arr_list[index[1]])
        )
        denom1 = np.trace(
            np.dot(np.transpose(scal_arr_list[index[0]]), scal_arr_list[index[0]])
        )
        denom2 = np.trace(
            np.dot(np.transpose(scal_arr_list[index[1]]), scal_arr_list[index[1]])
        )
        Rv = nom / np.sqrt(np.dot(denom1, denom2))
        C[index[0], index[1]] = Rv

    if remove_diagonal:
        C -= np.eye(n_matrices)

    return C


def pairwise_congruence_coefficient(M, remove_diagonal=False):
    """Computes the congruence coefficient between covariance/correlation matrices
    Parameters
    ----------
    M : np.ndarray
        Set of symmetric semi-positive definite matrices. Shape is (n_matrices, N, N).

    Returns
    -------
    C : np.ndarray
        Matrix of pairwise congruence coefficients. Shape is (n_matrices, n_matrices).
    """

    n_matrices = M.shape[0]
    C = np.zeros([n_matrices, n_matrices])
    for index, element in np.ndenumerate(C):
        nom = np.trace(np.dot(np.transpose(M[index[0]]), M[index[1]]))
        denom1 = np.trace(np.dot(np.transpose(M[index[0]]), M[index[0]]))
        denom2 = np.trace(np.dot(np.transpose(M[index[1]]), M[index[1]]))
        cc = nom / np.sqrt(np.dot(denom1, denom2))
        C[index[0], index[1]] = cc

    if remove_diagonal:
        C -= np.eye(n_matrices)

    return C
