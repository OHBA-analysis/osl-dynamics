"""Metrics for evaluating model performance.

"""

import warnings
import numpy as np
from scipy.linalg import eigvalsh
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from tqdm.auto import trange


def alpha_correlation(alpha_1, alpha_2, return_diagonal=True):
    """Calculates the correlation between mixing coefficient time series.

    Parameters
    ----------
    alpha_1 : np.ndarray or list of numpy.ndarray
        First alpha time series. Shape must be (n_samples, n_modes).
        alpha_1 will be concatenated if it's a list.
    alpha_2 : np.ndarray or list of numpy.ndarray
        Second alpha time series. Shape must be (n_samples, n_modes).
        alpha_2 will be concatenated if it's a list.
    return_diagonal: bool, optional
        (optional, default=True) Whether to return the diagonal elements of the correlation.
    Returns
    -------
    corr : np.ndarray
        Correlation of each mode in the corresponding alphas.
        Shape is (n_modes,) if return_diagonal = True, (n_modes,n_modes) otherwise.
    """
    if isinstance(alpha_1, list):
        alpha_1 = np.concatenate(alpha_1, axis=0)
    if isinstance(alpha_2, list):
        alpha_2 = np.concatenate(alpha_2, axis=0)
    if alpha_1.shape[1] != alpha_2.shape[1]:
        raise ValueError(
            "alpha_1 and alpha_2 shapes are incomptible. "
            + f"alpha_1.shape={alpha_1.shape}, alpha_2.shape={alpha_2.shape}."
        )
    n_modes = alpha_1.shape[1]
    corr = np.corrcoef(alpha_1, alpha_2, rowvar=False)
    corr = corr[:n_modes, n_modes:]
    if return_diagonal:
        corr = np.diagonal(corr)
    return corr


def confusion_matrix(state_time_course_1, state_time_course_2):
    """Calculate the `confusion_matrix \
    <https://scikit-learn.org/stable/modules/generated/\
    sklearn.metrics.confusion_matrix.html>`_ of two state time courses.

    For two state time courses, calculate the confusion matrix (i.e. the
    disagreement between the state selection for each sample). If either
    sequence is two dimensional, it will first have :code:`argmax(axis=1)`
    applied to it. The produces the expected result for a one-hot encoded
    sequence but other inputs are not guaranteed to behave.

    Parameters
    ----------
    state_time_course_1 : np.ndarray
        Mode time course. Shape must be (n_samples, n_states) or (n_samples,).
    state_time_course_2 : np.ndarray
        Mode time course. Shape must be (n_samples, n_states) or (n_samples,).

    Returns
    -------
    cm : np.ndarray
        Confusion matrix. Shape is (n_states, n_states).
    """
    if state_time_course_1.ndim == 2:
        state_time_course_1 = state_time_course_1.argmax(axis=1)
    if state_time_course_2.ndim == 2:
        state_time_course_2 = state_time_course_2.argmax(axis=1)
    if not ((state_time_course_1.ndim == 1) and (state_time_course_2.ndim == 1)):
        raise ValueError("Both state time courses must be either 1D or 2D.")
    return sklearn_confusion_matrix(state_time_course_1, state_time_course_2)


def dice_coefficient(sequence_1, sequence_2):
    """Calculates the Dice coefficient.

    The Dice coefficient is 2 times the number of equal elements (equivalent to
    true-positives) divided by the sum of the total number of elements.

    Parameters
    ----------
    sequence_1 : np.ndarray
        A sequence containing either 1D discrete or 2D continuous data.
        Shape must be (n_samples, n_states) or (n_samples,).
    sequence_2 : np.ndarray
        A sequence containing either 1D discrete or 2D continuous data.
        Shape must be (n_samples, n_states) or (n_samples,).

    Returns
    -------
    dice : float
        The Dice coefficient of the two sequences.
    """
    if (sequence_1.ndim not in [1, 2]) or (sequence_2.ndim not in [1, 2]):
        raise ValueError("Both sequences must be either 1D or 2D")
    if sequence_1.ndim == 2:
        sequence_1 = sequence_1.argmax(axis=1)
    if sequence_2.ndim == 2:
        sequence_2 = sequence_2.argmax(axis=1)
    return 2 * ((sequence_1 == sequence_2).sum()) / (len(sequence_1) + len(sequence_2))


def frobenius_norm(A, B):
    """Calculates the Frobenius norm of the difference of two matrices.

    The Frobenius norm is calculated as
    :math:`\sqrt{\displaystyle\sum_{ij} |a_{ij} - b_{ij}|^2}`.

    Parameters
    ----------
    A : np.ndarray
        First matrix. Shape must be (n_modes, n_channels, n_channels) or
        (n_channels, n_channels).
    B : np.ndarray
        Second matrix. Shape must be (n_modes, n_channels, n_channels) or
        (n_channels, n_channels)`.

    Returns
    -------
    norm : float
        The Frobenius norm of the difference of :code:`A` and :code:`B`.
        If :code:`A` and :code:`B` are stacked matrices, we sum the Frobenius
        norm of each sub-matrix.
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
    """Calculates the pairwise Frobenius distance of a set of matrices.

    Parameters
    ----------
    matrices : np.ndarray
        The set of matrices. Shape must be (n_matrices, n_channels, n_channels).

    Returns
    -------
    pairwise_distance : np.ndarray
        Matrix of pairwise Frobenius distance.
        Shape is (n_matrices, n_matrices).

    See Also
    --------
    frobenius_norm
    """
    return np.sqrt(
        np.sum(
            np.square(
                np.expand_dims(matrices, 0) - np.expand_dims(matrices, 1),
            ),
            axis=(-2, -1),
        )
    )


def pairwise_matrix_correlations(matrices, remove_diagonal=False):
    """Calculate the correlation between (flattened) covariance matrices.

    Parameters
    ----------
    matrices : np.ndarray
        Matrices. Shape must be (M, N, N).
    remove_diagonal : bool, optional
        Should we remove the diagonal before calculating the correction?

    Returns
    -------
    correlations : np.ndarray
        Pairwise Pearson correlation between elements of each flattened matrix.
        Shape is (M, M).
    """
    n_matrices = matrices.shape[0]
    matrices = matrices.reshape(n_matrices, -1)
    correlations = np.corrcoef(matrices)
    if remove_diagonal:
        correlations -= np.eye(n_matrices)
    return correlations


def riemannian_distance(M1, M2, threshold=1e-3):
    """Calculate the Riemannian distance between two matrices.

    The Riemannian distance is defined as
    :math:`d = \sqrt{\displaystyle\sum \log(\mathrm{eig}(M_1 * M_2))}`.

    Parameters
    ----------
    M1 : np.ndarray
        First matrix. Shape must be (N, N).
    M2 : np.ndarray
        Second matrix. Shape must be (N, N).
    threshold : float, optional
        Threshold to apply when there are negative eigenvalues.
        Must be positive.

    Returns
    -------
    d : float
        Riemannian distance.
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
        Matrices. Shape must be (M, N, N).
    threshold : float, optional
        Threshold to apply when there are negative eigenvalues.
        Must be positive.

    Returns
    -------
    riemannian_distances : np.ndarray
        Matrix containing the pairwise Riemannian distances between matrices.
        Shape is (M, M).

    See Also
    --------
    riemannian_distance
    """
    matrices.astype(np.float64)
    n_matrices = matrices.shape[0]
    riemannian_distances = np.zeros([n_matrices, n_matrices])
    for i in trange(n_matrices, desc="Computing Riemannian distances"):
        for j in range(i + 1, n_matrices):
            riemannian_distances[i][j] = riemannian_distance(
                matrices[i], matrices[j], threshold=threshold
            )

    # Only the upper triangular entries are filled,
    # the diagonal entries are zeros
    riemannian_distances = riemannian_distances + riemannian_distances.T

    return riemannian_distances


def pairwise_rv_coefficient(matrices, remove_diagonal=False):
    """Calculate the RV coefficient for two matrices.

    Parameters
    ----------
    matrices : np.ndarray
        Set of matrices. Shape must be (M, N, N).
    remove_diagonal : bool, optional
        Should we remove the diagonal before calculating the correction?

    Returns
    -------
    rv_coefficients : np.ndarray
        Matrix of pairwise RV coefficients. Shape is (M, M).
    """
    # First compute the scalar product matrices for each data set X
    scal_arr_list = []
    for arr in matrices:
        scal_arr = np.dot(arr, np.transpose(arr))
        scal_arr_list.append(scal_arr)

    # Now compute the 'between study cosine matrix'
    n_matrices = matrices.shape[0]
    rv_coefficients = np.zeros([n_matrices, n_matrices])

    for index, element in np.ndenumerate(rv_coefficients):
        nom = np.trace(
            np.dot(
                np.transpose(scal_arr_list[index[0]]),
                scal_arr_list[index[1]],
            )
        )
        denom1 = np.trace(
            np.dot(
                np.transpose(scal_arr_list[index[0]]),
                scal_arr_list[index[0]],
            )
        )
        denom2 = np.trace(
            np.dot(
                np.transpose(scal_arr_list[index[1]]),
                scal_arr_list[index[1]],
            )
        )
        Rv = nom / np.sqrt(np.dot(denom1, denom2))
        rv_coefficients[index[0], index[1]] = Rv

    if remove_diagonal:
        rv_coefficients -= np.eye(n_matrices)

    return rv_coefficients


def pairwise_congruence_coefficient(matrices, remove_diagonal=False):
    """Computes the congruence coefficient between matrices.

    Parameters
    ----------
    matrices : np.ndarray
        Set of symmetric semi-positive definite matrices. Shape is (M, N, N).
    remove_diagonal : bool, optional
        Should we remove the diagonal before calculating the correction?

    Returns
    -------
    congruence_coefficient : np.ndarray
        Matrix of pairwise congruence coefficients. Shape is (M, M).
    """

    n_matrices = matrices.shape[0]
    congruence_coefficient = np.zeros([n_matrices, n_matrices])
    for index, element in np.ndenumerate(congruence_coefficient):
        nom = np.trace(
            np.dot(np.transpose(matrices[index[0]]), matrices[index[1]]),
        )
        denom1 = np.trace(
            np.dot(np.transpose(matrices[index[0]]), matrices[index[0]]),
        )
        denom2 = np.trace(
            np.dot(np.transpose(matrices[index[1]]), matrices[index[1]]),
        )
        cc = nom / np.sqrt(np.dot(denom1, denom2))
        congruence_coefficient[index[0], index[1]] = cc

    if remove_diagonal:
        congruence_coefficient -= np.eye(n_matrices)

    return congruence_coefficient


def pairwise_l2_distance(arrays, batch_dims=0):
    """Calculate the pairwise L2 distance
    along the first axis after :code:`batch_dims`.

    Parameters
    ----------
    arrays : np.ndarray
        Set of arrays. Shape is (..., n_sessions, ...).
    batch_dims : int, optional
        Number of batch dimensions.

    Returns
    -------
    pairwise_distance : np.ndarray
        Matrix of pairwise L2 distance. Shape is (..., n_sessions, n_sessions).
    """
    if batch_dims > arrays.ndim - 1:
        raise ValueError("batch_dims must be less than arrays.ndim - 1")
    pairwise_axis = batch_dims
    return np.sqrt(
        np.sum(
            np.square(
                np.expand_dims(arrays, pairwise_axis)
                - np.expand_dims(arrays, pairwise_axis + 1)
            ),
            axis=tuple(range(pairwise_axis + 2, arrays.ndim + 1)),
        )
    )


def fisher_z_transform(r):
    """
    Fisher z-transform each element of the input
    :math:`z = \frac{1}{2}\ln(\frac{1+r}{1-r})`
    Parameters
    ----------
    r: np.ndarray
        Elements in (-1,1)

    Returns
    -------
    z: np.ndarray
        Fisher z-transformation of the input
    """
    return 0.5 * np.log((1 + r) / (1 - r))


def fisher_z_correlation(M1, M2):
    """
    Calculate the Fisher z-transformed vector correlation.
    M1 and M2 are symmetric semi-positive definite matrices. Shape is (N, N).
    Step 1: Obtain the lower triangular element of M1 and M2,
    unwrap to vectors v1, v2 with length N * (N - 1) / 2
    Step 2: Fisher z-transform vectors v1, v2 to z1, z2
    Step 3: return the Pearson's correlation between z1, z2
    Parameters
    ----------
    M1: np.ndarray
        The first matrix. The shape is (N, N).
    M2: np.ndarray
        The second matrix. The shape is (N,N).

    Returns
    -------
    dist: (float) distance between M1 and M2
    """
    N = M1.shape[0]
    assert N == M2.shape[0]

    # Obtain the upper triangular elements
    upper_indices = np.triu_indices(N, k=1)
    v1 = M1[upper_indices]
    v2 = M2[upper_indices]

    # Fisher-z-transformation
    z1 = fisher_z_transform(v1)
    z2 = fisher_z_transform(v2)

    # return the correlation
    return np.cov(z1, z2, ddof=0)[0, 1] / (np.std(z1, ddof=0) * np.std(z2, ddof=0))


def pairwise_fisher_z_correlations(matrices):
    """
    Compute the pairwise Fisher z-transformed correlations of matrices.
    Parameters
    ----------
    matrices: numpy.ndarray
        set of symmetric semi-positive definite correlation matrices. Shape is (M, N, N).

    Returns
    -------
    correlations: numpy.ndarray
        Fisher z-transformed correlation. Shape is (M, M)
    """

    N = len(matrices)
    correlation_metrics = np.eye(N)
    for i in trange(N, desc='Compute Fisher-z transformated correlation'):
        for j in range(i + 1, N):
            correlation_metrics[i][j] = fisher_z_correlation(
                matrices[i], matrices[j]
            )
            correlation_metrics[j][i] = correlation_metrics[i][j]

    return correlation_metrics


def twopair_vector_correlation(vectors_1, vectors_2):
    """
    Compute the pairwise correlation between ith vectors_1 and jth vectors_j
    Parameters
    ----------
    vectors_1: np.ndarray
        The first set of vectors with shape (N_vectors_1,vector_length)
    vectors_2: np.ndarray
        The second set of vectors with shape (N_vectors_2,vector_length)

    Returns
    -------
    correlation: np.ndarray
        The vector correlations with shape (N_vectors_1, N_vectors_2)
    """
    M1, N1 = vectors_1.shape
    M2, N2 = vectors_2.shape
    assert N1 == N2
    correlations = np.empty((M1, M2))

    for i in range(M1):
        for j in range(M2):
            correlation = np.corrcoef(vectors_1[i, :], vectors_2[j, :])[0, 1]
            correlations[i, j] = correlation
    return correlations


def twopair_fisher_z_correlations(matrices_1, matrices_2):
    """
    Compute the pairwise Fisher z-transformed correlations of matrices.
    Parameters
    ----------
    matrices_1: np.ndarray
        set of symmetric semi-positive definite correlation matrices. Shape is (M1, N, N).
    matrices_2: np.ndarray
        set of symmetric semi-positive definite correlation matrices. Shape is (M2, N, N).

    Returns
    -------
    correlations: np.ndarray
        Fisher z-transformed correlation. Shape is (M1, M2)
    """
    M1 = len(matrices_1)
    M2 = len(matrices_2)
    fisher_z_transformed_correlation = np.zeros([M1, M2])
    for i in trange(M1, desc='Compute twopair Fisher-z transformated correlation'):
        for j in range(M2):
            fisher_z_transformed_correlation[i][j] = fisher_z_correlation(
                matrices_1[i], matrices_2[j]
            )

    return fisher_z_transformed_correlation


def regularisation(matrices, eps=1e-6):
    """
    Regularise a bunch of matrices by adding eps to the diagonal
    Parameters
    ----------
    matrices: np.ndarray
        The set of matrices with shape (M, N, N) or (N, N)
    eps: float
        The regularisation on the diagonal

    Returns
    -------
    regularised_matrices: np.ndarray
        The shape is (M, N, N) or (N, N).
    """
    if matrices.ndim == 2:
        N = len(matrices)
    elif matrices.ndim == 3:
        _, N, _ = matrices.shape
    else:
        return ValueError('Matrices dimension is incorrect!')

    return matrices + np.eye(N) * eps

def twopair_riemannian_distance(matrices_1,matrices_2,eps_value=[0,1e-9,1e-8,1e-7,1e-6,1e-5],threshold:float=1e-3):
    """
    Compute the Riemannian distance between two sets of matrices
    Parameters
    ----------
    matrices_1: np.ndarray
        The first set of positive semi-definite matrices.
        The shape is (N_modes_1, N_channels, N_channels).
    matrices_1: np.ndarray
        The second set of positive semi-definite matrices.
        The shape is (N_modes_2, N_channels, N_channels).
    eps: list,optional
        The regularisation factor to try.
    threshold: float
        Threshold to apply when there are negative eigenvalues. Must be positive.
    Returns
    riemannian_distances: np.ndarray
        Riemannian distance. The shape is (N_modes_1,N_modes_2)
    -------
    """
    M1,N1,P1 = matrices_1.shape
    M2,N2,P2 = matrices_2.shape
    assert N1 == P1
    assert N1 == N2
    assert P1 == P2

    matrices_1.astype(np.float64)
    matrices_2.astype(np.float64)
    riemannian_distances = np.zeros([M1,M2])
    for eps in eps_value:
        try:
            for i in trange(M1, desc="Computing Riemannian distances"):
                for j in range(M2):
                    riemannian_distances[i][j] = riemannian_distance(
                        regularisation(matrices_1[i],eps),
                        regularisation(matrices_2[j],eps),
                        threshold=threshold
                    )
            return riemannian_distances
        except np.linalg.LinAlgError:
            warnings.warn(f'Calculation of Riemannian distance failed for eps={eps}')

    raise ValueError('Riemannian distance failed for all eps values!')
