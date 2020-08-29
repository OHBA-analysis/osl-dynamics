"""Helper functions using NumPy.

"""
from typing import List, Tuple

import numpy as np
import scipy.special
from scipy.optimize import linear_sum_assignment
from vrad.utils.decorators import transpose


def softplus(time_course: np.ndarray):
    """Calculate the softplus activation of a time series."""
    time_course = np.asarray(time_course)
    zero = np.asarray(0).astype(time_course.dtype)
    return np.logaddexp(zero, time_course)


def softmax(time_course: np.ndarray):
    """Calculate the softmax activation of a time series over the last axis."""
    time_course = np.asarray(time_course)
    return scipy.special.softmax(time_course, axis=-1)


def match_matrices(*matrices: np.ndarray) -> Tuple[np.ndarray]:
    """Matches matrices based on Frobenius norm of the difference of the matrices.

    Each matrix must be 3D: (n_states, n_channels, n_channels).

    The Frobenius norm is F = [Sum_{i,j} abs(a_{ij}^2)]^0.5,
    where A is the element-wise difference of two matrices.
    """
    # Check all matrices have the same shape
    for matrix in matrices[1:]:
        if matrix.shape != matrices[0].shape:
            raise ValueError("Matrices must have the same shape.")

    # Number of arguments and number of matrices in each argument passed
    n_args = len(matrices)
    n_matrices = matrices[0].shape[0]

    # Calculate the similarity between matrices
    F = np.empty([n_matrices, n_matrices])
    matched_matrices = [matrices[0]]
    for i in range(1, n_args):
        for j in range(n_matrices):
            # Find the matrix that is most similar to matrix j
            for k in range(n_matrices):
                A = abs(np.diagonal(matrices[i][k]) - np.diagonal(matrices[0][j]))
                F[j, k] = np.linalg.norm(A)
        order = linear_sum_assignment(F)[1]

        # Add the ordered matrix to the list
        matched_matrices.append(matrices[i][order])

    return tuple(matched_matrices)


def get_one_hot(values: np.ndarray, n_states: int = None):
    """Expand a categorical variable to a series of boolean columns (one-hot encoding).

    +----------------------+
    | Categorical Variable |
    +======================+
    |           A          |
    +----------------------+
    |           C          |
    +----------------------+
    |           D          |
    +----------------------+
    |           B          |
    +----------------------+

    becomes

    +---+---+---+---+
    | A | B | C | D |
    +===+===+===+===+
    | 1 | 0 | 0 | 0 |
    +---+---+---+---+
    | 0 | 0 | 1 | 0 |
    +---+---+---+---+
    | 0 | 0 | 0 | 1 |
    +---+---+---+---+
    | 0 | 1 | 0 | 0 |
    +---+---+---+---+

    Parameters
    ----------
    values : numpy.ndarray
        Categorical variable in a 1D array. Values should be integers (i.e. state 0, 1,
        2, 3, ... , `n_states`).
    n_states : int
        Total number of states in `values`. Must be at least the number of states
        present in `values`. Default is the number of unique values in `values`.

    Returns
    -------
    one_hot : numpy.ndarray
        A 2D array containing the one-hot encoded form of the input data.

    """
    if values.ndim == 2:
        _logger.info("argmax being taken on shorter axis.")
        values = values.argmax(axis=1)
    if n_states is None:
        n_states = values.max() + 1
    res = np.eye(n_states)[np.array(values).reshape(-1)]
    return res.reshape(list(values.shape) + [n_states])


@transpose(0, "sequence_1", 1, "sequence_2")
def align_arrays(*sequences, alignment: str = "left") -> List[np.ndarray]:
    """Given a list of sequences, return the sequences trimmed to equal length.

    Given a list of sequences of unequal length, remove either the start, end or a
    portion of both the start and end of the arrays such that their lengths are equal
    to the length of the shortest array.

    If alignment is "left", values will be trimmed from the ends of the arrays
    (i.e. the starts of the arrays will be aligned). If "right", values will be trimmed
    from the starts of the arrays (i.e. the ends will be aligned). If "center", an
    equal amount will be trimmed from the start and end of the array (i.e. the arrays
    are aligned by their middle values.


    Parameters
    ----------
    sequences: list of numpy.ndarray
        Time courses with differing lengths.
    alignment: str
        One of "left", "center" and "right".
    Returns
    -------
    aligned_arrays: list of numpy.ndarray
    """
    min_length = min(len(sequence) for sequence in sequences)

    if alignment == "left":
        return [sequence[:min_length] for sequence in sequences]

    elif alignment == "right":
        return [sequence[-min_length:] for sequence in sequences]
    elif alignment == "center":
        half_length = int(min_length / 2)
        mids = [int(len(sequence) / 2) for sequence in sequences]

        return [
            sequence[mid - half_length : mid + half_length]
            for sequence, mid in zip(sequences, mids)
        ]

    else:
        raise ValueError("Alignment must be left, right or center.")


def from_cholesky(cholesky_matrix: np.ndarray):
    """Given a Cholesky matrix return the recomposed matrix.

    Operates on the assumption that cholesky_matrix is a valid Cholesky decomposition
    A = LL* and performs LL^T to recover A.

    Parameters
    ----------
    cholesky_matrix: numpy.ndarray
        A valid Cholesky decomposition.

    Returns
    -------
    full_matrix: numpy.ndarray
        A = LL^T where L is the Cholesky decomposition of A.
    """
    if cholesky_matrix.ndim == 2:
        return cholesky_matrix @ cholesky_matrix.transpose()
    return cholesky_matrix @ cholesky_matrix.transpose((0, 2, 1))


@transpose(0, "state_time_course")
def calculate_trans_prob_matrix(
    state_time_course: np.ndarray, zero_diagonal: bool = False, n_states: int = None,
) -> np.ndarray:
    """For a given state time course, calculate the transition probability matrix.

    If a 2D array is given, argmax(axis=1) will be performed upon it before proceeding.

    Parameters
    ----------
    state_time_course: numpy.ndarray
    zero_diagonal: bool
        If True, return the array with diagonals set to zero.
    n_states: int
        The number of states in the state time course. Default is to take the highest
        state number present in a 1D time course or the number of columns in a 2D
        (one-hot encoded) time course.

    Returns
    -------

    """
    if state_time_course.ndim == 2:
        n_states = state_time_course.shape[1]
        state_time_course = state_time_course.argmax(axis=1)
    if state_time_course.ndim != 1:
        raise ValueError("state_time_course should either be 1D or 2D.")

    vals, counts = np.unique(
        state_time_course[
            np.arange(2)[None, :] + np.arange(len(state_time_course) - 1)[:, None]
        ],
        axis=0,
        return_counts=True,
    )

    if n_states is None:
        n_states = state_time_course.max() + 1

    trans_prob = np.zeros((n_states, n_states))
    trans_prob[vals[:, 0], vals[:, 1]] = counts

    with np.errstate(divide="ignore", invalid="ignore"):
        trans_prob = trans_prob / trans_prob.sum(axis=1)[:, None]
    trans_prob = np.nan_to_num(trans_prob)

    if zero_diagonal:
        np.fill_diagonal(trans_prob, 0)
    return trans_prob


def trace_normalize(matrix: np.ndarray):
    """Given a matrix, divide all of its values by the sum of its diagonal.

    Parameters
    ----------
    matrix: numpy.ndarray

    Returns
    -------
    normalized_matrix: numpy.ndarray
        trace(M) = 1

    """
    matrix = np.array(matrix)
    if matrix.ndim == 2:
        return matrix / matrix.trace()
    if matrix.ndim != 3:
        raise ValueError("Matrix should be 2D or 3D.")

    return matrix / matrix.trace(axis1=1, axis2=2)[:, None, None]


def mean_diagonal(array: np.ndarray):
    """Set the diagonal of a matrix to the mean of all non-diagonal elements.

    This is primarily useful for plotting without being concerned about the magnitude
    of diagonal values compressing the color scale.

    Parameters
    ----------
    array: numpy.ndarray

    Returns
    -------
    mean_diagonal_array: numpy.ndarray
        Array with diagonal set to mean of non-diagonal elements.

    """
    off_diagonals = ~np.eye(array.shape[0], dtype=bool)
    new_array = array.copy()
    np.fill_diagonal(new_array, array[off_diagonals].mean())
    return new_array


@transpose
def batch(
    array: np.ndarray,
    window_size: int,
    step_size: int = None,
    selection: np.ndarray = slice(None),
):
    step_size = step_size or window_size
    final_slice_start = array.shape[0] - window_size + 1
    index = np.arange(0, final_slice_start, step_size)[:, None] + np.arange(window_size)
    return array[index[selection]]
