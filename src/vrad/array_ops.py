"""Helper functions using NumPy.

"""
import logging
from typing import List

import numpy as np
from vrad.utils.decorators import transpose

_logger = logging.getLogger("VRAD")


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
