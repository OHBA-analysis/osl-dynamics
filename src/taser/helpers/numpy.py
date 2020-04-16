"""Helper functions using NumPy

"""
import numpy as np


def get_one_hot(values: np.ndarray, n_states: int):
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
        present in `values`.

    Returns
    -------
    one_hot : numpy.ndarray
        A 2D array containing the one-hot encoded form of the input data.

    """
    res = np.eye(n_states)[np.array(values).reshape(-1)]
    return res.reshape(list(values.shape) + [n_states])
