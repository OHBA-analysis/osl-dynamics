"""Functions to perform regression analysis.

"""

import numpy as np


def pinv(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Find the parameters of a regression using a pseudo inverse.

    If A = B @ C, where A, B and C are 2D matrices. This function calculates
    C using A and B: C = pinv(B) @ A.

    Parameters
    ----------
    A : np.ndarray
        First matrix. If this is not a 2D matrix, the extra dimensions are
        concatenated.
    B : np.ndarray
        Second array. Must be 2D.

    Returns
    -------
    C : np.ndarray
    """
    original_shape = A.shape
    new_shape = [B.shape[1]] + list(original_shape[1:])
    A = A.reshape(original_shape[0], -1)
    C = np.linalg.pinv(B) @ A
    C = C.reshape(new_shape)
    return C
