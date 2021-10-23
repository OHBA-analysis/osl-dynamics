"""Functions to perform regression analysis.

"""

import numpy as np
from sklearn.linear_model import LinearRegression


def pinv(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Find the parameters of a regression using a pseudo inverse.

    If A = B @ C, where A, B and C are 2D matrices. This function calculates
    C using A and B: C = pinv(B) @ A.

    Parameters
    ----------
    B : np.ndarray
        2D matrix.
    A : np.ndarray
        2D matrix. If a higher dimension array is passed, the extra
        dimensions are concatenated.

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


def sklearn_linear_regression(
    X: np.ndarray, y: np.ndarray, print_message: bool = True, **kwargs
) -> np.ndarray:
    """Linear regression from sklearn.

    Wrapper for sklearn's LinearRegression. Fits te model:  y = X b.

    Parameters
    ----------
    X : np.ndarray
        Regressors.
    y : np.ndarray
        Target.
    print_message : bool
        Should we print a message? Optional.

    Returns
    -------
    b : np.ndarray
        Coefficients.
    """
    if print_message:
        print("Fitting linear regression")

    # Reshape in case non 2D matrices were passed
    original_shape = y.shape
    new_shape = [X.shape[1]] + list(original_shape[1:])
    y = y.reshape(original_shape[0], -1)

    # Fit the linear regression
    reg = LinearRegression(**kwargs)
    reg.fit(X, y)

    # Reshape the inferred coefficients to match the input
    b = reg.coef_.T
    b = b.reshape(new_shape)

    return b
