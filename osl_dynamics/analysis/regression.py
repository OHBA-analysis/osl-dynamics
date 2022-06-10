"""Functions to perform regression analysis.

"""

import numpy as np
from sklearn.linear_model import LinearRegression
from osl_dynamics.data.processing import standardize


def linear(X, y, fit_intercept, normalize=False):
    """Wrapper for sklearn's LinearRegression.

    Parameters
    ----------
    X : np.ndarray
        2D matrix. Regressors.
    y : np.ndarray
        2D matrix. Targets. If a higher dimension array is passed, the extra
        dimensions are concatenated.
    fit_intercept : bool
        Should we fit an intercept?
    normalize : bool
        Should we z-transform the regressors?

    Returns
    -------
    coefs : np.ndarray
        2D or higher dimension array. Regression coefficients.
    intercept : np.ndarray
        1D or higher dimension array. Regression intercept.
        Returned if fit_intercept=True.
    """

    # Reshape in case non 2D matrices were passed
    original_shape = y.shape
    new_shape = [X.shape[1]] + list(original_shape[1:])
    y = y.reshape(original_shape[0], -1)

    # Normalise the regressors
    if normalize:
        X = standardize(X)

    # Fit linear regression
    reg = LinearRegression(fit_intercept=fit_intercept, n_jobs=-1)
    reg.fit(X, y)

    # Return regression coefficients and intercept
    coefs = reg.coef_.T.reshape(new_shape)
    if fit_intercept:
        intercept = reg.intercept_.reshape(new_shape[1:])
        return coefs, intercept
    else:
        return coefs


def pinv(X, y):
    """Find the parameters of a linear regression using a pseudo inverse.

    If y = X @ b, where y, X and b are 2D matrices. This function calculates
    b using y and X: b = pinv(X) @ y.

    Parameters
    ----------
    X : np.ndarray
        2D matrix. Regressors.
    y : np.ndarray
        2D matrix. Targets. If a higher dimension array is passed, the extra
        dimensions are concatenated.

    Returns
    -------
    b : np.ndarray
        2D or higher dimension matrix. Regression coefficients.
    """
    original_shape = y.shape
    new_shape = [X.shape[1]] + list(original_shape[1:])
    y = y.reshape(original_shape[0], -1)
    b = np.linalg.pinv(X) @ y
    b = b.reshape(new_shape)
    return b
