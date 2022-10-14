"""Functions to perform regression analysis.

"""

import numpy as np
from sklearn.linear_model import LinearRegression


def linear(X, y, fit_intercept, normalize=False, print_message=True):
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
    print_message : bool
        Should we print a message?

    Returns
    -------
    coefs : np.ndarray
        2D or higher dimension array. Regression coefficients.
    intercept : np.ndarray
        1D or higher dimension array. Regression intercept.
        Returned if fit_intercept=True.
    """
    if print_message:
        print("Fitting linear regression")

    # Reshape in case non 2D matrices were passed
    original_shape = y.shape
    new_shape = [X.shape[1]] + list(original_shape[1:])
    y = y.reshape(original_shape[0], -1)

    # Normalise the regressors
    if normalize:
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)

    if y.dtype == np.complex64 or y.dtype == np.complex_:
        # Fit two linear regressions:
        # One for the real part
        reg = LinearRegression(fit_intercept=fit_intercept, n_jobs=-1)
        reg.fit(X, y.real)
        coefs_real = reg.coef_.T.reshape(new_shape)
        if fit_intercept:
            intercept_real = reg.intercept_.reshape(new_shape[1:])

        # Another for the imaginary part
        reg = LinearRegression(fit_intercept=fit_intercept, n_jobs=-1)
        reg.fit(X, y.imag)
        coefs_imag = reg.coef_.T.reshape(new_shape)
        if fit_intercept:
            intercept_imag = reg.intercept_.reshape(new_shape[1:])

        # Regression parameters
        coefs = coefs_real + 1j * coefs_imag
        if fit_intercept:
            intercept = intercept_real + 1j * intercept_imag

    else:
        # Only need to fit one linear regression
        reg = LinearRegression(fit_intercept=fit_intercept, n_jobs=-1)
        reg.fit(X, y)

        # Regression parameters
        coefs = reg.coef_.T.reshape(new_shape)
        if fit_intercept:
            intercept = reg.intercept_.reshape(new_shape[1:])

    if fit_intercept:
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
