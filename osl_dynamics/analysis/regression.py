"""Functions to perform regression analysis.

"""

import logging

import numpy as np
from sklearn.linear_model import LinearRegression

_logger = logging.getLogger("osl-dynamics")


def linear(X, y, fit_intercept, normalize=False, log_message=False):
    """Wrapper for `sklearn.linear_model.LinearRegression \
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model\
    .LinearRegression.html>`_.

    Parameters
    ----------
    X : np.ndarray
        Regressors, should be a 2D array (n_targets, n_regressors).
    y : np.ndarray
        Targets. Should be a 2D array: (n_targets, n_features).
        If a higher dimension array is passed, the extra dimensions
        are concatenated.
    fit_intercept : bool
        Should we fit an intercept?
    normalize : bool, optional
        Should we z-transform the regressors?
    log_message : bool, optional
        Should we log a message?

    Returns
    -------
    coefs : np.ndarray
        Regression coefficients. 2D array or higher dimensionality:
        (n_regressors, n_features).
    intercept : np.ndarray
        Regression intercept. 1D array or higher dimensionality:
        (n_features,). Returned if :code:`fit_intercept=True`.
    """
    if log_message:
        _logger.info("Fitting linear regression")

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
