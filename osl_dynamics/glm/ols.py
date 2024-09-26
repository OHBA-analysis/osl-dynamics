import logging

import numpy as np
from sklearn.linear_model import LinearRegression

_logger = logging.getLogger("osl-dynamics")


def _validate_dimensions(X=None, y=None, contrasts=None):
    """
    Validate dimensions of input arrays.
    """
    # Check X dimensions
    if X is None:
        X_n_samples, X_n_features = None, None
    elif X.ndim == 2:
        X_n_samples, X_n_features = X.shape
    else:
        raise ValueError(f"X must be 2D, got {X.ndim}D")

    # Check y dimensions
    if y is None:
        y_n_samples = None
    elif y.ndim == 2:
        y_n_samples = y.shape[0]
    else:
        raise ValueError(f"y must be 2D, got {y.ndim}D")

    # Check contrasts dimensions
    if contrasts is None:
        contrasts_n_features = None
    elif contrasts.ndim == 2:
        contrasts_n_features = contrasts.shape[1]
    else:
        raise ValueError(f"contrasts must be 2D, got {contrasts.ndim}D")

    # Validate dimensions
    if (
        X_n_samples is not None
        and y_n_samples is not None
        and X_n_samples != y_n_samples
    ):
        raise ValueError(
            f"X and y must have the same number of samples. Got {X_n_samples} samples in X and {y_n_samples} samples in y."
        )
    if (
        X_n_features is not None
        and contrasts_n_features is not None
        and X_n_features != contrasts_n_features
    ):
        raise ValueError(
            f"X and contrasts must have the same number of features. Got {X_n_features} features in X and {contrasts_n_features} features in contrasts."
        )


def get_residuals(X, y, predictor):
    """
    Get residuals from a linear model.

    Parameters
    ----------
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).
    y : np.ndarray
        Target variable. Shape is (n_samples, n_targets).
    predictor : sklearn.linear_model.LinearRegression
        Sklearn LinearRegression object.

    Returns
    -------
    residuals : np.ndarray
        Residuals. Shape is (n_samples, n_targets).
    """
    _validate_dimensions(X=X, y=y)
    if not isinstance(predictor, LinearRegression):
        raise ValueError(
            f"predictor must be a LinearRegression object, got {type(predictor)}"
        )
    return y - predictor.predict(X)


def get_degree_of_freedom(X):
    """
    Get degree of freedom.

    Parameters
    ----------
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).

    Returns
    -------
    dof : int
        Degree of freedom.
    """
    _validate_dimensions(X=X)
    return X.shape[0] - np.linalg.matrix_rank(X)


def get_varcopes(X, y, contrasts, predictor):
    """
    Get the variance of the copes.

    Parameters
    ----------
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).
    y : np.ndarray
        Target variable. Shape is or (n_samples, n_targets).
    contrasts : np.ndarray
        Contrasts matrix. Shape is (n_contrasts, n_features).
    predictor : sklearn.linear_model.LinearRegression
        Sklearn LinearRegression object.

    Returns
    -------
    varcopes : np.ndarray
        Variance of the copes. Shape is (n_contrasts, n_targets).
    """
    _validate_dimensions(X=X, y=y, contrasts=contrasts)
    if not isinstance(predictor, LinearRegression):
        raise ValueError(
            f"predictor must be a LinearRegression object, got {type(predictor)}"
        )

    xxt = X.T @ X
    xxt_inv = np.linalg.pinv(xxt)
    c_xxt_inv_ct = np.diag(contrasts @ xxt_inv @ contrasts.T)  # Shape is (n_contrasts,)

    # Get estimate of standard error
    residuals = get_residuals(X, y, predictor)
    dof = get_degree_of_freedom(X)
    s2 = np.sum(residuals**2, axis=0) / dof  # Shape is (n_targets,)

    varcopes = c_xxt_inv_ct[:, None] * s2[None, :]  # Shape is (n_contrasts, n_targets)
    return varcopes


def osl_fit(X, y, contrasts):
    """
    Fit Ordinary Least Squares (OLS) model.

    Parameters
    ----------
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).
    y : np.ndarray
        Target variable. Shape is (n_samples, n_targets).
    contrasts : np.ndarray
        Contrasts matrix. Shape is (n_contrasts, n_features).

    Returns
    -------
    betas : np.ndarray
        Betas (regression coefficients). Shape is (n_features, n_targets).
    copes : np.ndarray
        Contrast parameter estimates. Shape is (n_contrasts, n_targets).
    varcopes : np.ndarray
        Variance of the copes. Shape is (n_contrasts, n_targets).
    """
    _validate_dimensions(X=X, y=y, contrasts=contrasts)
    # TODO: Other imputation methods
    X, y = remove_nan_rows(X, y)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)
    betas = lr.coef_.T
    copes = contrasts @ betas
    varcopes = get_varcopes(X, y, contrasts, lr)

    return betas, copes, varcopes


def remove_nan_rows(X, y):
    """
    Remove rows with NaN values.

    Parameters
    ----------
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).
    y : np.ndarray
        Target variable. Shape is (n_samples, n_targets).

    Returns
    -------
    X_copy : np.ndarray
        Design matrix without NaN rows. Shape is (n_samples', n_features).
    y_copy : np.ndarray
        Target variable without NaN rows. Shape is (n_samples', n_targets).
    """
    X_copy = X.copy()
    y_copy = y.copy()

    X_nan_rows = np.any(np.isnan(X_copy), axis=1)
    y_nan_rows = np.any(np.isnan(y_copy), axis=1)
    nan_rows = np.logical_or(X_nan_rows, y_nan_rows)

    if np.sum(nan_rows) > 0:
        _logger.info(f"Removing {np.sum(nan_rows)} rows with NaN values.")
    X_copy = X_copy[~nan_rows]
    y_copy = y_copy[~nan_rows]

    return X_copy, y_copy
