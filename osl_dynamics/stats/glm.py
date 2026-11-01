"""Fitting a General Linear Model (GLM) with ordinary least squares."""

import logging
from typing import List, Tuple

import numpy as np

from osl_dynamics.stats.design import Design
from osl_dynamics.stats.results import GLMResult

_logger = logging.getLogger("osl-dynamics")


def validate_y(y, n_samples: int) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Check the data and flatten the target dimensions.

    Parameters
    ----------
    y : np.ndarray or list
        Data. Shape is (n_samples, \\*target_dims).
    n_samples : int
        Expected number of samples (rows of the design matrix).

    Returns
    -------
    y : np.ndarray
        Data as floats. Shape is (n_samples, n_targets).
    target_dims : tuple of int
        Original target dimensions.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] != n_samples:
        raise ValueError(
            f"Data has {y.shape[0]} samples but the design has {n_samples}."
        )
    target_dims = y.shape[1:]
    return y.reshape(n_samples, -1), target_dims


def complete_case_groups(
    X: np.ndarray, y: np.ndarray, warn: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Group the targets by their pattern of missing values.

    Each target is fitted with the samples that are complete for that
    target (per-target complete-case analysis).

    Parameters
    ----------
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).
    y : np.ndarray
        Data. Shape is (n_samples, n_targets).
    warn : bool, optional
        Log a warning when there are missing values.

    Returns
    -------
    groups : list of tuple
        One :code:`(rows, targets)` pair per missing-value pattern, where
        :code:`rows` is a boolean mask of the complete samples and
        :code:`targets` the indices of the targets with that pattern.
    """
    n_samples, n_targets = y.shape
    missing = np.isnan(y) | np.isnan(X).any(axis=1)[:, None]
    if not missing.any():
        return [(np.ones(n_samples, dtype=bool), np.arange(n_targets))]

    patterns, inverse = np.unique(missing, axis=1, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    groups = [
        (~patterns[:, j], np.where(inverse == j)[0]) for j in range(patterns.shape[1])
    ]

    if warn:
        dropped = missing.sum(axis=0)
        n_affected = int((dropped > 0).sum())
        _logger.warning(
            f"{n_affected} of {n_targets} targets have missing values. "
            f"Between {dropped[dropped > 0].min()} and {dropped.max()} of "
            f"{n_samples} samples were dropped for these targets."
        )
    return groups


def _check_enough_samples(n_complete: int, n_features: int) -> None:
    if n_complete < n_features + 2:
        raise ValueError(
            f"Only {n_complete} complete samples for a design with {n_features} "
            "features. At least n_features + 2 are needed."
        )


def hat_adjustment(varcopes: np.ndarray, hat: float) -> np.ndarray:
    """'Hat' variance adjustment [1]_.

    Adds a fraction of the largest variance across targets to every target's
    variance, which stops targets with implausibly small variances producing
    very large t-statistics.

    Parameters
    ----------
    varcopes : np.ndarray
        Variances. Shape is (..., n_targets).
    hat : float
        Fraction of the maximum variance to add, e.g. :code:`1e-3`.
        :code:`0` leaves the variances unchanged.

    Returns
    -------
    varcopes : np.ndarray
        Adjusted variances.

    References
    ----------
    .. [1] Ridgway, G. R. et al. (2012). The problem of low variance voxels
           in statistical parametric mapping; a new hat avoids a 'haircut'.
           NeuroImage, 59(3), 2131-2141.
    """
    if hat < 0:
        raise ValueError(f"hat must be non-negative, got {hat}.")
    if hat == 0:
        return varcopes
    return varcopes + hat * np.nanmax(varcopes, axis=-1, keepdims=True)


def fit_ols(
    X: np.ndarray,
    y: np.ndarray,
    contrasts: np.ndarray,
    warn: bool = True,
    hat: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a GLM with ordinary least squares.

    Parameters
    ----------
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).
    y : np.ndarray
        Data. Shape is (n_samples, n_targets). Targets with missing values
        are fitted with their complete samples.
    contrasts : np.ndarray
        Contrast matrix. Shape is (n_contrasts, n_features).
    warn : bool, optional
        Log a warning when there are missing values.
    hat : float, optional
        'Hat' variance adjustment, see :func:`hat_adjustment`.

    Returns
    -------
    betas : np.ndarray
        Regression coefficients. Shape is (n_features, n_targets).
    copes : np.ndarray
        Contrast parameter estimates. Shape is (n_contrasts, n_targets).
    varcopes : np.ndarray
        Variance of the copes. Shape is (n_contrasts, n_targets).
    dof : np.ndarray
        Degrees of freedom of each target. Shape is (n_targets,).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    contrasts = np.asarray(contrasts, dtype=float)
    if X.ndim != 2 or y.ndim != 2 or contrasts.ndim != 2:
        raise ValueError("X, y and contrasts must be 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if X.shape[1] != contrasts.shape[1]:
        raise ValueError("X and contrasts must have the same number of features.")

    n_samples, n_features = X.shape
    n_targets = y.shape[1]
    n_contrasts = contrasts.shape[0]

    betas = np.empty((n_features, n_targets))
    copes = np.empty((n_contrasts, n_targets))
    varcopes = np.empty((n_contrasts, n_targets))
    dof = np.empty(n_targets)

    for rows, targets in complete_case_groups(X, y, warn=warn):
        X_r = X[rows]
        y_r = y[rows][:, targets]
        _check_enough_samples(X_r.shape[0], n_features)

        pinv = np.linalg.pinv(X_r)
        dof_r = X_r.shape[0] - np.linalg.matrix_rank(X_r)
        b = pinv @ y_r
        residuals = y_r - X_r @ b
        s2 = np.sum(residuals**2, axis=0) / dof_r

        # diag(C (X^T X)^+ C^T) using (X^T X)^+ = pinv pinv^T
        c_pinv = contrasts @ pinv
        c_xtx_c = np.einsum("ij,ij->i", c_pinv, c_pinv)

        betas[:, targets] = b
        copes[:, targets] = contrasts @ b
        varcopes[:, targets] = c_xtx_c[:, None] * s2[None, :]
        dof[targets] = dof_r

    varcopes = hat_adjustment(varcopes, hat)
    return betas, copes, varcopes, dof


class GLM:
    """General Linear Model fitted with ordinary least squares.

    Parameters
    ----------
    design : Design
        The design (regressors and contrasts).

    Examples
    --------
    >>> glm = GLM(design)
    >>> result = glm.fit(data)  # data has shape (n_samples, *target_dims)
    >>> result.copes.shape      # (n_contrasts, *target_dims)
    >>> result.test(contrast_indx=0, correction="fdr").significant()
    """

    def __init__(self, design: Design) -> None:
        self.design = design
        self.X = design.build_X()
        self.contrasts = design.build_contrast_array()

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_contrasts(self) -> int:
        return self.contrasts.shape[0]

    def fit(self, y, warn: bool = True, hat: float = 0.0) -> GLMResult:
        """Fit the model.

        Parameters
        ----------
        y : np.ndarray or list
            Data. Shape is (n_samples, \\*target_dims).
        warn : bool, optional
            Log a warning when there are missing values.
        hat : float, optional
            'Hat' variance adjustment, see :func:`hat_adjustment`.

        Returns
        -------
        result : GLMResult
            Betas, copes, varcopes and degrees of freedom.
        """
        y, target_dims = validate_y(y, self.n_samples)
        betas, copes, varcopes, dof = fit_ols(
            self.X, y, self.contrasts, warn=warn, hat=hat
        )
        return GLMResult(
            design=self.design,
            betas=betas.reshape((self.n_features, *target_dims)),
            copes=copes.reshape((self.n_contrasts, *target_dims)),
            varcopes=varcopes.reshape((self.n_contrasts, *target_dims)),
            dof=dof.reshape(target_dims),
        )
