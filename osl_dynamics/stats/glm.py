"""Fitting a General Linear Model (GLM) with ordinary least squares."""

import logging
from typing import List, Optional, Tuple

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
            f"{n_affected} of {n_targets} targets have missing values (in the data "
            "or the design). "
            f"Between {dropped[dropped > 0].min()} and {dropped.max()} of "
            f"{n_samples} samples were dropped for these targets."
        )
    return groups


def residual_dof(X: np.ndarray) -> int:
    """Residual degrees of freedom of a design matrix: the number of samples
    minus its rank (0 for an empty matrix)."""
    if X.shape[0] == 0:
        return 0
    return X.shape[0] - np.linalg.matrix_rank(X)


def hat_adjustment(varcopes: np.ndarray, hat: float) -> np.ndarray:
    """'Hat' variance adjustment.

    Adds a fraction of the largest variance across targets to every target's
    variance [#ridgway2012]_, which stops targets with implausibly small variances
    producing very large t-statistics.

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
    .. [#ridgway2012] Ridgway, G. R. et al. (2012). The problem of low variance
       voxels in statistical parametric mapping; a new hat avoids a 'haircut'.
       NeuroImage, 59(3), 2131-2141.
    """
    if hat < 0:
        raise ValueError(f"hat must be non-negative, got {hat}.")
    if hat == 0:
        return varcopes
    return varcopes + hat * np.nanmax(varcopes, axis=-1, keepdims=True)


# Relative tolerances: a contrast row counts as estimable when its component
# outside the row space of the design is below this fraction of its size, and
# a residual sum of squares below this fraction of the total sum of squares
# counts as zero (a constant target)
_ESTIMABLE_RTOL = 1e-8
_ZERO_SS_RTOL = 1e-10


def estimable_contrast(C: np.ndarray, X: np.ndarray, pinv: np.ndarray) -> np.ndarray:
    """The estimable part of a contrast for a design.

    A hypothesis :math:`C \\beta = 0` can only be tested for the part of
    :math:`C` that lies in the row space of :math:`X`. This is the whole of
    :math:`C` unless the design is rank deficient in a way that involves the
    contrast, e.g. a group with no data for a target (so its indicator column
    is all zero for the complete samples of that target): the difference
    between that group and another is then not estimable, while differences
    between the remaining groups still are.

    Parameters
    ----------
    C : np.ndarray
        Contrast matrix. Shape is (n_rows, n_features).
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).
    pinv : np.ndarray
        Pseudo-inverse of :code:`X`. Shape is (n_features, n_samples).

    Returns
    -------
    C_est : np.ndarray
        Rows spanning the estimable part of the contrast. Shape is
        (n_estimable, n_features), with :code:`n_estimable <= n_rows` and
        possibly zero.
    """
    # Component of each row outside the row space of X
    outside = C - (C @ pinv) @ X
    scale = np.abs(C).max(axis=1, keepdims=True)
    if np.all(np.abs(outside) <= _ESTIMABLE_RTOL * scale):
        return C
    # Combinations of the rows that have no component outside: the left
    # null space of `outside`
    U, s, _ = np.linalg.svd(outside, full_matrices=True)
    rank = int(np.sum(s > _ESTIMABLE_RTOL * max(s.max(), np.abs(C).max())))
    A = U[:, rank:]
    return A.T @ C


def _estimable_rows(C: np.ndarray, X: np.ndarray, pinv: np.ndarray) -> np.ndarray:
    """Which rows of a contrast matrix are estimable (boolean array)."""
    outside = C - (C @ pinv) @ X
    scale = np.abs(C).max(axis=1)
    return np.all(np.abs(outside) <= _ESTIMABLE_RTOL * scale[:, None], axis=1)


def _zero_small(ss: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Set residual sums of squares that are zero up to floating point
    noise (constant targets) to exactly zero."""
    return np.where(ss <= _ZERO_SS_RTOL * total, 0.0, ss)


def f_whitening(c_pinv: np.ndarray, name: str = "") -> np.ndarray:
    """Whitening matrix for the rows of an F-contrast.

    For an F-contrast :math:`C` the copes :math:`C \\hat\\beta` have
    covariance :math:`\\sigma^2 M` with :math:`M = C (X^T X)^+ C^T`. The
    whitening matrix :math:`L` satisfies :math:`L M L^T = I`, so that the
    F-statistic is :math:`\\| L C \\hat\\beta \\|^2 / (q \\hat\\sigma^2)`
    where :math:`q` is the rank of :math:`M` (the numerator degrees of
    freedom).

    Parameters
    ----------
    c_pinv : np.ndarray
        :math:`C X^+`, the F-contrast times the pseudo-inverse of the design
        matrix. Shape is (n_rows, n_samples).
    name : str, optional
        Contrast name, for messages.

    Returns
    -------
    L : np.ndarray
        Whitening matrix. Shape is (q, n_rows).
    """
    M = c_pinv @ c_pinv.T
    eigenvalues, V = np.linalg.eigh(M)
    tol = max(eigenvalues.max(), 0) * M.shape[0] * np.finfo(float).eps
    keep = eigenvalues > tol
    if not keep.any():
        raise ValueError(f"F-contrast {name} is not estimable with this design.")
    return (V[:, keep] / np.sqrt(eigenvalues[keep])).T


def fit_ols(
    X: np.ndarray,
    y: np.ndarray,
    contrasts: np.ndarray,
    warn: bool = True,
    hat: float = 0.0,
    f_contrasts: Optional[List[np.ndarray]] = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, np.ndarray
]:
    """Fit a GLM with ordinary least squares.

    Parameters
    ----------
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).
    y : np.ndarray
        Data. Shape is (n_samples, n_targets). Targets with missing values
        are fitted with their complete samples.
    contrasts : np.ndarray
        t-contrast matrix. Shape is (n_contrasts, n_features).
    warn : bool, optional
        Log a warning when there are missing values.
    hat : float, optional
        'Hat' variance adjustment, see :func:`hat_adjustment`. For
        F-statistics the adjustment is applied to the denominator (the
        residual variance times the numerator degrees of freedom).
    f_contrasts : list of np.ndarray, optional
        F-contrast matrices. Shape of each is (n_rows, n_features).

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
    fstats : np.ndarray
        F-statistics. Shape is (n_f_contrasts, n_targets).
    f_copes : list of np.ndarray
        Copes of the rows of each F-contrast. Shape of each is
        (n_rows, n_targets).
    f_dof : np.ndarray
        Numerator degrees of freedom of each F-contrast for each target (the
        number of estimable rows). Shape is (n_f_contrasts, n_targets).

    Notes
    -----
    A contrast that is not estimable for a target (e.g. a group difference
    when the group has no data for that target) gets NaN statistics; an
    F-contrast that is partly estimable tests its estimable part with fewer
    numerator degrees of freedom. Targets with zero residual variance, or
    with too few complete samples for any residual degrees of freedom, get
    NaN statistics (the former unless :code:`hat > 0`).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    contrasts = np.asarray(contrasts, dtype=float)
    f_contrasts = [np.asarray(c, dtype=float) for c in (f_contrasts or [])]
    if X.ndim != 2 or y.ndim != 2 or contrasts.ndim != 2:
        raise ValueError("X, y and contrasts must be 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if X.shape[1] != contrasts.shape[1]:
        raise ValueError("X and contrasts must have the same number of features.")
    for c in f_contrasts:
        if c.ndim != 2 or c.shape[1] != X.shape[1]:
            raise ValueError("F-contrasts must be 2D with n_features columns.")

    n_samples, n_features = X.shape
    n_targets = y.shape[1]
    n_contrasts = contrasts.shape[0]
    n_f_contrasts = len(f_contrasts)

    betas = np.empty((n_features, n_targets))
    copes = np.empty((n_contrasts, n_targets))
    varcopes = np.empty((n_contrasts, n_targets))
    dof = np.empty(n_targets)
    f_num = np.empty((n_f_contrasts, n_targets))
    f_den = np.empty((n_f_contrasts, n_targets))
    f_copes = [np.empty((c.shape[0], n_targets)) for c in f_contrasts]
    f_dof = np.zeros((n_f_contrasts, n_targets), dtype=int)
    not_estimable = {}
    too_few = 0

    for rows, targets in complete_case_groups(X, y, warn=warn):
        X_r = X[rows]
        y_r = y[rows][:, targets]
        dof_r = residual_dof(X_r)
        dof[targets] = dof_r
        if dof_r < 1:
            too_few += len(targets)
            betas[:, targets] = np.nan
            copes[:, targets] = np.nan
            varcopes[:, targets] = np.nan
            f_num[:, targets] = np.nan
            f_den[:, targets] = np.nan
            for cb in f_copes:
                cb[:, targets] = np.nan
            continue

        pinv = np.linalg.pinv(X_r)
        b = pinv @ y_r
        residuals = y_r - X_r @ b
        ss = _zero_small(np.sum(residuals**2, axis=0), np.sum(y_r**2, axis=0))
        s2 = ss / dof_r

        # diag(C (X^T X)^+ C^T) using (X^T X)^+ = pinv pinv^T
        c_pinv = contrasts @ pinv
        c_xtx_c = np.einsum("ij,ij->i", c_pinv, c_pinv)

        betas[:, targets] = b
        copes[:, targets] = contrasts @ b
        varcopes[:, targets] = c_xtx_c[:, None] * s2[None, :]

        if n_contrasts:
            estimable = _estimable_rows(contrasts, X_r, pinv)
            for i in np.where(~estimable)[0]:
                copes[i, targets] = np.nan
                varcopes[i, targets] = np.nan
                not_estimable[f"t-contrast {i}"] = not_estimable.get(
                    f"t-contrast {i}", 0
                ) + len(targets)

        for i, C in enumerate(f_contrasts):
            C_est = estimable_contrast(C, X_r, pinv)
            cb = C @ b
            cb[~_estimable_rows(C, X_r, pinv)] = np.nan
            f_copes[i][:, targets] = cb
            f_dof[i, targets] = C_est.shape[0]
            if C_est.shape[0] < C.shape[0]:
                not_estimable[f"F-contrast {i}"] = not_estimable.get(
                    f"F-contrast {i}", 0
                ) + len(targets)
            if C_est.shape[0] == 0:
                f_num[i, targets] = np.nan
                f_den[i, targets] = np.nan
                continue
            L = f_whitening(C_est @ pinv, name=str(i))
            f_num[i, targets] = np.sum((L @ C_est @ b) ** 2, axis=0)
            f_den[i, targets] = C_est.shape[0] * s2

    if warn and too_few:
        _logger.warning(
            f"{too_few} of {n_targets} targets have too few complete samples for "
            "any residual degrees of freedom. Their statistics are NaN."
        )
    if warn:
        for name, n in not_estimable.items():
            _logger.warning(
                f"{name} is not (fully) estimable for {n} of {n_targets} targets, "
                "e.g. a group has no data for them. Its estimable part is tested, "
                "with NaN statistics where there is none."
            )

    varcopes = hat_adjustment(varcopes, hat)
    f_den = hat_adjustment(f_den, hat)
    with np.errstate(divide="ignore", invalid="ignore"):
        fstats = np.where(f_den > 0, f_num / f_den, np.nan)
    return betas, copes, varcopes, dof, fstats, f_copes, f_dof


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
    >>> result.test(contrast_index=0, correction="fdr").significant()
    >>> result.ftest(0).pvalues  # if the design has an F-contrast
    """

    def __init__(self, design: Design) -> None:
        self.design = design
        self.X = design.build_X()
        self.contrasts = design.build_contrast_array()
        self.f_contrasts = design.build_f_contrast_arrays()

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_contrasts(self) -> int:
        return self.contrasts.shape[0]

    @property
    def n_f_contrasts(self) -> int:
        return len(self.f_contrasts)

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
            Betas, copes, varcopes, degrees of freedom and, if the design
            has F-contrasts, F-statistics.
        """
        y, target_dims = validate_y(y, self.n_samples)
        betas, copes, varcopes, dof, fstats, f_copes, f_dof = fit_ols(
            self.X, y, self.contrasts, warn=warn, hat=hat, f_contrasts=self.f_contrasts
        )
        return GLMResult(
            design=self.design,
            betas=betas.reshape((self.n_features, *target_dims)),
            copes=copes.reshape((self.n_contrasts, *target_dims)),
            varcopes=varcopes.reshape((self.n_contrasts, *target_dims)),
            dof=dof.reshape(target_dims),
            fstats=fstats.reshape((self.n_f_contrasts, *target_dims)),
            f_copes=[c.reshape((c.shape[0], *target_dims)) for c in f_copes],
            f_dof=f_dof.reshape((self.n_f_contrasts, *target_dims)),
            data=y.reshape((self.n_samples, *target_dims)),
        )
