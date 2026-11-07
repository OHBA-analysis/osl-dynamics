"""Effect sizes with bootstrap confidence intervals."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import stats as scipy_stats

from osl_dynamics.stats.design import Design
from osl_dynamics.stats.glm import fit_ols, validate_y
from osl_dynamics.utils.progress import progress

_logger = logging.getLogger("osl-dynamics")


@dataclass
class EffectSize:
    """Effect sizes of a contrast with confidence intervals.

    Parameters
    ----------
    contrast : str
        Name of the contrast.
    statistic : str
        :code:`'t'` or :code:`'F'`.
    ci : float
        Confidence level of the intervals.
    n_boot : int
        Number of bootstrap samples (0: parametric interval for the cope).
    cope : np.ndarray
        Contrast estimate, e.g. the group difference in the units of the
        data. Shape is (\\*target_dims), or (n_rows, \\*target_dims) for an
        F-contrast.
    cope_ci : np.ndarray
        Lower and upper confidence limits of the cope. Shape is
        (2, \\*cope.shape).
    d : np.ndarray, optional
        Cohen's d: the cope divided by the residual standard deviation of the
        GLM. For a group difference this is the difference in means in units
        of the (pooled, covariate-adjusted) standard deviation; for a
        one-sample test the mean in units of the standard deviation; for a
        within-subject design the standard deviation is that of the
        within-subject residuals. Shape is (\\*target_dims). t-tests only.
    d_ci : np.ndarray, optional
        Confidence limits of :code:`d`. Shape is (2, \\*target_dims).
    eta2 : np.ndarray, optional
        Partial eta squared: the fraction of the variance not explained by
        the other regressors that the contrast explains. Shape is
        (\\*target_dims). F-tests only.
    eta2_ci : np.ndarray, optional
        Confidence limits of :code:`eta2`. Shape is (2, \\*target_dims).
    """

    contrast: str
    statistic: str
    ci: float
    n_boot: int
    cope: np.ndarray
    cope_ci: np.ndarray
    d: Optional[np.ndarray] = None
    d_ci: Optional[np.ndarray] = None
    eta2: Optional[np.ndarray] = None
    eta2_ci: Optional[np.ndarray] = None

    def summary(self) -> Dict:
        """Summary over the targets."""

        def span(x):
            return (float(np.nanmin(x)), float(np.nanmax(x)))

        summary = {
            "contrast": self.contrast,
            "statistic": self.statistic,
            "ci": self.ci,
            "n_boot": self.n_boot,
            "cope (min, max)": span(self.cope),
        }
        if self.d is not None:
            summary["d (min, max)"] = span(self.d)
        if self.eta2 is not None:
            summary["eta2 (min, max)"] = span(self.eta2)
        return summary

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.summary().items())
        return f"EffectSize({items})"


def _resample_rows(strata: list, blocks: Optional[np.ndarray]) -> np.ndarray:
    """Rows of one bootstrap sample: blocks (e.g. subjects) are resampled
    with replacement when there are blocks, otherwise rows within each
    stratum (e.g. group) to keep the design balanced."""
    if blocks is not None:
        labels = np.unique(blocks)
        chosen = np.random.choice(labels, size=len(labels), replace=True)
        return np.concatenate([np.where(blocks == label)[0] for label in chosen])
    return np.concatenate(
        [np.random.choice(rows, size=len(rows), replace=True) for rows in strata]
    )


def _strata(design: Design, contrast) -> list:
    """Rows grouped by the categorical features of the contrast (one
    stratum for the rows in none of them), or a single stratum."""
    X = design.build_X()
    indices = contrast.feature_indices
    if not np.all(np.array(design.feature_types)[indices] == "categorical"):
        return [np.arange(X.shape[0])]
    membership = X[:, indices] != 0
    strata = [np.where(membership[:, j])[0] for j in range(len(indices))]
    rest = np.where(~membership.any(axis=1))[0]
    return [s for s in strata + [rest] if len(s)]


def _estimates(X, y, contrast_matrix, f_contrast, statistic):
    """Cope, d and partial eta squared from one fit."""
    if statistic == "t":
        betas, copes, varcopes, dof, *_ = fit_ols(X, y, contrast_matrix, warn=False)
        residuals = y - X @ betas
        s2 = np.nansum(residuals**2, axis=0) / dof
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(s2 > 0, copes[0] / np.sqrt(s2), np.nan)
        return copes[0], d
    *_, fstats, f_copes, f_dof = fit_ols(
        X, y, np.zeros((0, X.shape[1])), warn=False, f_contrasts=[f_contrast]
    )
    dof = X.shape[0] - np.linalg.matrix_rank(X)
    ss_effect = fstats[0] * f_dof[0]
    with np.errstate(invalid="ignore"):
        eta2 = ss_effect / (ss_effect + dof)
    return f_copes[0], eta2


def effect_size(
    design: Design,
    data: np.ndarray,
    contrast_index: int = 0,
    statistic: str = "t",
    blocks: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> EffectSize:
    """Effect sizes of a contrast with bootstrap confidence intervals.

    The samples (rows) are resampled with replacement :code:`n_boot` times
    and the GLM refitted each time; the confidence interval is the
    percentile interval of the bootstrap estimates. Samples are resampled
    within each group of a categorical contrast so that the design stays
    balanced, and whole blocks are resampled when there are blocks (e.g.
    subjects in a within-subject design).

    Parameters
    ----------
    design : Design
        The design.
    data : np.ndarray
        Data. Shape is (n_samples, \\*target_dims).
    contrast_index : int, optional
        Index of the contrast (among the t-contrasts or the F-contrasts).
    statistic : str, optional
        :code:`'t'` or :code:`'F'`.
    blocks : np.ndarray, optional
        Exchangeability blocks, see
        :class:`~osl_dynamics.stats.permutation.Permutation`.
    n_boot : int, optional
        Number of bootstrap samples. :code:`0` gives the parametric interval
        :code:`cope +/- t * sqrt(varcope)` for the cope only.
    ci : float, optional
        Confidence level.

    Returns
    -------
    effect_size : EffectSize
    """
    if not 0 < ci < 1:
        raise ValueError(f"ci must be between 0 and 1, got {ci}.")
    if statistic not in ("t", "F"):
        raise ValueError(f"statistic must be 't' or 'F', got {statistic}.")
    contrasts = design.contrasts if statistic == "t" else design.f_contrasts
    if not 0 <= contrast_index < len(contrasts):
        raise ValueError(
            f"contrast_index must be between 0 and {len(contrasts) - 1} (the "
            f"design has {len(contrasts)} {statistic}-contrasts)."
        )
    contrast = contrasts[contrast_index]
    X = design.build_X()
    contrast_matrix = contrast.matrix if statistic == "t" else np.zeros((0, X.shape[1]))
    f_contrast = contrast.matrix if statistic == "F" else None

    y, target_dims = validate_y(data, X.shape[0])
    cope, size = _estimates(X, y, contrast_matrix, f_contrast, statistic)

    if n_boot == 0:
        if statistic == "F":
            raise ValueError(
                "A parametric interval (n_boot=0) is only available for t-contrasts."
            )
        _, copes, varcopes, dof, *_ = fit_ols(X, y, contrast_matrix, warn=False)
        t_crit = scipy_stats.t.ppf((1 + ci) / 2, dof)
        half = t_crit * np.sqrt(varcopes[0])
        cope_ci = np.stack([copes[0] - half, copes[0] + half])
        size_ci = None
    else:
        strata = _strata(design, contrast)
        if blocks is not None:
            blocks = np.asarray(blocks)
        boot_cope = np.empty((n_boot, *cope.shape))
        boot_size = np.empty((n_boot, *size.shape))
        bar = progress(total=n_boot, desc="Bootstrap")
        for b in range(n_boot):
            rows = _resample_rows(strata, blocks)
            try:
                boot_cope[b], boot_size[b] = _estimates(
                    X[rows], y[rows], contrast_matrix, f_contrast, statistic
                )
            except ValueError:  # too few complete samples in this resample
                boot_cope[b], boot_size[b] = np.nan, np.nan
            bar.update(1)
        bar.close()
        q = [100 * (1 - ci) / 2, 100 * (1 + ci) / 2]
        cope_ci = np.nanpercentile(boot_cope, q, axis=0)
        size_ci = np.nanpercentile(boot_size, q, axis=0)

    cope = (
        cope.reshape((-1, *target_dims))
        if statistic == "F"
        else cope.reshape(target_dims)
    )
    cope_ci = cope_ci.reshape((2, *cope.shape))
    size = size.reshape(target_dims)
    if size_ci is not None:
        size_ci = size_ci.reshape((2, *target_dims))
    return EffectSize(
        contrast=contrast.name,
        statistic=statistic,
        ci=ci,
        n_boot=n_boot,
        cope=cope,
        cope_ci=cope_ci,
        d=size if statistic == "t" else None,
        d_ci=size_ci if statistic == "t" else None,
        eta2=size if statistic == "F" else None,
        eta2_ci=size_ci if statistic == "F" else None,
    )
