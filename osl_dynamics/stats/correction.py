"""Corrections for multiple comparisons applied to p-values."""

from typing import Optional

import numpy as np

CORRECTIONS = ("bonferroni", "holm", "fdr")


def _prepare(pvalues: np.ndarray):
    p = np.asarray(pvalues, dtype=float)
    flat = p.reshape(-1)
    valid = ~np.isnan(flat)
    return p, flat, valid


def bonferroni(pvalues: np.ndarray) -> np.ndarray:
    """Bonferroni correction (controls the family-wise error rate).

    Parameters
    ----------
    pvalues : np.ndarray
        Uncorrected p-values. Any shape. NaNs are ignored.

    Returns
    -------
    corrected : np.ndarray
        Corrected p-values, :code:`min(p * n_tests, 1)`. Same shape.
    """
    p, flat, valid = _prepare(pvalues)
    corrected = flat.copy()
    corrected[valid] = np.minimum(flat[valid] * valid.sum(), 1.0)
    return corrected.reshape(p.shape)


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction (controls the family-wise error rate).

    Uniformly more powerful than Bonferroni.

    Parameters
    ----------
    pvalues : np.ndarray
        Uncorrected p-values. Any shape. NaNs are ignored.

    Returns
    -------
    corrected : np.ndarray
        Corrected p-values. Same shape.
    """
    p, flat, valid = _prepare(pvalues)
    corrected = flat.copy()
    pv = flat[valid]
    n = len(pv)
    order = np.argsort(pv)
    adjusted = pv[order] * (n - np.arange(n))
    adjusted = np.minimum(np.maximum.accumulate(adjusted), 1.0)
    out = np.empty(n)
    out[order] = adjusted
    corrected[valid] = out
    return corrected.reshape(p.shape)


def fdr(pvalues: np.ndarray, method: str = "bh") -> np.ndarray:
    """False discovery rate correction.

    Parameters
    ----------
    pvalues : np.ndarray
        Uncorrected p-values. Any shape. NaNs are ignored.
    method : str, optional
        :code:`'bh'` for Benjamini-Hochberg (assumes independent or
        positively dependent tests) or :code:`'by'` for Benjamini-Yekutieli
        (valid under any dependence, more conservative).

    Returns
    -------
    corrected : np.ndarray
        FDR-adjusted p-values (q-values). Same shape. Rejecting
        :code:`corrected < alpha` controls the FDR at :code:`alpha`.
    """
    if method not in ("bh", "by"):
        raise ValueError(f"method must be 'bh' or 'by', got {method}.")
    p, flat, valid = _prepare(pvalues)
    corrected = flat.copy()
    pv = flat[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranks = np.arange(1, n + 1)
    adjusted = pv[order] * n / ranks
    if method == "by":
        adjusted *= np.sum(1.0 / ranks)
    adjusted = np.minimum(np.minimum.accumulate(adjusted[::-1])[::-1], 1.0)
    out = np.empty(n)
    out[order] = adjusted
    corrected[valid] = out
    return corrected.reshape(p.shape)


def correct(pvalues: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply a correction by name.

    Parameters
    ----------
    pvalues : np.ndarray
        Uncorrected p-values.
    method : str or None
        :code:`'bonferroni'`, :code:`'holm'`, :code:`'fdr'` or :code:`None`
        (no correction).

    Returns
    -------
    corrected : np.ndarray
        Corrected p-values.
    """
    if method is None:
        return np.asarray(pvalues, dtype=float)
    if method == "bonferroni":
        return bonferroni(pvalues)
    if method == "holm":
        return holm(pvalues)
    if method == "fdr":
        return fdr(pvalues)
    raise ValueError(f"method must be one of {CORRECTIONS} or None, got {method}.")
