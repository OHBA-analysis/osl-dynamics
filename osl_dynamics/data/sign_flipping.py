"""Sign flipping of source-reconstructed data.

Source reconstruction leaves the sign of each channel (parcel) arbitrary, so the
same parcel can have opposite polarity across sessions. Before pooling sessions
we align these signs: for each session we search for the ``+1``/``-1`` per-channel
vector whose (time-delay embedded, standardized) covariance best matches a
template.

This module holds the reusable pieces of that search:

- :func:`calc_cov` — covariance of time-delay embedded (optionally standardized)
  data, the representation the signs are aligned in.
- :func:`calc_corr` — upper-triangle correlation between two such covariances,
  the objective the search maximises.
- :func:`find_flips` — the ``±1`` per-channel flip vector (and the achieved
  correlation) that best aligns a covariance to a template.

:meth:`osl_dynamics.data.base.Data.align_channel_signs` uses these to *apply* the
flips across a set of sessions. Callers that only need the flip vector (e.g. to
apply it elsewhere, or to precompute a template) can use :func:`find_flips`
directly.
"""

import numpy as np

from osl_dynamics.data import processing


def calc_cov(array, n_embeddings, standardize=True):
    """Covariance of time-delay embedded (optionally standardized) data.

    Parameters
    ----------
    array : np.ndarray
        Data, shape (n_samples, n_channels).
    n_embeddings : int
        Number of time-delay embeddings.
    standardize : bool, optional
        Standardize the embedded data before computing the covariance.

    Returns
    -------
    cov : np.ndarray
        Covariance, shape (n_channels * n_embeddings, n_channels * n_embeddings).
    """
    array = processing.time_embed(array, n_embeddings)
    if standardize:
        array = processing.standardize(array, create_copy=False)
    return np.cov(array.T)


def calc_corr(M1, M2, n_embeddings, mode=None):
    """Correlation between the upper triangles of two covariances.

    The first ``n_embeddings`` diagonals are skipped (``k=n_embeddings`` in
    :func:`numpy.triu_indices`) so within-channel, near-lag terms — which do not
    depend on the sign — are excluded from the comparison.

    Parameters
    ----------
    M1, M2 : np.ndarray
        Covariance matrices of the same shape.
    n_embeddings : int
        Number of time-delay embeddings (sets the skipped diagonal offset).
    mode : str, optional
        If ``"abs"``, compare the absolute values (sign-invariant), used when
        picking a median template session.

    Returns
    -------
    corr : float
        Pearson correlation between the selected entries.
    """
    if mode == "abs":
        M1 = np.abs(M1)
        M2 = np.abs(M2)
    m, n = np.triu_indices(M1.shape[0], k=n_embeddings)
    return np.corrcoef([M1[m, n], M2[m, n]])[0, 1]


def apply_flips_to_cov(cov, flips, n_embeddings):
    """Apply a ``±1`` per-channel flip vector to an embedded covariance."""
    f = np.repeat(flips, n_embeddings)[np.newaxis, ...]
    return cov * (f.T @ f)


def find_flips(
    cov,
    template_cov,
    n_channels,
    n_embeddings,
    n_init=3,
    n_iter=2500,
    max_flips=20,
    seed=None,
):
    """Find the ``±1`` per-channel flip vector aligning ``cov`` to a template.

    Greedy random search: from an all-``+1`` start, repeatedly flip a random
    subset of channels and keep the change whenever it increases the correlation
    with ``template_cov`` (:func:`calc_corr`). The best of ``n_init`` restarts is
    returned. This is the search used by
    :meth:`osl_dynamics.data.base.Data.align_channel_signs`; here it returns the
    vector instead of applying it.

    Parameters
    ----------
    cov : np.ndarray
        This session's embedded covariance (from :func:`calc_cov`).
    template_cov : np.ndarray
        Template embedded covariance to align to (same shape as ``cov``).
    n_channels : int
        Number of channels (parcels).
    n_embeddings : int
        Number of time-delay embeddings used to build ``cov``.
    n_init : int, optional
        Number of random restarts.
    n_iter : int, optional
        Number of proposals per restart.
    max_flips : int, optional
        Maximum number of channels flipped in a single proposal.
    seed : int, optional
        If given, a :func:`numpy.random.default_rng` seed for a reproducible
        search. If ``None`` the global :mod:`numpy.random` state is used
        (matching the historical behaviour of ``align_channel_signs``).

    Returns
    -------
    flips : np.ndarray
        Per-channel flips in ``{+1, -1}``, shape (n_channels,), dtype int.
    metric : float
        The achieved correlation with ``template_cov``.
    """
    rng = np.random if seed is None else np.random.default_rng(seed)
    max_flips = min(max_flips, n_channels)

    def _randomly_flip(flips):
        n_to_flip = rng.choice(max_flips, size=1)
        idx = rng.choice(n_channels, size=n_to_flip, replace=False)
        new_flips = np.copy(flips)
        new_flips[idx] *= -1
        return new_flips

    best_flips = np.ones(n_channels)
    best_metric = 0.0
    for _ in range(n_init):
        flips = np.ones(n_channels)
        metric = calc_corr(cov, template_cov, n_embeddings)
        for _ in range(n_iter):
            new_flips = _randomly_flip(flips)
            new_cov = apply_flips_to_cov(cov, new_flips, n_embeddings)
            new_metric = calc_corr(new_cov, template_cov, n_embeddings)
            if new_metric > metric:
                flips = new_flips
                metric = new_metric
        if metric > best_metric:
            best_metric = metric
            best_flips = flips
    return best_flips.astype(int), float(best_metric)
