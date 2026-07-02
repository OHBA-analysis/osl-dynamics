"""Sign flipping.

Source reconstruction leaves the sign of each channel (parcel) arbitrary, so the
same parcel can have opposite polarity across sessions. Before pooling sessions
we align these signs: for each session we search for the ``+1``/``-1`` per-channel
vector whose (time-delay embedded, standardized) covariance best matches a
template.
"""

from typing import Optional, Tuple

import numpy as np

from osl_dynamics.data import processing


def calc_cov(
    array: np.ndarray, n_embeddings: int, standardize: bool = True
) -> np.ndarray:
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


def calc_corr(
    M1: np.ndarray, M2: np.ndarray, n_embeddings: int, mode: Optional[str] = None
) -> float:
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


def apply_flips_to_cov(
    cov: np.ndarray, flips: np.ndarray, n_embeddings: int
) -> np.ndarray:
    """Apply a ``±1`` per-channel flip vector to an embedded covariance."""
    f = np.repeat(flips, n_embeddings)[np.newaxis, ...]
    return cov * (f.T @ f)


class _FlipObjective:
    """Reduced ``O(n_channels²)`` evaluator of the sign-flip correlation.

    The correlation :func:`calc_corr` compares depends on the ``±1`` per-channel
    flip vector ``f`` only through two quadratic forms,

        P(f) = sum_{p<q} f_p f_q A[p, q]      (agreement with the template)
        Q(f) = sum_{p<q} f_p f_q B[p, q]      (mean of the flipped covariance)

    where ``A`` and ``B`` are ``n_channels × n_channels`` matrices built once from
    the selected entries of the off-diagonal covariance blocks. The correlation is
    then a closed-form function of ``P(f)``, ``Q(f)`` and a few constants, so it
    can be evaluated exactly without rebuilding the full covariance. This
    reproduces ``calc_corr(apply_flips_to_cov(cov, f, E), template_cov, E)`` to
    machine precision.

    Parameters
    ----------
    cov, template_cov : np.ndarray
        Session and template embedded covariances (from :func:`calc_cov`).
    n_channels : int
        Number of channels (parcels).
    n_embeddings : int
        Number of time-delay embeddings used to build the covariances.
    """

    def __init__(
        self,
        cov: np.ndarray,
        template_cov: np.ndarray,
        n_channels: int,
        n_embeddings: int,
    ):
        C = np.asarray(cov)
        T = np.asarray(template_cov)
        E = n_embeddings
        N = n_channels * E

        # Validate shapes so misuse fails with a clear message rather than a
        # confusing error deeper in the search.
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError(f"cov must be a square 2D array, got shape {C.shape}.")
        if C.shape != T.shape:
            raise ValueError(
                f"cov {C.shape} and template_cov {T.shape} must have the same shape."
            )
        if C.shape[0] != N:
            raise ValueError(
                f"cov side length {C.shape[0]} must equal "
                f"n_channels * n_embeddings = {n_channels} * {E} = {N}."
            )

        # The exact entries calc_corr compares (strict upper triangle, k=E). All
        # of these fall in off-diagonal channel blocks; within-channel blocks are
        # skipped and do not depend on the sign.
        rows, cols = np.triu_indices(N, k=E)
        c = C[rows, cols]
        t = T[rows, cols]
        pc = rows // E  # channel of each selected row
        qc = cols // E  # channel of each selected column

        A = np.zeros((n_channels, n_channels))
        B = np.zeros((n_channels, n_channels))
        np.add.at(A, (pc, qc), c * t)
        np.add.at(B, (pc, qc), c)
        # Symmetrize; the diagonal stays zero (within-channel entries excluded).
        self.A = A + A.T
        self.B = B + B.T

        self.n_channels = n_channels
        self.S = float(len(rows))  # number of compared entries, |S|
        self.mT = t.sum() / self.S  # mean of the template over S
        self.sigT = np.sqrt((t * t).sum() / self.S - self.mT**2)  # std of template
        self.K = (c * c).sum() / self.S  # mean of C^2 over S (flip-invariant)

        if self.sigT == 0.0 or self.K == 0.0:
            raise ValueError(
                "Degenerate covariance: zero variance across the compared "
                "off-diagonal entries of cov or template_cov."
            )

    def corr_from_PQ(self, P, Q):
        """Correlation given the two quadratic forms ``P``, ``Q``.

        ``P`` and ``Q`` may each be a scalar or an array (the coordinate-ascent
        sweep evaluates every single-channel flip at once).
        """
        mean_CF = Q / self.S
        # Clip tiny negatives from rounding; the true variance is non-negative.
        std_CF = np.sqrt(np.maximum(self.K - mean_CF**2, 0.0))
        num = P / self.S - mean_CF * self.mT
        denom = std_CF * self.sigT
        # denom is 0 only for a (near-)constant flipped covariance; report 0 there
        # so the search sees no correlation instead of propagating nan/inf.
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = num / denom
        return np.where(denom > 0, corr, 0.0)

    def corr(self, f: np.ndarray) -> np.ndarray:
        """Correlation with the template for flip vector ``f``."""
        f = np.asarray(f, dtype=float)
        return self.corr_from_PQ(0.5 * f @ (self.A @ f), 0.5 * f @ (self.B @ f))


def _coordinate_ascent(obj: _FlipObjective, f: np.ndarray) -> Tuple[np.ndarray, float]:
    """Greedy single-flip coordinate ascent on the exact correlation.

    Repeatedly flip the single channel that most increases the correlation until
    no single flip helps. Fields ``u = A f`` and ``v = B f`` are maintained so
    each sweep costs ``O(n_channels)``.
    """
    f = np.asarray(f, dtype=float).copy()
    u = obj.A @ f
    v = obj.B @ f
    P = 0.5 * f @ u
    Q = 0.5 * f @ v
    cur = obj.corr_from_PQ(P, Q)
    while True:
        # Flipping channel p: P -> P - 2 f_p u_p, Q -> Q - 2 f_p v_p.
        P_new = P - 2.0 * f * u
        Q_new = Q - 2.0 * f * v
        cand = obj.corr_from_PQ(P_new, Q_new)
        p = int(np.argmax(cand))
        if cand[p] <= cur + 1e-12:
            break
        df = -2.0 * f[p]
        u += obj.A[:, p] * df
        v += obj.B[:, p] * df
        f[p] = -f[p]
        P, Q, cur = P_new[p], Q_new[p], cand[p]
    return f, float(cur)


def _stochastic_polish(
    obj: _FlipObjective,
    f: np.ndarray,
    cur: float,
    n_iter: int,
    max_flips: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Random multi-flip hill-climb (the previous search) on the reduced objective.

    Keeps the change whenever a random flip of up to ``max_flips`` channels
    increases the correlation. Cheap in the reduced form, so it acts as a safety
    net that cannot leave the result worse than the previous search would.
    """
    n = obj.n_channels
    f = f.copy()
    for _ in range(n_iter):
        k = int(rng.integers(1, max_flips + 1))
        idx = rng.choice(n, size=k, replace=False)
        g = f.copy()
        g[idx] *= -1
        m = float(obj.corr(g))
        if m > cur:
            f, cur = g, m
    return f, cur


def find_flips(
    cov: np.ndarray,
    template_cov: np.ndarray,
    n_channels: int,
    n_embeddings: int,
    n_init: int = 3,
    n_iter: int = 500,
    max_flips: int = 20,
    seed: Optional[int] = None,
    n_spectral: int = 2,
) -> Tuple[np.ndarray, float]:
    """Find the ``±1`` per-channel flip vector aligning ``cov`` to a template.

    The objective is reduced to a small ``n_channels × n_channels`` problem that
    is exactly the correlation :func:`calc_corr` maximizes. In that form aligning
    the signs is a Z2 (Ising / MAX-CUT) synchronization problem, solved by:

    1. warm starts — the sign of the leading eigenvector(s) of the reduced matrix
       (the spectral relaxation), plus the historical all-``+1`` start and
       ``n_init`` random restarts;
    2. exact greedy coordinate ascent on the true correlation from each start;
    3. a short stochastic polish (the previous random-restart search) from the
       best local optimum — essentially free in the reduced form, and guarantees
       the result is no worse than that search.

    This is the search used by
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
        Number of random restarts (in addition to the spectral and all-``+1``
        starts).
    n_iter : int, optional
        Number of proposals in the stochastic polish.
    max_flips : int, optional
        Maximum number of channels flipped in a single polish proposal.
    seed : int, optional
        Seed for :func:`numpy.random.default_rng` (random restarts and polish).
        The spectral warm start and coordinate ascent are deterministic, so the
        result is largely independent of the seed.
    n_spectral : int, optional
        Number of leading eigenvectors of the reduced matrix used as warm starts.

    Returns
    -------
    flips : np.ndarray
        Per-channel flips in ``{+1, -1}``, shape (n_channels,), dtype int.
    metric : float
        The achieved correlation with ``template_cov``.
    """
    rng = np.random.default_rng(seed)
    max_flips = min(max_flips, n_channels)
    obj = _FlipObjective(cov, template_cov, n_channels, n_embeddings)

    # Candidate starting points: all-+1, spectral warm starts, random restarts.
    starts = [np.ones(n_channels)]
    n_spectral = min(n_spectral, n_channels)
    if n_spectral > 0:
        _, evecs = np.linalg.eigh(obj.A)
        for k in range(1, n_spectral + 1):
            s = np.sign(evecs[:, -k])
            s[s == 0] = 1.0
            starts.append(s)
    for _ in range(n_init):
        starts.append(rng.choice([-1.0, 1.0], size=n_channels).astype(float))

    # Exact coordinate ascent from each start; keep the best local optimum.
    best_flips = np.ones(n_channels)
    best_metric = -np.inf
    for s in starts:
        flips, metric = _coordinate_ascent(obj, s)
        if metric > best_metric:
            best_flips, best_metric = flips, metric

    # Stochastic polish from the best local optimum, then re-settle.
    if n_iter > 0 and max_flips > 0:
        best_flips, best_metric = _stochastic_polish(
            obj, best_flips, best_metric, n_iter, max_flips, rng
        )
        best_flips, best_metric = _coordinate_ascent(obj, best_flips)

    return best_flips.astype(int), float(best_metric)
