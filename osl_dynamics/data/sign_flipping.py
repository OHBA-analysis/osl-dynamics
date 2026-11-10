"""Sign flipping.

Source reconstruction leaves the sign of each channel (parcel) arbitrary, so the
same parcel can have opposite polarity across sessions. Before pooling sessions
we align these signs: for each session we search for the ``+1``/``-1`` per-channel
vector whose (time-delay embedded, standardized) covariance best matches a
template covariance. The template only sets the sign convention: it is usually
built from the dataset being flipped (:meth:`Data.align_channel_signs
<osl_dynamics.data.Data.align_channel_signs>` uses the median session by
default), and only needs to come from another dataset to match that dataset's
signs (see two-stage alignment below).

Building blocks:

- :func:`calc_cov` / :func:`calc_cov_mne_raw`: the embedded covariance of a
  session (an array, or a parcellated fif file / :class:`mne.io.Raw`).
- :func:`find_flips`: the flips that align one covariance to a template.
- :func:`build_template` / :func:`build_template_mne_raw`: a template built from
  many sessions (covariances / parcellated fif files).
- :func:`sign_flip` / :func:`sign_flip_mne_raw`: find and apply the flips to an
  array / a parcellated fif file.

The covariances may also be **stacked** across frequency bands, shape
``(n_bands, N, N)`` (``calc_cov_mne_raw(..., frequency_bands=...)``). The entries
of every band are then pooled into one correlation and a single flip vector is
shared across the bands. This helps when the broadband covariance carries
little sign information (e.g. EEG).

**Two-stage alignment.** To align a dataset to a template from *another*
dataset (e.g. the template of a pre-trained model), build a template from the
new dataset, find the flips ``g`` that align it to the external template, then
align each session to the dataset template and multiply its flips by ``g``
(``template_flips=g`` in :func:`sign_flip` and :func:`sign_flip_mne_raw`).
Individual sessions only have to match their own dataset, and the mismatch
between the datasets is absorbed by a single flip search on an averaged
covariance. :meth:`Data.align_channel_signs
<osl_dynamics.data.Data.align_channel_signs>` does this with ``two_stage=True``.
"""

from typing import Optional, Sequence, Tuple, Union

import mne
import numpy as np

from osl_dynamics.data import processing
from osl_dynamics.utils.progress import parallel_map


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


def calc_cov_mne_raw(
    raw,
    n_embeddings: int = 15,
    standardize: bool = True,
    frequency_bands: Optional[Sequence[Tuple[float, float]]] = None,
    picks: str = "misc",
) -> np.ndarray:
    """Covariance of a parcellated fif file, optionally per frequency band.

    Each band is band-pass filtered on the continuous data before bad segments
    are dropped, so the filter does not ring at the gaps.

    Parameters
    ----------
    raw : str or mne.io.Raw
        Path to a parcellated fif file, or a loaded :class:`mne.io.Raw` object.
    n_embeddings : int, optional
        Number of time-delay embeddings.
    standardize : bool, optional
        Standardize the embedded data before computing the covariance.
    frequency_bands : list of tuple, optional
        ``(low, high)`` band edges in Hz, e.g.
        ``[(1, 4), (4, 8), (8, 13), (13, 30)]``. If ``None``, the broadband
        covariance is returned.
    picks : str, optional
        Channel type holding the parcel time courses.

    Returns
    -------
    cov : np.ndarray
        Shape ``(N, N)`` if ``frequency_bands`` is ``None``, otherwise
        ``(n_bands, N, N)``, where ``N = n_channels * n_embeddings``.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raw = mne.io.read_raw_fif(str(raw), preload=True, verbose=False)

    def _cov(r):
        data = r.get_data(picks=picks, reject_by_annotation="omit", verbose=False)
        return calc_cov(data.T, n_embeddings, standardize)

    if frequency_bands is None:
        return _cov(raw)
    return np.stack(
        [
            _cov(raw.copy().load_data().filter(low, high, picks=picks, verbose=False))
            for low, high in frequency_bands
        ]
    )


def _load_cov(cov: Union[str, np.ndarray]) -> np.ndarray:
    """A covariance, or the path to a ``.npy`` file containing one."""
    return cov if isinstance(cov, np.ndarray) else np.load(str(cov))


def _triu_entries(M: np.ndarray, n_embeddings: int) -> np.ndarray:
    """Entries compared by :func:`calc_corr`, pooled over any leading axes."""
    m, n = np.triu_indices(M.shape[-1], k=n_embeddings)
    return M[..., m, n].ravel()


def _abs_similarity(covs: Sequence[np.ndarray], n_embeddings: int) -> np.ndarray:
    """Sign-invariant similarity of every pair of sessions.

    ``calc_corr(covs[i], covs[j], n_embeddings, mode="abs")`` for all pairs, as one
    matrix product of the standardized absolute entries. The diagonal is zero.
    """
    v = np.array([np.abs(_triu_entries(c, n_embeddings)) for c in covs], np.float32)
    v -= v.mean(axis=1, keepdims=True)
    v /= v.std(axis=1, keepdims=True)
    sim = (v @ v.T) / v.shape[1]
    np.fill_diagonal(sim, 0)
    return sim


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
        Covariance matrices of the same shape, ``(N, N)`` or stacked across
        frequency bands ``(n_bands, N, N)`` (the entries of all bands are
        pooled).
    n_embeddings : int
        Number of time-delay embeddings (sets the skipped diagonal offset).
    mode : str, optional
        If ``"abs"``, compare the absolute values (sign-invariant), used when
        picking a template session.

    Returns
    -------
    corr : float
        Pearson correlation between the selected entries.
    """
    if mode == "abs":
        M1 = np.abs(M1)
        M2 = np.abs(M2)
    return np.corrcoef(
        [_triu_entries(M1, n_embeddings), _triu_entries(M2, n_embeddings)]
    )[0, 1]


def apply_flips_to_cov(
    cov: np.ndarray, flips: np.ndarray, n_embeddings: int
) -> np.ndarray:
    """Apply a ``±1`` per-channel flip vector to an embedded covariance.

    ``cov`` may be ``(N, N)`` or stacked across frequency bands
    ``(n_bands, N, N)``.
    """
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
        Session and template embedded covariances (from :func:`calc_cov`),
        ``(N, N)`` or stacked across frequency bands ``(n_bands, N, N)``.
    n_embeddings : int
        Number of time-delay embeddings used to build the covariances.
    """

    def __init__(
        self,
        cov: np.ndarray,
        template_cov: np.ndarray,
        n_embeddings: int,
    ):
        C = np.asarray(cov)
        T = np.asarray(template_cov)
        E = n_embeddings
        N = C.shape[-1]
        n_channels = N // E

        # Validate shapes so misuse fails with a clear message rather than a
        # confusing error deeper in the search.
        if C.ndim not in (2, 3) or C.shape[-1] != C.shape[-2]:
            raise ValueError(
                "cov must be a square (N, N) or stacked (n_bands, N, N) array, "
                f"got shape {C.shape}."
            )
        if C.shape != T.shape:
            raise ValueError(
                f"cov {C.shape} and template_cov {T.shape} must have the same shape."
            )
        if N % E != 0:
            raise ValueError(
                f"cov side length {N} must be a multiple of n_embeddings = {E}."
            )

        # The exact entries calc_corr compares (strict upper triangle, k=E,
        # pooled over the bands of a stacked covariance). All of these fall in
        # off-diagonal channel blocks; within-channel blocks are skipped and do
        # not depend on the sign.
        rows, cols = np.triu_indices(N, k=E)
        n_bands = 1 if C.ndim == 2 else C.shape[0]
        c = _triu_entries(C, E).astype(np.float64)
        t = _triu_entries(T, E).astype(np.float64)
        # Flat (channel of row, channel of column) index of each entry
        pq = np.tile((rows // E) * n_channels + cols // E, n_bands)

        size = n_channels * n_channels
        A = np.bincount(pq, c * t, size).reshape(n_channels, n_channels)
        B = np.bincount(pq, c, size).reshape(n_channels, n_channels)
        # Symmetrize; the diagonal stays zero (within-channel entries excluded).
        self.A = A + A.T
        self.B = B + B.T

        self.n_channels = n_channels
        self.S = float(len(c))  # number of compared entries, |S|
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


def find_flips(
    cov: np.ndarray,
    template_cov: np.ndarray,
    n_embeddings: int,
    n_random_starts: int = 3,
    n_spectral: int = 2,
) -> Tuple[np.ndarray, float]:
    """Find the ``±1`` per-channel flip vector aligning ``cov`` to a template.

    The objective is reduced to a small ``n_channels × n_channels`` problem that
    is exactly the correlation :func:`calc_corr` maximizes. In that form aligning
    the signs is a Z2 (Ising / MAX-CUT) synchronization problem, solved by:

    1. warm starts: the sign of the leading eigenvector(s) of the reduced matrix
       (the spectral relaxation), plus the all-``+1`` start and
       ``n_random_starts`` random restarts;
    2. exact greedy coordinate ascent on the true correlation from each start,
       keeping the best local optimum.

    Parameters
    ----------
    cov : np.ndarray
        This session's embedded covariance (from :func:`calc_cov`), ``(N, N)``
        or stacked across frequency bands ``(n_bands, N, N)`` (from
        :func:`calc_cov_mne_raw`).
    template_cov : np.ndarray
        Template embedded covariance to align to (same shape as ``cov``).
    n_embeddings : int
        Number of time-delay embeddings used to build ``cov``.
    n_random_starts : int, optional
        Number of random restarts (in addition to the spectral and all-``+1``
        starts). The restarts use a fixed internal seed, so the whole search is
        deterministic and reproducible.
    n_spectral : int, optional
        Number of leading eigenvectors of the reduced matrix used as warm starts.

    Returns
    -------
    flips : np.ndarray
        Per-channel flips in ``{+1, -1}``, shape (n_channels,), dtype int.
    metric : float
        The achieved correlation with ``template_cov``.
    """
    # Fixed seed: the restarts only need to be diverse, not run-to-run random,
    # so the search stays fully deterministic and reproducible.
    rng = np.random.default_rng(0)
    obj = _FlipObjective(cov, template_cov, n_embeddings)
    n_channels = obj.n_channels

    # Candidate starting points: all-+1, spectral warm starts, random restarts.
    starts = [np.ones(n_channels)]
    n_spectral = min(n_spectral, n_channels)
    if n_spectral > 0:
        _, evecs = np.linalg.eigh(obj.A)
        for k in range(1, n_spectral + 1):
            s = np.sign(evecs[:, -k])
            s[s == 0] = 1.0
            starts.append(s)
    for _ in range(n_random_starts):
        starts.append(rng.choice([-1.0, 1.0], size=n_channels).astype(float))

    # Exact coordinate ascent from each start; keep the best local optimum.
    best_flips = np.ones(n_channels)
    best_metric = -np.inf
    for s in starts:
        flips, metric = _coordinate_ascent(obj, s)
        if metric > best_metric:
            best_flips, best_metric = flips, metric

    return best_flips.astype(int), float(best_metric)


def build_template(
    covs: Sequence[np.ndarray],
    n_embeddings: int,
    n_iter: int = 2,
    n_random_starts: int = 3,
) -> Tuple[np.ndarray, int]:
    """Build a template covariance from many sessions.

    1. Medoid: the session whose covariance is most similar to all others,
       using the sign-invariant (``mode="abs"``) correlation of
       :func:`calc_corr`.
    2. ``n_iter`` rounds of: align every session to the current template with
       :func:`find_flips`, then set the template to the mean of the aligned
       covariances.

    The template keeps the (arbitrary) sign convention of the medoid session.
    To align it to another template, find the flips with :func:`find_flips` and
    pass them as ``template_flips`` when flipping each session (see the module
    docstring).

    Parameters
    ----------
    covs : list of np.ndarray
        Embedded covariance of each session, each ``(N, N)`` or stacked across
        frequency bands ``(n_bands, N, N)``.
    n_embeddings : int
        Number of time-delay embeddings used to build the covariances.
    n_iter : int, optional
        Number of align-to-mean rounds after the medoid.
    n_random_starts : int, optional
        Random restarts of each :func:`find_flips` search.

    Returns
    -------
    template_cov : np.ndarray
        Template covariance, same shape as each element of ``covs``.
    medoid : int
        Index of the medoid session.
    """
    medoid = int(np.argmax(_abs_similarity(covs, n_embeddings).sum(axis=1)))

    template = np.asarray(covs[medoid], dtype=np.float64)
    for _ in range(n_iter):
        aligned = np.zeros_like(template)
        for c in covs:
            flips, _ = find_flips(
                c, template, n_embeddings, n_random_starts=n_random_starts
            )
            aligned += apply_flips_to_cov(c, flips, n_embeddings)
        template = aligned / len(covs)
    return template, medoid


def build_template_mne_raw(
    fifs: Sequence,
    n_embeddings: int = 15,
    standardize: bool = True,
    frequency_bands: Optional[Sequence[Tuple[float, float]]] = None,
    picks: str = "misc",
    n_iter: int = 2,
    n_random_starts: int = 3,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, int]:
    """Build a template covariance from parcellated fif files.

    Computes the covariance of each file with :func:`calc_cov_mne_raw` (in
    parallel, stored as float32 to save memory), then builds the template with
    :func:`build_template`.

    Parameters
    ----------
    fifs : list of str or mne.io.Raw
        Parcellated fif files (or :class:`mne.io.Raw` objects), one per session.
    n_embeddings : int, optional
        Number of time-delay embeddings.
    standardize : bool, optional
        Standardize the embedded data before computing the covariance.
    frequency_bands : list of tuple, optional
        ``(low, high)`` band edges in Hz. If passed, the template is stacked
        across bands, shape ``(n_bands, N, N)`` (see :func:`calc_cov_mne_raw`).
    picks : str, optional
        Channel type holding the parcel time courses.
    n_iter : int, optional
        Number of align-to-mean rounds after the medoid.
    n_random_starts : int, optional
        Random restarts of each :func:`find_flips` search.
    n_jobs : int, optional
        Number of files to compute covariances for in parallel.

    Returns
    -------
    template_cov : np.ndarray
        Template covariance.
    medoid : int
        Index of the medoid session in ``fifs``.
    """

    def _cov(fif):
        cov = calc_cov_mne_raw(fif, n_embeddings, standardize, frequency_bands, picks)
        return cov.astype(np.float32)

    covs = parallel_map(
        _cov, fifs, desc="Calculating covariances", n_jobs=n_jobs, label="session"
    )
    return build_template(covs, n_embeddings, n_iter, n_random_starts)


def sign_flip(
    parcel_data: np.ndarray,
    template_cov: Union[str, np.ndarray],
    n_embeddings: int = 15,
    standardize: bool = True,
    template_flips: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sign flip parcel time courses to match a template covariance.

    Finds the per-parcel ``+1``/``-1`` signs whose (time-delay embedded,
    standardized) covariance best matches ``template_cov`` and applies them.

    Parameters
    ----------
    parcel_data : np.ndarray
        Parcel time courses, shape ``(n_parcels, n_samples)``.
    template_cov : str or np.ndarray
        Template covariance to align to, or the path to a ``.npy`` file
        containing it. This must be the covariance of the time-delay embedded
        data (see :func:`calc_cov`), built with the same ``n_embeddings`` and
        ``standardize`` settings passed here.
    n_embeddings : int, optional
        Number of time-delay embeddings used to build the covariances.
    standardize : bool, optional
        Standardize the embedded data before computing the covariance.
    template_flips : np.ndarray, optional
        Flips that align ``template_cov`` to another template, shape
        ``(n_parcels,)``. If passed, they are multiplied into the flips found
        for this session (two-stage alignment, see the module docstring).

    Returns
    -------
    flipped_data : np.ndarray
        ``parcel_data`` with the per-parcel signs applied, same shape as the
        input.
    flips : np.ndarray
        The per-parcel ``+1``/``-1`` signs that were applied, shape
        ``(n_parcels,)``.
    corr : float
        Correlation with ``template_cov`` after the flips found for this
        session (before ``template_flips``), the metric maximized by the flip
        search.
    """
    cov = calc_cov(parcel_data.T, n_embeddings, standardize)
    flips, corr = _find_session_flips(cov, template_cov, n_embeddings, template_flips)
    return parcel_data * flips[:, np.newaxis], flips, corr


def sign_flip_mne_raw(
    fif,
    template_cov: Union[str, np.ndarray],
    output_file: Optional[str] = None,
    n_embeddings: int = 15,
    standardize: bool = True,
    picks: str = "misc",
    frequency_bands: Optional[Sequence[Tuple[float, float]]] = None,
    template_flips: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, float], "mne.io.Raw"]:
    """Sign flip a parcellated fif file to match a template covariance.

    Loads parcel time courses from a fif file (or :class:`mne.io.Raw`), finds
    the per-parcel signs that best align its (time-delay embedded) covariance
    to ``template_cov``, and applies them to the ``picks`` channels. Every
    other channel, the annotations and the data in bad segments are kept
    (bad segments are only left out of the covariance). The result is either
    saved to a new fif file or returned as an :class:`mne.io.Raw` object (see
    ``output_file``).

    Parameters
    ----------
    fif : str or mne.io.Raw
        Path to a parcellated fif file, or an :class:`mne.io.Raw` object (which
        is not modified). The parcel time courses are read from the ``picks``
        channels.
    template_cov : str or np.ndarray
        Template covariance to align to, or the path to a ``.npy`` file
        containing it. This must be the covariance of the time-delay embedded
        data (see :func:`calc_cov`), built with the same ``n_embeddings``,
        ``standardize`` and ``frequency_bands`` settings passed here.
    output_file : str, optional
        Path to save the sign-flipped fif file to. If ``None`` (the default),
        nothing is written to disk and the sign-flipped :class:`mne.io.Raw`
        object is returned instead (see Returns).
    n_embeddings : int, optional
        Number of time-delay embeddings used to build the covariances.
    standardize : bool, optional
        Standardize the embedded data before computing the covariance.
    picks : str, optional
        Channel type holding the parcel time courses. Parcellated fif files
        written by osl-dynamics store parcels as ``"misc"`` channels.
    frequency_bands : list of tuple, optional
        ``(low, high)`` band edges in Hz. If passed, the flips are found on the
        covariances of the band-pass filtered data stacked across bands (see
        :func:`calc_cov_mne_raw`), and ``template_cov`` must be stacked in the
        same way, shape ``(n_bands, N, N)``. The saved data is not filtered.
    template_flips : np.ndarray, optional
        Flips that align ``template_cov`` to another template, shape
        ``(n_parcels,)``. If passed, they are multiplied into the flips found
        for this session (two-stage alignment, see the module docstring).

    Returns
    -------
    flips, corr : np.ndarray, float
        If ``output_file`` is passed: the per-parcel ``+1``/``-1`` signs that
        were applied and the correlation with ``template_cov`` (see
        :func:`sign_flip`).
    raw : mne.io.Raw
        If ``output_file`` is ``None``: the sign-flipped data.

    Examples
    --------
    Save the sign-flipped data to a new fif file::

        from osl_dynamics.data import sign_flipping

        sign_flipping.sign_flip_mne_raw(
            "lcmv-parc-raw.fif",
            "template_cov.npy",
            "sflip-lcmv-parc-raw.fif",
        )

    Or get the sign-flipped data back as an :class:`mne.io.Raw` without
    saving::

        raw = sign_flipping.sign_flip_mne_raw("lcmv-parc-raw.fif", "template_cov.npy")
    """
    if isinstance(fif, mne.io.BaseRaw):
        raw = fif.copy().load_data()
    else:
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)

    cov = calc_cov_mne_raw(raw, n_embeddings, standardize, frequency_bands, picks)
    flips, corr = _find_session_flips(cov, template_cov, n_embeddings, template_flips)
    raw.apply_function(
        lambda x: x * flips[:, np.newaxis], picks=picks, channel_wise=False
    )

    if output_file is None:
        return raw
    raw.save(str(output_file), overwrite=True)
    return flips, corr


def _find_session_flips(cov, template_cov, n_embeddings, template_flips=None):
    """Shared core of :func:`sign_flip` and :func:`sign_flip_mne_raw`."""
    flips, corr = find_flips(cov, _load_cov(template_cov), n_embeddings)
    if template_flips is not None:
        flips = flips * np.asarray(template_flips, dtype=int)
    print(
        f"Sign flipping: flipped {(flips == -1).sum()}/{len(flips)} parcels "
        f"(template corr {corr:.3f})"
    )
    return flips, corr
