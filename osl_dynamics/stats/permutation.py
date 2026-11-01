"""Permutation tests for GLM contrasts.

Null distributions are built with the Freedman-Lane scheme [1]_: the data are
regressed on the nuisance regressors (the features with a zero weight in the
contrast), the residuals are sign-flipped or row-shuffled, the nuisance fit
is added back and the full model is refitted. Without nuisance regressors
this is the same as permuting the data.

References
----------
.. [1] Winkler, A. M. et al. (2014). Permutation inference for the general
       linear model. NeuroImage, 92, 381-397.
"""

import itertools
import logging
import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from scipy import stats as scipy_stats

from osl_dynamics.stats.cluster import (
    Adjacency,
    ClusterStructure,
    cluster_stat_map,
    max_cluster_stat,
    max_tfce,
    tfce,
)
from osl_dynamics.stats.design import Design
from osl_dynamics.stats.glm import GLM, complete_case_groups, hat_adjustment, validate_y
from osl_dynamics.stats.results import TestResult
from osl_dynamics.utils.progress import progress

_logger = logging.getLogger("osl-dynamics")

PERM_TYPES = ("sign_flip", "row_shuffle")

# Largest number of floats to hold in memory at once when calculating the
# null statistics in batches, and the largest null distribution to keep
_MAX_BATCH_ELEMENTS = 2**24
_MAX_NULL_ELEMENTS = 2**24


class Permutation:
    """Permutation test of a GLM contrast with uncorrected p-values.

    Each target gets its own null distribution, so the p-values are not
    corrected for multiple comparisons. Use :class:`MaxStatPermutation` for
    family-wise error control, or apply :func:`osl_dynamics.stats.fdr`
    to the p-values.

    Parameters
    ----------
    design : Design
        The design (regressors and contrasts).
    contrast_indx : int, optional
        Index of the contrast to test.
    n_perm : int, optional
        Number of permutations. If every possible permutation can be
        enumerated within this number the test is exact.
    perm_type : str, optional
        :code:`'sign_flip'` (within-subject designs, tests of a mean) or
        :code:`'row_shuffle'` (between-subject designs, group differences
        and continuous regressors). Chosen from the feature and contrast
        types if :code:`None`.
    tail : int, optional
        :code:`0` for a two-tailed test, :code:`1` to test for a positive
        effect, :code:`-1` for a negative effect.
    metric : str, optional
        Statistic to build the null distribution from: :code:`'tstats'`
        or :code:`'copes'`.
    hat : float, optional
        'Hat' variance adjustment, see
        :func:`~osl_dynamics.stats.glm.hat_adjustment`. Default is :code:`0`
        (none).
    n_jobs : int, optional
        Number of parallel jobs. Only used by cluster-based corrections.
    """

    correction: Optional[str] = None

    # Largest number of permutations processed at once (None: memory bound)
    _max_batch_size: Optional[int] = None

    def __init__(
        self,
        design: Design,
        contrast_indx: int = 0,
        n_perm: int = 1000,
        perm_type: Optional[str] = None,
        tail: int = 0,
        metric: str = "tstats",
        hat: float = 0.0,
        n_jobs: int = 1,
    ) -> None:
        self.design = design
        self.glm = GLM(design)

        if not 0 <= contrast_indx < design.n_contrasts:
            raise ValueError(
                f"contrast_indx must be between 0 and {design.n_contrasts - 1}."
            )
        self.contrast_indx = contrast_indx
        self.contrast_name = design.contrast_names[contrast_indx]
        self.contrast = self.glm.contrasts[contrast_indx]

        if n_perm < 1:
            raise ValueError("n_perm must be at least 1.")
        self.n_perm = int(n_perm)

        if tail not in (-1, 0, 1):
            raise ValueError(f"tail must be -1, 0 or 1, got {tail}.")
        self.tail = tail

        if metric not in ("tstats", "copes"):
            raise ValueError(f"metric must be 'tstats' or 'copes', got {metric}.")
        self.metric = metric

        if hat < 0:
            raise ValueError(f"hat must be non-negative, got {hat}.")
        self.hat = hat

        self.n_jobs = n_jobs
        self.perm_type = self._resolve_perm_type(perm_type)
        self.pooled_dims: Tuple[int, ...] = ()

    # ------------------------------------------------------------------
    # Permutation scheme

    def _resolve_perm_type(self, perm_type: Optional[str]) -> str:
        if perm_type is not None:
            if perm_type not in PERM_TYPES:
                raise ValueError(f"perm_type must be one of {PERM_TYPES}.")
            return perm_type

        indices = self.design.contrast_feature_indices(self.contrast_indx)
        feature_types = set(np.array(self.design.feature_types)[indices])
        if len(feature_types) > 1:
            raise ValueError(
                "perm_type cannot be chosen automatically when the contrast "
                f"mixes feature types {sorted(feature_types)}. Pass perm_type."
            )
        feature_type = feature_types.pop()
        contrast_type = self.design.contrast_types[self.contrast_indx]

        if feature_type == "constant":
            return "sign_flip"
        if feature_type == "categorical":
            return "row_shuffle" if contrast_type == "differential" else "sign_flip"
        return "row_shuffle"

    def _rows_to_shuffle(self) -> np.ndarray:
        """Rows that are shuffled: only those in the groups being compared
        when the contrast is between categorical features."""
        indices = self.design.contrast_feature_indices(self.contrast_indx)
        feature_types = np.array(self.design.feature_types)[indices]
        if np.all(feature_types == "categorical"):
            return np.where(np.any(self.glm.X[:, indices] != 0, axis=1))[0]
        return np.arange(self.glm.n_samples)

    def _permutations(
        self, allow_exact_shuffle: bool = True
    ) -> Tuple[np.ndarray, bool]:
        """Draw the permutations.

        Parameters
        ----------
        allow_exact_shuffle : bool, optional
            Whether row shuffles may be enumerated exhaustively. This must be
            :code:`False` when any target has missing values: the
            permutations induced on the complete rows of such a target do not
            visit every assignment equally often, so the test would not be
            exact. Sign flips are unaffected because every sign vector of the
            complete rows is induced equally often.

        Returns
        -------
        perms : np.ndarray
            Sign vectors (sign flip) or row indices (row shuffle) for each
            permutation. Shape is (n_perm, n_samples).
        exact : bool
            Whether every possible permutation was enumerated.
        """
        n = self.glm.n_samples

        if self.perm_type == "sign_flip":
            if 2**n <= self.n_perm:
                perms = np.array(list(itertools.product([-1, 1], repeat=n)))
                return perms, True
            return np.random.choice([-1, 1], size=(self.n_perm, n)), False

        rows = self._rows_to_shuffle()
        n_rows = len(rows)
        identity = np.arange(n)

        # Exhaustive enumeration of two-group assignments. This is only
        # exact when there are no continuous regressors, because under
        # Freedman-Lane a shuffled residual inherits the covariate values of
        # the row it moves to.
        indices = self.design.contrast_feature_indices(self.contrast_indx)
        no_continuous = "continuous" not in self.design.feature_types
        if allow_exact_shuffle and len(indices) == 2 and no_continuous:
            in_first = self.glm.X[rows, indices[0]] != 0
            n_first = int(in_first.sum())
            n_total = math.comb(n_rows, n_first)
            if n_total <= self.n_perm:
                perms = np.tile(identity, (n_total, 1))
                first_rows = rows[in_first]
                second_rows = rows[~in_first]
                for i, chosen in enumerate(itertools.combinations(rows, n_first)):
                    chosen = np.array(chosen)
                    rest = np.setdiff1d(rows, chosen)
                    perms[i, first_rows] = chosen
                    perms[i, second_rows] = rest
                return perms, True

        if allow_exact_shuffle and math.factorial(n_rows) <= self.n_perm:
            perms = np.tile(identity, (math.factorial(n_rows), 1))
            for i, p in enumerate(itertools.permutations(rows)):
                perms[i, rows] = p
            return perms, True

        perms = np.tile(identity, (self.n_perm, 1))
        for i in range(self.n_perm):
            perms[i, rows] = np.random.permutation(rows)
        return perms, False

    # ------------------------------------------------------------------
    # Null distribution

    def _apply_tail(self, stat: np.ndarray) -> np.ndarray:
        if self.tail == 0:
            return np.abs(stat)
        if self.tail == -1:
            return -stat
        return stat

    def _transform_observed(self, stat: np.ndarray) -> np.ndarray:
        """Transform the observed statistic map into what is compared with
        the null (here: apply the tail).

        Parameters
        ----------
        stat : np.ndarray
            Signed statistic map. Shape is (\\*target_dims).

        Returns
        -------
        observed : np.ndarray
            Shape is (\\*target_dims).
        """
        return self._apply_tail(stat)

    def _reduce(self, stat: np.ndarray) -> np.ndarray:
        """Reduce a batch of signed null statistic maps to the null statistic.

        Parameters
        ----------
        stat : np.ndarray
            Signed statistic maps. Shape is (batch_size, \\*target_dims).

        Returns
        -------
        null : np.ndarray
            Shape is (batch_size, \\*nonpooled_dims).
        """
        stat = self._apply_tail(stat)
        if len(self.pooled_dims) == 0:
            return stat
        axes = tuple(d + 1 for d in self.pooled_dims)
        return np.nanmax(stat, axis=axes)

    def _before_permutations(self, glm_result) -> None:
        """Hook called after the GLM is fitted and before the permutations."""

    def _close(self) -> None:
        """Hook called when :meth:`fit` finishes."""

    def _expand(self, null: np.ndarray) -> np.ndarray:
        """Insert the pooled dimensions back into a reduced null so it
        broadcasts against the observed statistic maps."""
        axes = tuple(d + 1 for d in self.pooled_dims)
        return np.expand_dims(null, axis=axes) if axes else null

    def _resolve_pooled_dims(self, target_dims: Tuple[int, ...]) -> None:
        self.pooled_dims = ()

    def _prepare_group(self, X: np.ndarray, y: np.ndarray, rows: np.ndarray) -> dict:
        """Quantities needed to compute the null statistics of one
        complete-case group of targets."""
        rows = np.where(rows)[0]
        X_r = X[rows]
        y = y[rows]
        pinv = np.linalg.pinv(X_r)
        dof = X_r.shape[0] - np.linalg.matrix_rank(X_r)

        # Residuals of the data after fitting the nuisance regressors
        nuisance = self.design.nuisance_feature_indices(self.contrast_indx)
        if len(nuisance) > 0:
            Z = X_r[:, nuisance]
            r = y - Z @ (np.linalg.pinv(Z) @ y)
        else:
            r = y

        # Orthonormal basis of the column space of the full design (valid for
        # rank-deficient designs too), for the residual sum of squares
        U, singular_values, _ = np.linalg.svd(X_r, full_matrices=False)
        tol = singular_values.max() * max(X_r.shape) * np.finfo(float).eps
        Q = U[:, singular_values > tol]

        c_pinv = self.contrast @ pinv  # cope = c_pinv @ y*
        return {
            "rows": rows,
            "dof": dof,
            "w": c_pinv,
            "c_xtx_c": float(c_pinv @ c_pinv),
            "Q": Q,
            "r": r,
            "r2": np.sum(r**2, axis=0),
            "shuffle_local": np.where(np.isin(rows, self._rows_to_shuffle()))[0],
        }

    def _null_batch(self, group: dict, perms: np.ndarray) -> np.ndarray:
        """Null statistics of one group of targets for a batch of permutations.

        Returns
        -------
        stat : np.ndarray
            Shape is (batch_size, n_targets_in_group).
        """
        w, Q, r, r2 = group["w"], group["Q"], group["r"], group["r2"]
        n_r = len(w)
        batch_size = perms.shape[0]

        if self.perm_type == "sign_flip":
            S = perms[:, group["rows"]]  # (batch, n_r)
            W = S * w[None, :]
            QP = S[:, :, None] * Q[None, :, :]
        else:
            # Induce a permutation of the complete rows from the global one:
            # the relative order of the global permutation restricted to the
            # complete rows that are shuffled
            local = group["shuffle_local"]
            I = np.tile(np.arange(n_r), (batch_size, 1))
            if len(local) > 1:
                values = perms[:, group["rows"][local]]
                ranks = np.argsort(np.argsort(values, axis=1), axis=1)
                I[:, local] = local[ranks]
            # y*_i = r_{I[i]}, so cope = sum_i w_i r_{I[i]} = W @ r with
            # W[b, I[b, i]] = w[i]; similarly for the projection onto Q
            b = np.arange(batch_size)[:, None]
            W = np.zeros((batch_size, n_r))
            W[b, I] = w[None, :]
            QP = np.zeros((batch_size, n_r, Q.shape[1]))
            QP[b, I] = Q[None, :, :]

        copes = W @ r  # (batch, n_targets)
        if self.metric == "copes":
            return copes

        proj = QP.transpose(0, 2, 1) @ r  # (batch, rank, n_targets)
        ss = r2[None, :] - np.sum(proj**2, axis=1)
        varcopes = hat_adjustment(group["c_xtx_c"] * ss / group["dof"], self.hat)
        with np.errstate(divide="ignore", invalid="ignore"):
            return copes / np.sqrt(varcopes)

    def fit(self, y) -> TestResult:
        """Fit the GLM and run the permutations.

        Parameters
        ----------
        y : np.ndarray or list
            Data. Shape is (n_samples, \\*target_dims).

        Returns
        -------
        result : TestResult
            Observed statistics, p-values and the null distribution.
        """
        X = self.glm.X
        y, target_dims = validate_y(y, self.glm.n_samples)
        n_targets = y.shape[1]
        self._resolve_pooled_dims(target_dims)

        glm_result = self.glm.fit(y.reshape((-1, *target_dims)), hat=self.hat)
        self._before_permutations(glm_result)

        groups = [
            (self._prepare_group(X, y[:, targets], rows), targets)
            for rows, targets in complete_case_groups(X, y, warn=False)
        ]

        # The observed statistic is computed through the same code as the
        # null (with the identity permutation) so that comparisons between
        # the two are not affected by floating point differences
        if self.perm_type == "sign_flip":
            identity = np.ones((1, X.shape[0]), dtype=int)
        else:
            identity = np.arange(X.shape[0])[None]
        observed = np.empty((1, n_targets))
        for group, targets in groups:
            observed[:, targets] = self._null_batch(group, identity)
        self._observed = observed.reshape(target_dims)
        observed = self._transform_observed(self._observed)[None]

        all_complete = len(groups) == 1 and len(groups[0][0]["rows"]) == X.shape[0]
        perms, exact = self._permutations(allow_exact_shuffle=all_complete)
        n_perm = perms.shape[0]
        if exact:
            _logger.info(
                f"Enumerating all {n_perm} permutations of contrast "
                f"'{self.contrast_name}' (exact test)"
            )
        else:
            _logger.info(
                f"Running {n_perm} {self.perm_type} permutations of contrast "
                f"'{self.contrast_name}'"
            )

        nonpooled_dims = tuple(
            d for d in range(len(target_dims)) if d not in self.pooled_dims
        )
        null_shape = (n_perm, *[target_dims[d] for d in nonpooled_dims])
        keep_null = math.prod(null_shape) <= _MAX_NULL_ELEMENTS
        null = np.empty(null_shape) if keep_null else None
        if not keep_null:
            _logger.info("The null distribution is too large to keep")

        n_features = X.shape[1] + 1
        batch_size = int(
            np.clip(
                _MAX_BATCH_ELEMENTS // (n_features * n_targets * X.shape[0]), 1, n_perm
            )
        )
        if self._max_batch_size is not None:
            batch_size = min(batch_size, self._max_batch_size)

        counts = np.zeros(target_dims)
        bar = progress(total=n_perm, desc="Permutations")
        try:
            for start in range(0, n_perm, batch_size):
                batch = perms[start : start + batch_size]
                stat = np.empty((len(batch), n_targets))
                for group, targets in groups:
                    stat[:, targets] = self._null_batch(group, batch)
                reduced = self._reduce(stat.reshape((len(batch), *target_dims)))
                if keep_null:
                    null[start : start + len(batch)] = reduced
                counts += np.sum(self._expand(reduced) >= observed, axis=0)
                bar.update(len(batch))
        finally:
            bar.close()
            self._close()

        # Under an exact test the identity permutation is in the null and
        # equals the observed statistic, so no +1 is needed
        offset = 0 if exact else 1
        pvalues = (counts + offset) / (n_perm + offset)

        # A target with an undefined statistic (e.g. zero variance) has no
        # meaningful p-value
        undefined = ~np.isfinite(observed[0])
        if undefined.any():
            _logger.warning(
                f"{undefined.sum()} target(s) have an undefined statistic (e.g. "
                "zero variance). Their p-values are set to NaN."
            )
            pvalues[undefined] = np.nan
        if not exact and np.any(counts == 0):
            _logger.info(
                f"The smallest attainable p-value with {n_perm} permutations "
                f"is {1 / (n_perm + 1):.2g}. Increase n_perm for smaller p-values."
            )

        return TestResult(
            contrast=self.contrast_name,
            copes=glm_result.copes[self.contrast_indx],
            tstats=glm_result.tstats[self.contrast_indx],
            pvalues=pvalues,
            pvalues_uncorrected=pvalues if self.correction is None else None,
            correction=self.correction,
            tail=self.tail,
            metric=self.metric,
            parametric=False,
            n_perm=n_perm,
            perm_type=self.perm_type,
            exact=exact,
            pooled_dims=self.pooled_dims if self.correction else None,
            null=null,
            glm=glm_result,
        )

    def summary(self) -> dict:
        """Summary of the test settings."""
        summary = self.design.summary()
        summary.update(
            {
                "contrast": self.contrast_name,
                "n_perm": self.n_perm,
                "perm_type": self.perm_type,
                "tail": self.tail,
                "metric": self.metric,
                "correction": self.correction,
            }
        )
        return summary


class MaxStatPermutation(Permutation):
    """Permutation test with the maximum statistic across targets.

    The null distribution is the maximum of the statistic over the pooled
    target dimensions in each permutation, which controls the family-wise
    error rate across those dimensions [1]_.

    Parameters
    ----------
    design : Design
        The design (regressors and contrasts).
    contrast_indx : int, optional
        Index of the contrast to test.
    n_perm : int, optional
        Number of permutations.
    perm_type : str, optional
        :code:`'sign_flip'` or :code:`'row_shuffle'`. Chosen from the
        feature and contrast types if :code:`None`.
    tail : int, optional
        :code:`0` for two-tailed, :code:`1` for positive, :code:`-1` for
        negative effects.
    metric : str, optional
        :code:`'tstats'` or :code:`'copes'`. Use copes only when all the
        targets have the same units.
    pooled_dims : int or sequence of int, optional
        Target dimensions to take the maximum over (0 is the first dimension
        after the samples). Default is every target dimension, so the
        correction is across all targets. Dimensions that are not pooled get
        their own null distribution.
    hat : float, optional
        'Hat' variance adjustment, see
        :func:`~osl_dynamics.stats.glm.hat_adjustment`.
    n_jobs : int, optional
        Not used by this class.

    References
    ----------
    .. [1] Nichols, T. E. & Holmes, A. P. (2002). Nonparametric permutation
           tests for functional neuroimaging: a primer with examples. Human
           Brain Mapping, 15(1), 1-25.
    """

    correction = "max_stat"

    def __init__(
        self,
        design: Design,
        contrast_indx: int = 0,
        n_perm: int = 1000,
        perm_type: Optional[str] = None,
        tail: int = 0,
        metric: str = "tstats",
        pooled_dims: Optional[Sequence[int]] = None,
        hat: float = 0.0,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(
            design,
            contrast_indx=contrast_indx,
            n_perm=n_perm,
            perm_type=perm_type,
            tail=tail,
            metric=metric,
            hat=hat,
            n_jobs=n_jobs,
        )
        self._requested_pooled_dims = pooled_dims

    def _resolve_pooled_dims(self, target_dims: Tuple[int, ...]) -> None:
        n_dims = len(target_dims)
        if self._requested_pooled_dims is None:
            self.pooled_dims = tuple(range(n_dims))
            return
        dims = np.atleast_1d(self._requested_pooled_dims).astype(int)
        dims = tuple(
            sorted(set(d % n_dims if -n_dims <= d < n_dims else d for d in dims))
        )
        for d in dims:
            if not 0 <= d < n_dims:
                raise ValueError(
                    f"pooled_dims {self._requested_pooled_dims} out of range for "
                    f"data with {n_dims} target dimension(s)."
                )
        self.pooled_dims = dims


def _reduce_maps(
    maps: np.ndarray,
    structure: ClusterStructure,
    tail: int,
    threshold: Optional[float],
    tfce_params: Optional[dict],
) -> np.ndarray:
    """Maximum cluster statistic (or TFCE value) of each map in a batch."""
    if tfce_params is not None:
        return np.array([max_tfce(m, structure, tail, **tfce_params) for m in maps])
    return np.array([max_cluster_stat(m, threshold, structure, tail) for m in maps])


class ClusterPermutation(Permutation):
    """Cluster-based permutation test [1]_ or threshold-free cluster
    enhancement (TFCE) [2]_.

    Clusters are connected sets of targets whose t-statistic exceeds the
    cluster-forming threshold. The null distribution is the largest cluster
    statistic (the sum of the t-statistics in a cluster) found in each
    permutation, so the family-wise error rate is controlled across the
    whole map. A cluster's p-value says how often a cluster at least as
    large arises by chance anywhere in the map; it does not establish the
    effect at every target in the cluster, nor the cluster's exact extent.

    With :code:`tfce`, the t-statistics are enhanced over all thresholds
    instead and every target gets its own p-value.

    Parameters
    ----------
    design : Design
        The design (regressors and contrasts).
    contrast_indx : int, optional
        Index of the contrast to test.
    n_perm : int, optional
        Number of permutations.
    perm_type : str, optional
        :code:`'sign_flip'` or :code:`'row_shuffle'`. Chosen from the
        feature and contrast types if :code:`None`.
    tail : int, optional
        :code:`0` for two-tailed (positive and negative clusters are formed
        separately), :code:`1` for positive, :code:`-1` for negative effects.
    cluster_dims : int or sequence of int, optional
        Target dimensions along which clusters can extend (0 is the first
        dimension after the samples). Clusters never span two indices of any
        other dimension, e.g. two states, but the maximum is taken over them.
    adjacency : dict, optional
        Adjacency matrices for cluster dimensions that are not lattices,
        :code:`{dim: matrix}`. See
        :class:`~osl_dynamics.stats.cluster.ClusterStructure`.
    cluster_forming_threshold : float, optional
        Threshold on the t-statistic that forms clusters. Default is the
        critical t-value at p=0.05 (two-tailed: 0.025 in each tail) for the
        degrees of freedom of the data. Ignored with :code:`tfce`.
    tfce : bool or dict, optional
        Use threshold-free cluster enhancement. A dict passes options to
        :func:`~osl_dynamics.stats.cluster.tfce` (:code:`start`,
        :code:`step`, :code:`E`, :code:`H`).
    hat : float, optional
        'Hat' variance adjustment, see
        :func:`~osl_dynamics.stats.glm.hat_adjustment`.
    n_jobs : int, optional
        Number of processes used to find clusters in the permuted maps.

    References
    ----------
    .. [1] Maris, E. & Oostenveld, R. (2007). Nonparametric statistical
           testing of EEG- and MEG-data. Journal of Neuroscience Methods,
           164(1), 177-190.
    .. [2] Smith, S. M. & Nichols, T. E. (2009). Threshold-free cluster
           enhancement. NeuroImage, 44(1), 83-98.
    """

    _max_batch_size = 50

    def __init__(
        self,
        design: Design,
        contrast_indx: int = 0,
        n_perm: int = 1000,
        perm_type: Optional[str] = None,
        tail: int = 0,
        cluster_dims: Union[int, Sequence[int]] = 0,
        adjacency: Adjacency = None,
        cluster_forming_threshold: Optional[float] = None,
        tfce: Union[bool, dict] = False,
        hat: float = 0.0,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(
            design,
            contrast_indx=contrast_indx,
            n_perm=n_perm,
            perm_type=perm_type,
            tail=tail,
            metric="tstats",
            hat=hat,
            n_jobs=n_jobs,
        )
        self.cluster_dims = cluster_dims
        self.adjacency = adjacency
        if tfce is False or tfce is None:
            self.tfce_params = None
            self.correction = "cluster"
        else:
            self.tfce_params = {} if tfce is True else dict(tfce)
            self.correction = "tfce"
        if cluster_forming_threshold is not None and cluster_forming_threshold <= 0:
            raise ValueError("cluster_forming_threshold must be positive.")
        self._requested_threshold = cluster_forming_threshold
        self.cluster_forming_threshold = cluster_forming_threshold
        self.structure = None
        self._executor = None

    def _resolve_pooled_dims(self, target_dims: Tuple[int, ...]) -> None:
        self.pooled_dims = tuple(range(len(target_dims)))
        self.structure = ClusterStructure(
            target_dims, self.cluster_dims, self.adjacency
        )

    def _before_permutations(self, glm_result) -> None:
        if self.tfce_params is not None:
            return
        if self._requested_threshold is None:
            dof = np.nanmin(glm_result.dof)
            alpha = 0.025 if self.tail == 0 else 0.05
            self.cluster_forming_threshold = float(scipy_stats.t.ppf(1 - alpha, dof))
            _logger.info(
                "Using the critical t-value at p=0.05 as the cluster-forming "
                f"threshold: {self.cluster_forming_threshold:.3f} (dof={dof:.0f})"
            )

    def _transform_observed(self, stat: np.ndarray) -> np.ndarray:
        if self.tfce_params is not None:
            self._clusters, self._cluster_stats = None, None
            return np.abs(tfce(stat, self.structure, self.tail, **self.tfce_params))
        stat_map, self._clusters, self._cluster_stats = cluster_stat_map(
            stat, self.cluster_forming_threshold, self.structure, self.tail
        )
        return stat_map

    def _reduce(self, stat: np.ndarray) -> np.ndarray:
        args = (
            self.structure,
            self.tail,
            self.cluster_forming_threshold,
            self.tfce_params,
        )
        if self.n_jobs == 1 or len(stat) < 2:
            return _reduce_maps(stat, *args)
        if self._executor is None:
            from concurrent.futures import ProcessPoolExecutor

            self._executor = ProcessPoolExecutor(max_workers=self.n_jobs)
        chunks = np.array_split(stat, min(self.n_jobs, len(stat)))
        futures = [self._executor.submit(_reduce_maps, c, *args) for c in chunks]
        return np.concatenate([f.result() for f in futures])

    def _close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def fit(self, y) -> TestResult:
        """Fit the GLM and run the permutations.

        Parameters
        ----------
        y : np.ndarray or list
            Data. Shape is (n_samples, \\*target_dims).

        Returns
        -------
        result : TestResult
            For :code:`correction='cluster'`, :code:`result.clusters` holds
            a mask for each cluster, :code:`result.cluster_stats` their
            statistics and :code:`result.cluster_pvalues` their p-values;
            :code:`result.pvalues` gives every target the p-value of its
            cluster (1 outside clusters). For TFCE :code:`result.pvalues` are
            per-target p-values.
        """
        result = super().fit(y)
        result.cluster_forming_threshold = self.cluster_forming_threshold
        if self._clusters is not None:
            result.clusters = self._clusters
            result.cluster_stats = self._cluster_stats
            result.cluster_pvalues = np.array(
                [result.pvalues[mask].flat[0] for mask in self._clusters]
            )
            n_sig = int(np.sum(result.cluster_pvalues < 0.05))
            _logger.info(f"Found {len(self._clusters)} cluster(s), {n_sig} with p<0.05")
        return result
