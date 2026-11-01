"""Ready-made tests for common designs.

Every function builds a GLM design for the question asked, runs the test and
returns a :class:`~osl_dynamics.stats.results.TestResult`. For designs not
covered here build a :class:`~osl_dynamics.stats.design.DesignConfig` and
use :class:`~osl_dynamics.stats.permutation.MaxStatPermutation` or
:class:`~osl_dynamics.stats.glm.GLM` directly.
"""

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from osl_dynamics.stats.correction import CORRECTIONS, correct
from osl_dynamics.stats.design import Design, DesignConfig
from osl_dynamics.stats.glm import GLM
from osl_dynamics.stats.permutation import (
    ClusterPermutation,
    MaxStatPermutation,
    Permutation,
)
from osl_dynamics.stats.results import TestResult

_logger = logging.getLogger("osl-dynamics")

PERMUTATION_CORRECTIONS = ("max_stat", "cluster", "tfce")

_COMMON_PARAMETERS = """
    n_perm : int, optional
        Number of permutations. Ignored when :code:`parametric=True` and
        the correction is not permutation-based.
    covariates : dict, optional
        Continuous nuisance covariates to regress out, e.g.
        :code:`{"age": age, "head_size": head_size}`. Each value is a 1D
        array of length n_samples. They are z-scored internally.
    correction : str, optional
        Correction for multiple comparisons across the targets:

        - :code:`'max_stat'`: maximum statistic permutation test, controls
          the family-wise error rate (FWER). The default.
        - :code:`'cluster'`: cluster-based permutation test, FWER control
          with clusters formed along :code:`cluster_dims`. See
          :class:`~osl_dynamics.stats.permutation.ClusterPermutation`.
        - :code:`'tfce'`: threshold-free cluster enhancement, a
          cluster-based test without a cluster-forming threshold that gives
          per-target p-values.
        - :code:`'bonferroni'`, :code:`'holm'`: FWER control applied to
          uncorrected p-values.
        - :code:`'fdr'`: Benjamini-Hochberg false discovery rate control
          applied to uncorrected p-values.
        - :code:`None`: uncorrected p-values.

    metric : str, optional
        Statistic used by the :code:`'max_stat'` permutation test and
        per-target permutation p-values: :code:`'tstats'` or
        :code:`'copes'`. Cluster-based corrections always use t-statistics.
    tail : int, optional
        :code:`0` for a two-tailed test, :code:`1` to test for a positive
        effect, :code:`-1` for a negative effect.
    pooled_dims : int or sequence of int, optional
        For :code:`correction='max_stat'`, the target dimensions to take the
        maximum over (0 is the first dimension after the samples). Default is
        all of them, i.e. the FWER is controlled across every target.
    cluster_dims : int or sequence of int, optional
        For :code:`'cluster'` and :code:`'tfce'`: target dimensions along
        which clusters can extend (0 is the first dimension after the
        samples), e.g. time or frequency. Neighbouring indices are adjacent
        unless an :code:`adjacency` is given for that dimension. Clusters
        never span two indices of other dimensions (e.g. two states).
        Required for these corrections, except in
        :func:`~osl_dynamics.stats.inference.evoked_response` where it
        defaults to time (dimension 0).
    adjacency : dict, optional
        For :code:`'cluster'` and :code:`'tfce'`: adjacency matrices for
        cluster dimensions that are not lattices, :code:`{dim: matrix}`
        where :code:`matrix` has shape :code:`(n, n)` for a dimension of
        size :code:`n` (dense or :mod:`scipy.sparse`, e.g. from
        :func:`mne.channels.find_ch_adjacency`).
    cluster_forming_threshold : float, optional
        For :code:`'cluster'`: threshold on the t-statistic that forms
        clusters. Default is the critical t-value at p=0.05.
    hat : float, optional
        'Hat' variance adjustment, a fraction of the maximum variance added
        to every target's variance so that targets with tiny variances do
        not produce huge t-statistics. See
        :func:`~osl_dynamics.stats.glm.hat_adjustment`. Default is :code:`0`.
    parametric : bool, optional
        For :code:`'bonferroni'`, :code:`'holm'`, :code:`'fdr'` and
        :code:`None`: whether the uncorrected p-values come from the t
        distribution (:code:`True`, the default) or from per-target
        permutation tests (:code:`False`).
    n_jobs : int, optional
        Number of processes used to find clusters in the permuted maps for
        :code:`'cluster'` and :code:`'tfce'`, which only pays off for large
        maps or TFCE. Not used otherwise. A script that uses more than one
        process must keep its code under :code:`if __name__ == "__main__":`.

    Returns
    -------
    result : osl_dynamics.stats.results.TestResult
        :code:`result.copes` is the effect, :code:`result.pvalues` the
        corrected p-values and :code:`result.significant()` a boolean mask,
        each with shape (\\*target_dims).
"""


def _covariate_features(covariates: Optional[Dict[str, np.ndarray]]) -> list:
    if covariates is None:
        return []
    return [
        {"name": name, "values": np.asarray(values), "feature_type": "continuous"}
        for name, values in covariates.items()
    ]


def _run(
    design: Design,
    data: np.ndarray,
    n_perm: int,
    correction: Optional[str],
    metric: str,
    tail: int,
    pooled_dims,
    cluster_dims,
    adjacency,
    cluster_forming_threshold: Optional[float],
    hat: float,
    parametric: bool,
    n_jobs: int,
    contrast_indx: int = 0,
) -> TestResult:
    if correction == "max_stat":
        perm = MaxStatPermutation(
            design,
            contrast_indx=contrast_indx,
            n_perm=n_perm,
            tail=tail,
            metric=metric,
            pooled_dims=pooled_dims,
            hat=hat,
        )
        return perm.fit(data)

    if correction in ("cluster", "tfce"):
        if cluster_dims is None:
            raise ValueError(
                f"cluster_dims must be given for correction='{correction}': the "
                "target dimension(s) along which clusters can extend, e.g. time."
            )
        if pooled_dims is not None:
            _logger.warning(
                f"pooled_dims is ignored with correction='{correction}': the "
                "maximum is always taken over the whole map."
            )
        if metric != "tstats":
            _logger.warning(
                f"metric='{metric}' is ignored with correction='{correction}': "
                "clusters are always formed from t-statistics."
            )
        perm = ClusterPermutation(
            design,
            contrast_indx=contrast_indx,
            n_perm=n_perm,
            tail=tail,
            cluster_dims=cluster_dims,
            adjacency=adjacency,
            cluster_forming_threshold=cluster_forming_threshold,
            tfce=correction == "tfce",
            hat=hat,
            n_jobs=n_jobs,
        )
        return perm.fit(data)

    if correction is not None and correction not in CORRECTIONS:
        raise ValueError(
            "correction must be one of "
            f"{PERMUTATION_CORRECTIONS + CORRECTIONS} or None, got {correction}."
        )

    if parametric:
        glm_result = GLM(design).fit(data, hat=hat)
        return glm_result.test(contrast_indx, correction=correction, tail=tail)

    perm = Permutation(
        design,
        contrast_indx=contrast_indx,
        n_perm=n_perm,
        tail=tail,
        metric=metric,
        hat=hat,
    )
    result = perm.fit(data)
    n_targets = result.pvalues.size
    smallest_p = 1 / (result.n_perm + (0 if result.exact else 1))
    # Bonferroni/Holm need the smallest p-value below alpha / n_targets to
    # reject anything; Benjamini-Hochberg needs it below alpha at least
    needed = {"bonferroni": 0.05 / n_targets, "holm": 0.05 / n_targets, "fdr": 0.05}
    if correction is not None and smallest_p > needed[correction]:
        _logger.warning(
            f"The smallest attainable p-value with {result.n_perm} permutations is "
            f"{smallest_p:.2g}, but {correction} needs p-values below "
            f"{needed[correction]:.2g} to reject anything at alpha=0.05 across "
            f"{n_targets} targets. Increase n_perm."
        )
    result.pvalues = correct(result.pvalues_uncorrected, correction)
    result.correction = correction
    return result


def _one_sample_design(n_samples: int, covariates, name: str) -> Design:
    features = [
        {"name": name, "values": np.ones(n_samples), "feature_type": "constant"}
    ] + _covariate_features(covariates)
    contrasts = [{"name": name, "values": [1] + [0] * (len(features) - 1)}]
    return DesignConfig(features=features, contrasts=contrasts).create_design()


def evoked_response(
    data: np.ndarray,
    n_perm: int = 1000,
    covariates: Optional[Dict[str, np.ndarray]] = None,
    correction: Optional[str] = "max_stat",
    metric: Optional[str] = None,
    tail: int = 0,
    pooled_dims: Optional[Sequence[int]] = None,
    cluster_dims: Optional[Sequence[int]] = None,
    adjacency: Optional[Dict] = None,
    cluster_forming_threshold: Optional[float] = None,
    hat: float = 0.0,
    parametric: bool = True,
    n_jobs: int = 1,
) -> TestResult:
    """Test whether an evoked (e.g. network) response differs from zero.

    A one-sample test of the mean over samples (usually subjects) at every
    target, with sign-flip permutations. The data should be baseline
    corrected so that zero is the null. For cluster-based corrections,
    clusters are formed along time (the first target dimension) by default.
    With the default :code:`correction='max_stat'`, :code:`metric` defaults
    to :code:`'copes'` (the mean response, which has the same units at every
    target) rather than :code:`'tstats'`.

    Parameters
    ----------
    data : np.ndarray
        Baseline-corrected responses. Shape is (n_samples, \\*target_dims),
        e.g. (n_subjects, n_time, n_networks).
    metric : str, optional
        :code:`'copes'` for :code:`'max_stat'` and :code:`'tstats'` otherwise
        if :code:`None`.
    """
    data = np.asarray(data, dtype=float)
    if metric is None:
        metric = "copes" if correction == "max_stat" else "tstats"
    if cluster_dims is None:
        cluster_dims = 0  # time is the first target dimension
    design = _one_sample_design(data.shape[0], covariates, "Mean")
    return _run(
        design,
        data,
        n_perm,
        correction,
        metric,
        tail,
        pooled_dims,
        cluster_dims,
        adjacency,
        cluster_forming_threshold,
        hat,
        parametric,
        n_jobs,
    )


def paired_diff(
    data: np.ndarray,
    n_perm: int = 1000,
    covariates: Optional[Dict[str, np.ndarray]] = None,
    correction: Optional[str] = "max_stat",
    metric: str = "tstats",
    tail: int = 0,
    pooled_dims: Optional[Sequence[int]] = None,
    cluster_dims: Optional[Sequence[int]] = None,
    adjacency: Optional[Dict] = None,
    cluster_forming_threshold: Optional[float] = None,
    hat: float = 0.0,
    parametric: bool = True,
    n_jobs: int = 1,
) -> TestResult:
    """Test whether a paired difference differs from zero.

    A one-sample test of the mean difference over samples (usually subjects)
    at every target, with sign-flip permutations.

    Parameters
    ----------
    data : np.ndarray
        Paired differences, e.g. condition A minus condition B for each
        subject. Shape is (n_samples, \\*target_dims).
    """
    data = np.asarray(data, dtype=float)
    design = _one_sample_design(data.shape[0], covariates, "PairedDiff")
    return _run(
        design,
        data,
        n_perm,
        correction,
        metric,
        tail,
        pooled_dims,
        cluster_dims,
        adjacency,
        cluster_forming_threshold,
        hat,
        parametric,
        n_jobs,
    )


def group_diff(
    data: np.ndarray,
    assignments: np.ndarray,
    n_perm: int = 1000,
    groups: Optional[Tuple] = None,
    covariates: Optional[Dict[str, np.ndarray]] = None,
    correction: Optional[str] = "max_stat",
    metric: str = "tstats",
    tail: int = 0,
    pooled_dims: Optional[Sequence[int]] = None,
    cluster_dims: Optional[Sequence[int]] = None,
    adjacency: Optional[Dict] = None,
    cluster_forming_threshold: Optional[float] = None,
    hat: float = 0.0,
    parametric: bool = True,
    n_jobs: int = 1,
) -> TestResult:
    """Test for a difference between two groups.

    Each unique value in :code:`assignments` is modelled as a group and the
    two groups in :code:`groups` are compared, with row-shuffle permutations
    of the samples in those two groups.

    Parameters
    ----------
    data : np.ndarray
        Data. Shape is (n_samples, \\*target_dims).
    assignments : np.ndarray
        Group label of each sample. Shape is (n_samples,).
    groups : tuple, optional
        The two labels to compare, :code:`(group1, group2)`. The effect
        returned is :code:`group1 - group2`. Required when there are more
        than two labels. Defaults to the two labels in sorted order.
    """
    data = np.asarray(data, dtype=float)
    assignments = np.asarray(assignments)
    unique_groups = np.unique(assignments)
    if len(unique_groups) < 2:
        raise ValueError("assignments must have at least two unique values.")

    if groups is None:
        if len(unique_groups) != 2:
            raise ValueError(
                "groups must be specified when assignments has more than two "
                f"unique values, got {unique_groups}."
            )
        groups = tuple(unique_groups)
    if len(groups) != 2 or groups[0] == groups[1]:
        raise ValueError(f"groups must be two different labels, got {groups}.")
    for group in groups:
        if group not in unique_groups:
            raise ValueError(f"group {group} not found in assignments.")

    features = [
        {
            "name": f"Group_{group}",
            "values": (assignments == group).astype(float),
            "feature_type": "categorical",
        }
        for group in unique_groups
    ] + _covariate_features(covariates)
    contrast = [0] * len(features)
    contrast[list(unique_groups).index(groups[0])] = 1
    contrast[list(unique_groups).index(groups[1])] = -1
    contrasts = [{"name": f"Group_{groups[0]}-Group_{groups[1]}", "values": contrast}]
    design = DesignConfig(features=features, contrasts=contrasts).create_design()
    return _run(
        design,
        data,
        n_perm,
        correction,
        metric,
        tail,
        pooled_dims,
        cluster_dims,
        adjacency,
        cluster_forming_threshold,
        hat,
        parametric,
        n_jobs,
    )


def regression(
    data: np.ndarray,
    covariate: np.ndarray,
    n_perm: int = 1000,
    covariates: Optional[Dict[str, np.ndarray]] = None,
    name: str = "Covariate",
    correction: Optional[str] = "max_stat",
    metric: str = "tstats",
    tail: int = 0,
    pooled_dims: Optional[Sequence[int]] = None,
    cluster_dims: Optional[Sequence[int]] = None,
    adjacency: Optional[Dict] = None,
    cluster_forming_threshold: Optional[float] = None,
    hat: float = 0.0,
    parametric: bool = True,
    n_jobs: int = 1,
) -> TestResult:
    """Test for a linear effect of a continuous variable.

    The design is a constant plus the variable of interest (and any nuisance
    covariates), with row-shuffle permutations. The effect returned is per
    standard deviation of the variable because it is z-scored.

    Parameters
    ----------
    data : np.ndarray
        Data. Shape is (n_samples, \\*target_dims).
    covariate : np.ndarray
        Variable of interest, e.g. age. Shape is (n_samples,).
    name : str, optional
        Name of the variable of interest.
    """
    data = np.asarray(data, dtype=float)
    features = [
        {"name": "Mean", "values": np.ones(data.shape[0]), "feature_type": "constant"},
        {"name": name, "values": np.asarray(covariate), "feature_type": "continuous"},
    ] + _covariate_features(covariates)
    contrasts = [{"name": name, "values": [0, 1] + [0] * (len(features) - 2)}]
    design = DesignConfig(features=features, contrasts=contrasts).create_design()
    return _run(
        design,
        data,
        n_perm,
        correction,
        metric,
        tail,
        pooled_dims,
        cluster_dims,
        adjacency,
        cluster_forming_threshold,
        hat,
        parametric,
        n_jobs,
    )


for _function in (evoked_response, paired_diff, group_diff, regression):
    _function.__doc__ += _COMMON_PARAMETERS
del _function
