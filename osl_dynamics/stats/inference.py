"""Ready-made tests for common designs.

Every function builds a GLM design for the question asked, runs the test and
returns a :class:`~osl_dynamics.stats.results.TestResult`. For designs not
covered here build a :class:`~osl_dynamics.stats.design.DesignConfig` and
use :class:`~osl_dynamics.stats.permutation.MaxStatPermutation` or
:class:`~osl_dynamics.stats.glm.GLM` directly.
"""

import logging
import re
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
        effect, :code:`-1` for a negative effect. An F-test (three or more
        groups) has no direction and requires :code:`0`.
    pooled_dims : int or sequence of int, optional
        For :code:`correction='max_stat'`, the target dimensions to take the
        maximum over (0 is the first dimension after the samples). Default is
        all of them, i.e. the FWER is controlled across every target.
    cluster_dims : int or sequence of int, optional
        For :code:`'cluster'` and :code:`'tfce'`: target dimensions along
        which clusters can extend (0 is the first dimension after the
        samples), e.g. time or frequency. Neighboring indices are adjacent
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
        each with shape (\\*target_dims). :code:`result.stats` is the test
        statistic (t, or F for three or more groups).
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
    *,
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
    contrast_index: int = 0,
    statistic: str = "t",
    blocks: Optional[np.ndarray] = None,
) -> TestResult:
    """Run the test selected by :code:`correction` and :code:`parametric`."""
    if statistic == "F":
        if tail != 0:
            raise ValueError(
                "An F-test (three or more groups) has no direction, tail must "
                "be 0. Pass groups=(a, b) to test a directed difference between "
                "two of the groups."
            )
        if metric != "tstats":
            raise ValueError("metric='copes' is not available for an F-test.")

    if correction == "max_stat":
        perm = MaxStatPermutation(
            design,
            contrast_index=contrast_index,
            n_perm=n_perm,
            tail=tail,
            metric=metric,
            pooled_dims=pooled_dims,
            hat=hat,
            statistic=statistic,
            blocks=blocks,
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
            contrast_index=contrast_index,
            n_perm=n_perm,
            tail=tail,
            cluster_dims=cluster_dims,
            adjacency=adjacency,
            cluster_forming_threshold=cluster_forming_threshold,
            tfce=correction == "tfce",
            hat=hat,
            n_jobs=n_jobs,
            statistic=statistic,
            blocks=blocks,
        )
        return perm.fit(data)

    if correction is not None and correction not in CORRECTIONS:
        raise ValueError(
            "correction must be one of "
            f"{PERMUTATION_CORRECTIONS + CORRECTIONS} or None, got {correction}."
        )

    if parametric:
        glm_result = GLM(design).fit(data, hat=hat)
        if statistic == "F":
            result = glm_result.ftest(contrast_index, correction=correction)
        else:
            result = glm_result.test(contrast_index, correction=correction, tail=tail)
        result.blocks = blocks  # for bootstrapping effect sizes
        return result

    perm = Permutation(
        design,
        contrast_index=contrast_index,
        n_perm=n_perm,
        tail=tail,
        metric=metric,
        hat=hat,
        statistic=statistic,
        blocks=blocks,
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
        n_perm=n_perm,
        correction=correction,
        metric=metric,
        tail=tail,
        pooled_dims=pooled_dims,
        cluster_dims=cluster_dims,
        adjacency=adjacency,
        cluster_forming_threshold=cluster_forming_threshold,
        hat=hat,
        parametric=parametric,
        n_jobs=n_jobs,
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
    at every target, with sign-flip permutations. For more than two
    conditions see :func:`condition_diff`.

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
        n_perm=n_perm,
        correction=correction,
        metric=metric,
        tail=tail,
        pooled_dims=pooled_dims,
        cluster_dims=cluster_dims,
        adjacency=adjacency,
        cluster_forming_threshold=cluster_forming_threshold,
        hat=hat,
        parametric=parametric,
        n_jobs=n_jobs,
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
    """Test for a difference between groups.

    Each unique value in :code:`assignments` is modeled as a group. With two
    groups (or :code:`groups`), the difference between them is tested with a
    t-contrast and row-shuffle permutations of the samples in those two
    groups. With three or more groups and no :code:`groups`, an F-test asks
    whether any group differs from the others (a one-way ANOVA with
    permutations); significant targets can then be followed up with
    pairwise tests by passing :code:`groups`.

    Parameters
    ----------
    data : np.ndarray
        Data. Shape is (n_samples, \\*target_dims).
    assignments : np.ndarray
        Group label of each sample. Shape is (n_samples,).
    groups : tuple, optional
        Two labels to compare, :code:`(group1, group2)`. The effect returned
        is :code:`group1 - group2`. Default is to compare the two labels when
        there are two, and to test for any difference between the groups
        (F-test) when there are more. For the F-test :code:`result.copes`
        holds the differences between consecutive groups (in sorted order),
        :code:`result.stats` the F-statistics, and :code:`tail` must be 0.
    """
    data = np.asarray(data, dtype=float)
    assignments = np.asarray(assignments)
    unique_groups = list(np.unique(assignments))
    if len(unique_groups) < 2:
        raise ValueError("assignments must have at least two unique values.")

    if groups is None and len(unique_groups) == 2:
        groups = tuple(unique_groups)
    if groups is not None:
        if len(groups) != 2 or groups[0] == groups[1]:
            raise ValueError(f"groups must be two different labels, got {groups}.")
        for group in groups:
            if group not in unique_groups:
                raise ValueError(f"group {group} not found in assignments.")
        # Canonical labels (e.g. the user may pass 1.0 for the label 1)
        groups = tuple(unique_groups[unique_groups.index(group)] for group in groups)

    features = [
        {
            "name": f"Group_{group}",
            "values": (assignments == group).astype(float),
            "feature_type": "categorical",
        }
        for group in unique_groups
    ] + _covariate_features(covariates)

    if groups is not None:
        contrast = [0] * len(features)
        contrast[unique_groups.index(groups[0])] = 1
        contrast[unique_groups.index(groups[1])] = -1
        contrasts = [
            {"name": f"Group_{groups[0]}-Group_{groups[1]}", "values": contrast}
        ]
        design = DesignConfig(features=features, contrasts=contrasts).create_design()
        statistic = "t"
    else:
        _logger.info(
            f"Testing for any difference between the {len(unique_groups)} groups "
            "with an F-test. Pass groups=(a, b) to compare two of them."
        )
        rows = []
        for a, b in zip(unique_groups[:-1], unique_groups[1:]):
            row = [0] * len(features)
            row[unique_groups.index(a)] = 1
            row[unique_groups.index(b)] = -1
            rows.append(row)
        f_contrasts = [{"name": "AnyGroupDifference", "values": rows}]
        design = DesignConfig(
            features=features, f_contrasts=f_contrasts
        ).create_design()
        statistic = "F"

    return _run(
        design,
        data,
        n_perm=n_perm,
        correction=correction,
        metric=metric,
        tail=tail,
        pooled_dims=pooled_dims,
        cluster_dims=cluster_dims,
        adjacency=adjacency,
        cluster_forming_threshold=cluster_forming_threshold,
        hat=hat,
        parametric=parametric,
        n_jobs=n_jobs,
        statistic=statistic,
    )


def condition_diff(
    data: np.ndarray,
    n_perm: int = 1000,
    conditions: Optional[Tuple[int, int]] = None,
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
    """Test for a difference between conditions measured in the same subjects.

    A within-subject (repeated measures) design: the GLM has an indicator
    regressor per condition and one per subject (nuisance), and the
    permutations shuffle the conditions within each subject. With two
    conditions (or :code:`conditions`) the difference between them is tested
    with a t-contrast. With three or more conditions and no
    :code:`conditions`, an F-test asks whether any condition differs from
    the others (a repeated measures ANOVA with permutations); significant
    targets can then be followed up with pairwise tests by passing
    :code:`conditions`.

    Parameters
    ----------
    data : np.ndarray
        Data for each subject and condition. Shape is
        (n_subjects, n_conditions, \\*target_dims). A subject that lacks a
        condition for some targets can have NaNs there.
    conditions : tuple of int, optional
        Two conditions to compare, :code:`(a, b)` (indices along the second
        dimension). The effect returned is :code:`a - b`. Default is to
        compare the two conditions when there are two, and to test for any
        difference between the conditions (F-test) when there are more. For
        the F-test :code:`result.copes` holds the differences between
        consecutive conditions, :code:`result.stats` the F-statistics, and
        :code:`tail` must be 0.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim < 2:
        raise ValueError(
            "data must have shape (n_subjects, n_conditions, *target_dims)."
        )
    n_subjects, n_conditions = data.shape[:2]
    if n_conditions < 2:
        raise ValueError("data must have at least two conditions.")
    if conditions is None and n_conditions == 2:
        conditions = (0, 1)
    if conditions is not None:
        if len(conditions) != 2 or conditions[0] == conditions[1]:
            raise ValueError(
                f"conditions must be two different indices, got {conditions}."
            )
        for c in conditions:
            if c != int(c) or not 0 <= c < n_conditions:
                raise ValueError(
                    f"condition {c} is not an index into {n_conditions} conditions."
                )
        conditions = tuple(int(c) for c in conditions)

    # One row per subject and condition
    y = data.reshape((n_subjects * n_conditions, *data.shape[2:]))
    condition = np.tile(np.arange(n_conditions), n_subjects)
    subject = np.repeat(np.arange(n_subjects), n_conditions)
    features = [
        {
            "name": f"Condition_{c}",
            "values": (condition == c).astype(float),
            "feature_type": "categorical",
        }
        for c in range(n_conditions)
    ] + [
        {
            "name": f"Subject_{s}",
            "values": (subject == s).astype(float),
            "feature_type": "categorical",
        }
        for s in range(n_subjects)
    ]

    if conditions is not None:
        contrast = [0] * len(features)
        contrast[conditions[0]] = 1
        contrast[conditions[1]] = -1
        name = f"Condition_{conditions[0]}-Condition_{conditions[1]}"
        design = DesignConfig(
            features=features, contrasts=[{"name": name, "values": contrast}]
        ).create_design()
        statistic = "t"
    else:
        _logger.info(
            f"Testing for any difference between the {n_conditions} conditions "
            "with an F-test. Pass conditions=(a, b) to compare two of them."
        )
        rows = []
        for a in range(n_conditions - 1):
            row = [0] * len(features)
            row[a], row[a + 1] = 1, -1
            rows.append(row)
        design = DesignConfig(
            features=features,
            f_contrasts=[{"name": "AnyConditionDifference", "values": rows}],
        ).create_design()
        statistic = "F"

    return _run(
        design,
        y,
        n_perm=n_perm,
        correction=correction,
        metric=metric,
        tail=tail,
        pooled_dims=pooled_dims,
        cluster_dims=cluster_dims,
        adjacency=adjacency,
        cluster_forming_threshold=cluster_forming_threshold,
        hat=hat,
        parametric=parametric,
        n_jobs=n_jobs,
        statistic=statistic,
        blocks=subject,
    )


def regression(
    data: np.ndarray,
    regressor: np.ndarray,
    n_perm: int = 1000,
    covariates: Optional[Dict[str, np.ndarray]] = None,
    name: str = "Regressor",
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
    regressor : np.ndarray
        Variable of interest, e.g. age. Shape is (n_samples,).
    name : str, optional
        Name of the variable of interest.
    """
    data = np.asarray(data, dtype=float)
    features = [
        {"name": "Mean", "values": np.ones(data.shape[0]), "feature_type": "constant"},
        {"name": name, "values": np.asarray(regressor), "feature_type": "continuous"},
    ] + _covariate_features(covariates)
    contrasts = [{"name": name, "values": [0, 1] + [0] * (len(features) - 2)}]
    design = DesignConfig(features=features, contrasts=contrasts).create_design()
    return _run(
        design,
        data,
        n_perm=n_perm,
        correction=correction,
        metric=metric,
        tail=tail,
        pooled_dims=pooled_dims,
        cluster_dims=cluster_dims,
        adjacency=adjacency,
        cluster_forming_threshold=cluster_forming_threshold,
        hat=hat,
        parametric=parametric,
        n_jobs=n_jobs,
    )


for _function in (evoked_response, paired_diff, group_diff, regression):
    _function.__doc__ += _COMMON_PARAMETERS
# Subject-level covariates are absorbed by the subject regressors
condition_diff.__doc__ += re.sub(
    r"    covariates : .*?(?=\n    \w+ :)", "", _COMMON_PARAMETERS, flags=re.S
)
del _function
