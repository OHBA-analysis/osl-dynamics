"""Functions to perform statistical significance testing."""

from typing import Dict, Optional, Tuple

import numpy as np

from osl_dynamics.glm import DesignConfig, MaxStatPermutation


def evoked_response_max_stat_perm(
    data: np.ndarray,
    n_perm: int,
    covariates: Optional[Dict[str, np.ndarray]] = None,
    metric: str = "copes",
    n_jobs: int = 1,
) -> np.ndarray:
    r"""Statistical significance testing for evoked responses.

    This function fits a General Linear Model (GLM) with ordinary least
    squares and performs a sign-flip permutation to build a null distribution
    using maximum statistic across the target dimensions.

    Parameters
    ----------
    data : np.ndarray
        Data array for baseline corrected evoked responses.
        Shape is (n_samples, \*target_dims).
    n_perm : int
        Number of permutations.
    covariates : dict, optional
        Dictionary of continuous covariates to include as confound regressors.
        Each key is a covariate name and each value is a 1D array of length
        n_samples. These are z-scored internally. Example::

            covariates = {
                "age": np.array([25, 30, 28, ...]),
                "head_size": np.array([55.1, 57.3, 54.8, ...]),
            }

    metric : str, optional
        Statistic to compute p-values.
        Options are :code:`'copes'` or :code:`'tstats'`.
    n_jobs : int, optional
        Number of jobs to run in parallel.

    Returns
    -------
    pvalues : np.ndarray
        P-values. Shape is (\*target_dims,).
    """
    features = [
        {"name": "Mean", "values": np.ones(data.shape[0]), "feature_type": "constant"}
    ]
    if covariates is None:
        covariates = {}

    for key, value in covariates.items():
        features.append({"name": key, "values": value, "feature_type": "continuous"})
    contrasts = [{"name": "Mean", "values": [1] + [0] * len(covariates)}]

    DC = DesignConfig(features=features, contrasts=contrasts)
    design = DC.create_design()

    perm = MaxStatPermutation(
        design,
        contrast_indx=0,
        n_perm=n_perm,
        n_jobs=n_jobs,
    )
    perm.fit(data)
    pvalues = perm.get_pvalues(metric=metric)

    return pvalues


def group_diff_max_stat_perm(
    data: np.ndarray,
    assignments: np.ndarray,
    n_perm: int,
    groups: Optional[Tuple] = None,
    covariates: Optional[Dict[str, np.ndarray]] = None,
    metric: str = "tstats",
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Statistical significance testing for difference between two groups.

    This function fits a General Linear Model (GLM) with ordinary least
    squares and performs a row-shuffle permutation to build a null distribution
    using maximum statistic across the target dimensions (target_dims)
    for the difference between two groups. Each unique value in
    :code:`assignments` is modelled as a separate group, but only the two
    groups being compared are permuted.

    Parameters
    ----------
    data : np.ndarray
        Data array for baseline corrected evoked responses.
        Shape is (n_samples, \*target_dims).
    assignments : np.ndarray
        Group assignments. Shape is (n_samples,).
    n_perm : int
        Number of permutations.
    groups : tuple, optional
        The two group labels to compare, :code:`(group1, group2)`. The
        difference returned is :code:`group1 - group2`. Required if
        :code:`assignments` has more than two unique values. Defaults to the
        two unique values in :code:`assignments` in sorted order.
    covariates : dict, optional
        Dictionary of continuous covariates to include as confound regressors.
        Each key is a covariate name and each value is a 1D array of length
        n_samples. These are z-scored internally. Example::

            covariates = {
                "age": np.array([25, 30, 28, ...]),
                "head_size": np.array([55.1, 57.3, 54.8, ...]),
            }

    metric : str, optional
        Statistic to compute p-values with.
        Options are :code:`'copes'` or :code:`'tstats'`.
    n_jobs : int, optional
        Number of jobs to run in parallel.

    Returns
    -------
    group_diff : np.ndarray
        Difference between two groups. Shape is (\*target_dims,).
    pvalues : np.ndarray
        P-values. Shape is (\*target_dims,).
    """
    if covariates is None:
        covariates = {}

    assignments = np.asarray(assignments)
    unique_groups = np.unique(assignments)
    if len(unique_groups) < 2:
        raise ValueError("assignments must have at least two unique values.")

    if groups is None:
        if len(unique_groups) != 2:
            raise ValueError(
                "groups must be specified if assignments has more than two "
                f"unique values, got {unique_groups}."
            )
        groups = tuple(unique_groups)
    if len(groups) != 2 or groups[0] == groups[1]:
        raise ValueError(f"groups must be two different group labels, got {groups}.")
    for group in groups:
        if group not in unique_groups:
            raise ValueError(f"group {group} not found in assignments.")

    features = [
        {
            "name": f"Group_{group}",
            "values": (assignments == group).astype(int),
            "feature_type": "categorical",
        }
        for group in unique_groups
    ]
    for key, value in covariates.items():
        features.append({"name": key, "values": value, "feature_type": "continuous"})

    contrast_values = [0] * len(features)
    contrast_values[list(unique_groups).index(groups[0])] = 1
    contrast_values[list(unique_groups).index(groups[1])] = -1
    contrasts = [{"name": "GroupDiff", "values": contrast_values}]

    DC = DesignConfig(features=features, contrasts=contrasts)
    design = DC.create_design()

    perm = MaxStatPermutation(
        design,
        contrast_indx=0,
        n_perm=n_perm,
        n_jobs=n_jobs,
    )
    perm.fit(data)
    group_diff = perm.copes
    pvalues = perm.get_pvalues(metric=metric)

    return group_diff, pvalues


def paired_diff_max_stat_perm(
    data: np.ndarray,
    n_perm: int,
    metric: str = "copes",
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Statistical significance testing for paired difference.

    This function fits a General Linear Model (GLM) with ordinary least
    squares and performs sign-flip permutations to build a null distribution
    with the maximum statistic across the target dimensions for the paired
    differences.

    Parameters
    ----------
    data : np.ndarray
        Data array for baseline corrected evoked responses.
        Shape is (n_samples, \*target_dims).
    n_perm : int
        Number of permutations.
    metric : str, optional
        Statistic to compute p-values with.
        Options are :code:`'copes'` or :code:`'tstats'`.
    n_jobs : int, optional
        Number of jobs to run in parallel.

    Returns
    -------
    paired_diff : np.ndarray
        Paired differences. Shape is (\*target_dims,).
    pvalues : np.ndarray
        P-values. Shape is (\*target_dims,).
    """
    features = [
        {"name": "Mean", "values": np.ones(data.shape[0]), "feature_type": "constant"}
    ]
    contrasts = [{"name": "Mean", "values": [1]}]

    DC = DesignConfig(features=features, contrasts=contrasts)
    design = DC.create_design()

    perm = MaxStatPermutation(
        design,
        contrast_indx=0,
        n_perm=n_perm,
        n_jobs=n_jobs,
    )
    perm.fit(data)
    pvalues = perm.get_pvalues(metric=metric)
    paired_diff = perm.copes

    return paired_diff, pvalues
