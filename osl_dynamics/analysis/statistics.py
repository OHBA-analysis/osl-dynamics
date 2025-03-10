"""Functions to perform statistical significance testing."""

import numpy as np

from osl_dynamics.glm import DesignConfig, MaxStatPermutation


def evoked_response_max_stat_perm(
    data, n_perm, covariates=None, metric="copes", n_jobs=1
):
    """
    Statistical significance testing for evoked responses.

    Parameters
    ----------
    data : np.ndarray
        Data array for baseline corrected evoked responses.
        Shape is (n_samples, *target_dims).
    n_perm : int
        Number of permutations.
    covariates : dict, optional
        Dictionary of continuous covariates.
            - key: name of the covariate
            - value: np.ndarray of covariate values. Shape is (n_samples,).
        Default is None.
    metric : str, optional
        Metric to compute p-values. Options are 'copes' and 'tstats'.
        Default is 'copes'.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is 1.

    Returns
    -------
    pvalues : np.ndarray
        P-values. Shape is (*target_dims,).
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
    data, assignments, n_perm, covariates=None, metric="tstats", n_jobs=1
):
    """
    Statistical significance testing for difference between two groups.

    This function fits a General Linear Model (GLM) with ordinary least squares
    and performs a row shuffle permutation test with maximum statistic to
    determine a p-value for the difference between two groups.

    Parameters
    ----------
    data : np.ndarray
        Data array for baseline corrected evoked responses.
        Shape is (n_samples, *target_dims).
    assignments : np.ndarray
        Group assignments. Shape is (n_samples,). Must have exactly two unique values.
    n_perm : int
        Number of permutations.
    covariates : dict, optional
        Dictionary of continuous covariates.
            - key: name of the covariate
            - value: np.ndarray of covariate values. Shape is (n_samples,).
        Default is None.
    metric : str, optional
        Metric to compute p-values. Options are 'copes' and 'tstats'.
        Default is 'copes'.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is 1.

    Returns
    -------
    group_diff : np.ndarray
        Difference between two groups. Shape is (*target_dims,).
    pvalues : np.ndarray
        P-values. Shape is (*target_dims,).
    """
    if covariates is None:
        covariates = {}

    unique_groups = np.unique(assignments)
    if len(unique_groups) != 2:
        raise ValueError("assignments must have exactly two unique values.")

    features = [
        {
            "name": "Group1",
            "values": (assignments == unique_groups[0]).astype(int),
            "feature_type": "categorical",
        },
        {
            "name": "Group2",
            "values": (assignments == unique_groups[1]).astype(int),
            "feature_type": "categorical",
        },
    ]
    for key, value in covariates.items():
        features.append({"name": key, "values": value, "feature_type": "continuous"})

    contrasts = [{"name": "GroupDiff", "values": [1, -1] + [0] * len(covariates)}]

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
    group_diff = perm.copes
    return group_diff, pvalues


def paired_diff_max_stat_perm(data, n_perm, metric="copes", n_jobs=1):
    """
    Statistical significance testing for paired difference.

    This function fits a General Linear Model (GLM) with ordinary least squares
    and performs a sign flip permutations test with the maximum statistic to
    determine a p-value for paired differences.

    Parameters
    ----------
    data : np.ndarray
        Data array for baseline corrected evoked responses.
        Shape is (n_samples, *target_dims).
    n_perm : int
        Number of permutations.
    metric : str, optional
        Metric to compute p-values. Options are 'copes' and 'tstats'.
        Default is 'copes'.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is 1.

    Returns
    -------
    paired_diff : np.ndarray
        Paired differences. Shape is (*target_dims,).
    pvalues : np.ndarray
        P-values. Shape is (*target_dims,).
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
