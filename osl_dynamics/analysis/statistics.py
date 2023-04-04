"""Functions to perform statistical significance testing.

"""

import logging

import numpy as np
from scipy import stats

import glmtools as glm

_logger = logging.getLogger("osl-dynamics")


def evoked_response_max_stat_perm(data, n_perm, covariates={}, n_jobs=1):
    """Statistical significant testing for evoked responses.

    This function fits a General Linear Model (GLM) with ordinary least squares
    and performs a sign flip permutations test with the maximum statistic to
    determine a p-value for evoked responses.

    Parameters
    ----------
    data : np.ndarray
        Baseline corrected evoked responses. This will be the target data for the GLM.
        Must be shape (n_subjects, n_samples, n_modes).
    n_perm : int
        Number of permutations.
    covariates : dict
        Covariates (extra regressors) to add to the GLM fit. These will be z-transformed.
    n_jobs : int
        Number of processes to run in parallel.

    Returns
    -------
    pvalues : np.ndarray
        P-values for the evoked response. Shape is (n_subjects, n_samples, n_modes).
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array.")
    if data.ndim != 3:
        raise ValueError("data must be (n_subjects, n_samples, n_modes).")

    # Create GLM Dataset
    data = glm.data.TrialGLMData(
        data=data,
        **covariates,
        dim_labels=["subjects", "samples", "modes"],
    )

    # Create design matrix
    DC = glm.design.DesignConfig()
    for name in covariates:
        DC.add_regressor(name=name, rtype="Parametric", datainfo=name, preproc="z")
    DC.add_regressor(name="Mean", rtype="Constant")
    DC.add_contrast(name="Mean", values=[1] + [0] * len(covariates))
    design = DC.design_from_datainfo(data.info)

    # Fit model and get t-statistics
    model = glm.fit.OLSModel(design, data)
    tstats = abs(model.tstats[0])

    # Run permutations and get null distribution
    perm = glm.permutations.MaxStatPermutation(
        design,
        data,
        contrast_idx=0,  # selects the Mean contrast
        nperms=n_perm,
        metric="tstats",
        tail=0,  # two-sided test
        pooled_dims=(1, 2),  # pool over samples and modes dimension
        nprocesses=n_jobs,
    )
    null_dist = perm.nulls

    # Get p-values
    percentiles = stats.percentileofscore(null_dist, tstats)
    pvalues = 1 - percentiles / 100

    if np.all(pvalues < 0.05):
        _logger.warn(
            "All time points for all modes are significant with p-value<0.05. "
            + "Did you remember to baseline correct the evoked responses?"
        )

    return pvalues


def group_diff_max_stat_perm(data, assignments, n_perm, covariates={}, n_jobs=1):
    """Statistical significant testing for the difference between two groups.

    This function fits a General Linear Model (GLM) with ordinary least squares
    and performs a row shuffle permutations test with the maximum statistic to
    determine a p-value for differences between two groups.

    Parameters
    ----------
    data : np.ndarray
        Baseline corrected evoked responses. This will be the target data for the GLM.
        Must be shape (n_subjects, features1, features2, ...).
    assignments : np.ndarray
        1D numpy array containing group assignments. A value of 1 indicates
        Group1 and a value of 2 indicates Group2. Note, we test the contrast
        abs(Group1 - Group2) > 0.
    n_perm : int
        Number of permutations.
    covariates : dict
        Covariates (extra regressors) to add to the GLM fit. These will be z-transformed.
    n_jobs : int
        Number of processes to run in parallel.

    Returns
    -------
    group_diff : np.ndarray
        Group difference: Group1 - Group2. Shape is (features1, features2, ...).
    pvalues : np.ndarray
        P-values for the features. Shape is (features1, features2, ...).
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array.")
    ndim = data.ndim
    if ndim == 1:
        raise ValueError("data must be 2D or greater.")

    # Calculate group difference
    group1_mean = np.mean(data[assignments == 1], axis=0)
    group2_mean = np.mean(data[assignments == 2], axis=0)
    group_diff = group1_mean - group2_mean

    # Create GLM Dataset
    data = glm.data.TrialGLMData(
        data=data,
        **covariates,
        category_list=assignments,
        dim_labels=["subjects"] + [f"features {i}" for i in range(1, ndim)],
    )

    # Create design matrix
    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Group1", rtype="Categorical", codes=1)
    DC.add_regressor(name="Group2", rtype="Categorical", codes=2)
    for name in covariates:
        DC.add_regressor(name=name, rtype="Parametric", datainfo=name, preproc="z")
    DC.add_contrast(name="GroupDiff", values=[1, -1] + [0] * len(covariates))
    design = DC.design_from_datainfo(data.info)

    # Fit model and get t-statistics
    model = glm.fit.OLSModel(design, data)
    tstats = abs(model.tstats[0])

    # Which dimensions are we pooling over?
    if ndim == 2:
        pooled_dims = 1
    else:
        pooled_dims = tuple(range(1, ndim))

    # Run permutations and get null distribution
    perm = glm.permutations.MaxStatPermutation(
        design,
        data,
        contrast_idx=0,  # selects GroupDiff
        nperms=n_perm,
        metric="tstats",
        tail=0,  # two-sided test
        pooled_dims=pooled_dims,
        nprocesses=n_jobs,
    )
    null_dist = perm.nulls

    # Get p-values
    percentiles = stats.percentileofscore(null_dist, tstats)
    pvalues = 1 - percentiles / 100

    return group_diff, pvalues
