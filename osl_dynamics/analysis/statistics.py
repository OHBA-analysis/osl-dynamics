"""Functions to perform statistical significance testing.

"""

import logging

import numpy as np
from scipy import stats

import glmtools as glm

_logger = logging.getLogger("osl-dynamics")


def _check_glm_data(data, covariates, assignments=None):
    """Check shapes and remove subjects from GLM data which contain nans."""

    # Make sure the number of subjects in data and covariates match
    n_subjects = data.shape[0]
    for k, v in covariates.items():
        if v.shape[0] != n_subjects:
            raise ValueError(
                f"Got covariates['{k}'].shape[0]={v.shape[0]}, "
                + f"but was expecting {n_subjects}."
            )

    # Convert covariates to a numpy array
    if len(covariates) > 0:
        covariates_data = np.array(list(covariates.values())).T

    # Remove subjects with a nan in either array
    remove = []
    for i in range(n_subjects):
        if np.isnan(data[i]).any():
            remove.append(i)
        if len(covariates) > 0:
            if np.isnan(covariates_data[i]).any():
                remove.append(i)
        if assignments is not None:
            if np.isnan(assignments[i]):
                remove.append(i)
    remove = np.unique(remove)
    if len(remove) > 0:
        _logger.warn(f"The following subjects were removed from the GLM: {remove}")

    # Keep subjects without nans
    keep = [i for i in range(n_subjects) if i not in remove]
    data = np.copy(data)[keep]
    covariates_ = {}
    for key in covariates:
        covariates_[key] = covariates[key][keep]
    if assignments is not None:
        assignments = np.copy(assignments)[keep]

    # Check we have some subjects left
    if len(data) == 0:
        raise ValueError("No valid data to calculate the GLM.")

    if assignments is not None:
        return data, covariates_, assignments
    else:
        return data, covariates_


def evoked_response_max_stat_perm(
    data, n_perm, covariates=None, metric="copes", n_jobs=1
):
    """Statistical significant testing for evoked responses.

    This function fits a General Linear Model (GLM) with ordinary least squares
    and performs a sign flip permutations test with the maximum statistic to
    determine a p-value for evoked responses.

    Parameters
    ----------
    data : np.ndarray
        Baseline corrected evoked responses. This will be the target data for
        the GLM. Must be shape (n_subjects, n_samples, ...).
    n_perm : int
        Number of permutations.
    covariates : dict, optional
        Covariates (extra regressors) to add to the GLM fit. These will be
        z-transformed. Must a dict with values of shape (n_subjects,).
    metric : str, optional
        Metric to use to build the null distribution. Can be :code:`'tstats'` or
        :code:`'copes'`.
    n_jobs : int, optional
        Number of processes to run in parallel.

    Returns
    -------
    pvalues : np.ndarray
        P-values for the evoked response. Shape is (n_subjects, n_samples, ...).
    """
    if covariates is None:
        covariates = {}

    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array.")

    ndim = data.ndim
    if ndim < 3:
        raise ValueError("data must be 3D or greater.")

    if metric not in ["tstats", "copes"]:
        raise ValueError("metric must be 'tstats' or 'copes'.")

    data, covariates = _check_glm_data(data, covariates)

    # Create GLM Dataset
    data = glm.data.TrialGLMData(
        data=data,
        **covariates,
        dim_labels=["subjects", "time"] + [f"features {i}" for i in range(1, ndim - 1)],
    )

    # Create design matrix
    DC = glm.design.DesignConfig()
    for name in covariates:
        DC.add_regressor(
            name=name,
            rtype="Parametric",
            datainfo=name,
            preproc="z",
        )
    DC.add_regressor(name="Mean", rtype="Constant")
    DC.add_contrast(name="Mean", values=[1] + [0] * len(covariates))
    design = DC.design_from_datainfo(data.info)

    # Fit model and get t-statistics
    model = glm.fit.OLSModel(design, data)

    # Pool over all dimensions over than subjects
    pooled_dims = tuple(range(1, ndim))

    # Run permutations and get null distribution
    perm = glm.permutations.MaxStatPermutation(
        design,
        data,
        contrast_idx=0,  # selects the Mean contrast
        nperms=n_perm,
        metric=metric,
        tail=0,  # two-sided test
        pooled_dims=pooled_dims,
        nprocesses=n_jobs,
    )
    null_dist = perm.nulls

    # Get p-values
    if metric == "tstats":
        print("Using tstats as metric")
        tstats = abs(model.tstats[0])
        percentiles = stats.percentileofscore(null_dist, tstats)
    elif metric == "copes":
        print("Using copes as metric")
        copes = abs(model.copes[0])
        percentiles = stats.percentileofscore(null_dist, copes)
    pvalues = 1 - percentiles / 100

    if np.all(pvalues < 0.05):
        _logger.warn(
            "All time points for all modes are significant with p-value<0.05. "
            + "Did you remember to baseline correct the evoked responses?"
        )

    return pvalues


def group_diff_max_stat_perm(
    data, assignments, n_perm, covariates=None, metric="tstats", n_jobs=1
):
    """Statistical significant testing for the difference between two groups.

    This function fits a General Linear Model (GLM) with ordinary least squares
    and performs a row shuffle permutations test with the maximum statistic to
    determine a p-value for differences between two groups.

    Parameters
    ----------
    data : np.ndarray
        Subject-specific quantities to compare. This will be the target data
        for the GLM. Must be shape (n_subjects, features1, features2, ...).
    assignments : np.ndarray
        1D numpy array containing group assignments. A value of 1 indicates
        Group1 and a value of 2 indicates Group2. Note, we test the contrast
        :code:`abs(Group1 - Group2) > 0`.
    n_perm : int
        Number of permutations.
    covariates : dict, optional
        Covariates (extra regressors) to add to the GLM fit. These will be
        z-transformed. Must be a dict with values of shape (n_subjects,).
    metric : str, optional
        Metric to use to build the null distribution. Can be :code:`'tstats'` or
        :code:`'copes'`.
    n_jobs : int, optional
        Number of processes to run in parallel.

    Returns
    -------
    group_diff : np.ndarray
        Group difference: Group1 - Group2. Shape is (features1, features2, ...).
    pvalues : np.ndarray
        P-values for the features. Shape is (features1, features2, ...).
    """
    if covariates is None:
        covariates = {}

    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array.")

    ndim = data.ndim
    if ndim == 1:
        raise ValueError("data must be 2D or greater.")

    if metric not in ["tstats", "copes"]:
        raise ValueError("metric must be 'tstats' or 'copes'.")

    data, covariates, assignments = _check_glm_data(
        data,
        covariates,
        assignments,
    )

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
        DC.add_regressor(
            name=name,
            rtype="Parametric",
            datainfo=name,
            preproc="z",
        )
    DC.add_contrast(name="GroupDiff", values=[1, -1] + [0] * len(covariates))
    design = DC.design_from_datainfo(data.info)

    # Fit model and get t-statistics
    model = glm.fit.OLSModel(design, data)

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
        metric=metric,
        tail=0,  # two-sided test
        pooled_dims=pooled_dims,
        nprocesses=n_jobs,
    )
    null_dist = perm.nulls

    # Get p-values
    if metric == "tstats":
        print("Using tstats as metric")
        tstats = abs(model.tstats[0])
        percentiles = stats.percentileofscore(null_dist, tstats)
    elif metric == "copes":
        print("Using copes as metric")
        copes = abs(model.copes[0])
        percentiles = stats.percentileofscore(null_dist, copes)
    pvalues = 1 - percentiles / 100

    # Get group differences
    group_diff = model.copes[0]

    return group_diff, pvalues


def paired_diff_max_stat_perm(data, n_perm, metric="tstats", n_jobs=1):
    """Statistical significant testing for paired differences.

    This function fits a General Linear Model (GLM) with ordinary least squares
    and performs a sign flip permutations test with the maximum statistic to
    determine a p-value for paired differences.

    Parameters
    ----------
    data : np.ndarray
        Paired differences to compare. This will be the target data for the GLM.
        Must be shape (n_subjects, features1, features2, ...).
    n_perm : int
        Number of permutations.
    metric : str, optional
        Metric to use to build the null distribution. Can be :code:`'tstats'` or
        :code:`'copes'`.
    n_jobs : int, optional
        Number of processes to run in parallel.

    Returns
    -------
    paired_diff : np.ndarray
        Paired differences. Shape is (features1, features2, ...).
    pvalues : np.ndarray
        P-values for the features. Shape is (features1, features2, ...).
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array.")

    ndim = data.ndim
    if ndim == 1:
        raise ValueError("data must be 2D or greater.")

    if metric not in ["tstats", "copes"]:
        raise ValueError("metric must be 'tstats' or 'copes'.")

    # Create GLM Dataset
    data = glm.data.TrialGLMData(
        data=data,
        dim_labels=["subjects"] + [f"features {i}" for i in range(1, ndim)],
    )

    # Create design matrix
    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Mean", rtype="Constant")
    DC.add_contrast(name="Mean", values=[1])
    design = DC.design_from_datainfo(data.info)

    # Fit model and get t-statistics
    model = glm.fit.OLSModel(design, data)

    # Which dimensions are we pooling over?
    if ndim == 2:
        pooled_dims = 1
    else:
        pooled_dims = tuple(range(1, ndim))

    # Run permutations and get null distribution
    perm = glm.permutations.MaxStatPermutation(
        design,
        data,
        contrast_idx=0,
        nperms=n_perm,
        metric=metric,
        tail=0,  # two-sided test
        pooled_dims=pooled_dims,
        nprocesses=n_jobs,
    )
    null_dist = perm.nulls

    # Get p-values
    if metric == "tstats":
        print("Using tstats as metric")
        tstats = abs(model.tstats[0])
        percentiles = stats.percentileofscore(null_dist, tstats)
    elif metric == "copes":
        print("Using copes as metric")
        copes = abs(model.copes[0])
        percentiles = stats.percentileofscore(null_dist, copes)
    pvalues = 1 - percentiles / 100

    # Get paired differences
    paired_diff = model.copes[0]

    return paired_diff, pvalues
