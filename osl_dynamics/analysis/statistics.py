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
    for name, data in covariates.items():
        DC.add_regressor(name=name, rtype="Parameteric", datainfo=name, preproc="z")
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
        contrast_idx=0,
        nperms=n_perm,
        metric="tstats",
        pooled_dims=(1, 2),
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
