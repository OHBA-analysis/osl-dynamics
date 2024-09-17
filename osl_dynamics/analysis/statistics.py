"""Functions to perform statistical significance testing.

"""

import logging

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

import glmtools as glm

_logger = logging.getLogger("osl-dynamics")


def _validate_dimensions(X=None, y=None, contrasts=None):
    """
    Validate dimensions of input arrays.
    """
    # Check X dimensions
    if X is None:
        X_n_samples, X_n_features = None, None
    elif X.ndim == 2:
        X_n_samples, X_n_features = X.shape
    else:
        raise ValueError(f"X must be 2D, got {X.ndim}D")

    # Check y dimensions
    if y is None:
        y_n_samples = None
    if y.ndim == 1:
        # Add target dimension
        y = y[:, None]
    elif y.ndim == 2:
        y_n_samples = y.shape[0]
    else:
        raise ValueError(f"y must be 1D or 2D, got {y.ndim}D")

    # Check contrasts dimensions
    if contrasts is None:
        contrasts_n_features = None
    if contrasts.ndim == 1:
        # Add contrast dimension
        contrasts = contrasts[None, :]
    elif contrasts.ndim == 2:
        contrasts_n_features = contrasts.shape[1]
    else:
        raise ValueError(f"contrasts must be 1D or 2D, got {contrasts.ndim}D")

    # Validate dimensions
    if (
        X_n_samples is not None
        and y_n_samples is not None
        and X_n_samples != y_n_samples
    ):
        raise ValueError(
            f"X and y must have the same number of samples. Got {X_n_samples} samples in X and {y_n_samples} samples in y."
        )
    if (
        X_n_features is not None
        and contrasts_n_features is not None
        and X_n_features != contrasts_n_features
    ):
        raise ValueError(
            f"X and contrasts must have the same number of features. Got {X_n_features} features in X and {contrasts_n_features} features in contrasts."
        )

    return X, y, contrasts


def get_residuals(X, y, predictor):
    """
    Get residuals from a linear model.

    Parameters
    ----------
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).
    y : np.ndarray
        Target variable. Shape is (n_samples, ) or (n_samples, n_targets).
    predictor : sklearn.linear_model.LinearRegression
        Sklearn LinearRegression object.

    Returns
    -------
    residuals : np.ndarray
        Residuals. Shape is (n_samples, n_targets).
    """
    X, y, _ = _validate_dimensions(X=X, y=y)
    if not isinstance(predictor, LinearRegression):
        raise ValueError(
            f"predictor must be a LinearRegression object, got {type(predictor)}"
        )
    return y - predictor.predict(X)


def get_degree_of_freedom(X):
    """
    Get degree of freedom.

    Parameters
    ----------
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).

    Returns
    -------
    dof : int
        Degree of freedom.
    """
    X, _, _ = _validate_dimensions(X=X)
    return X.shape[0] - np.linalg.matrix_rank(X)


def get_varcopes(X, y, contrasts, predictor):
    """
    Get the variance of the copes.

    Parameters
    ----------
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).
    y : np.ndarray
        Target variable. Shape is (n_samples, ) or (n_samples, n_targets).
    contrasts : np.ndarray
        Contrasts matrix. Shape is (n_features, ) or (n_contrasts, n_features).
    predictor : sklearn.linear_model.LinearRegression
        Sklearn LinearRegression object.

    Returns
    -------
    varcopes : np.ndarray
        Variance of the copes. Shape is (n_contrasts, n_targets).
    """
    X, y, contrasts = _validate_dimensions(X=X, y=y, contrasts=contrasts)
    if not isinstance(predictor, LinearRegression):
        raise ValueError(
            f"predictor must be a LinearRegression object, got {type(predictor)}"
        )

    xxt = X.T @ X
    xxt_inv = np.linalg.inv(xxt)
    c_xxt_inv_ct = np.diag(contrasts @ xxt_inv @ contrasts.T)  # Shape is (n_contrasts,)

    # Get estimate of standard error
    residuals = get_residuals(X, y, predictor)
    dof = get_degree_of_freedom(X)
    s2 = np.sum(residuals**2, axis=0) / dof  # Shape is (n_targets,)

    varcopes = c_xxt_inv_ct[:, None] * s2[None, :]  # Shape is (n_contrasts, n_targets)
    return varcopes


def osl_fit(X, y, contrasts):
    """
    Fit Ordinary Least Squares (OLS) model.

    Parameters
    ----------
    X : np.ndarray
        Design matrix. Shape is (n_samples, n_features).
    y : np.ndarray
        Target variable. Shape is (n_samples, ) or (n_samples, n_targets).
    contrasts : np.ndarray
        Contrasts matrix. Shape is (n_features, ) or (n_contrasts, n_features).

    Returns
    -------
    betas : np.ndarray
        Betas (regression coefficients). Shape is (n_features, n_targets).
    copes : np.ndarray
        Contrast parameter estimates. Shape is (n_contrasts, n_targets).
    varcopes : np.ndarray
        Variance of the copes. Shape is (n_contrasts, n_targets).
    """
    X, y, contrasts = _validate_dimensions(X=X, y=y, contrasts=contrasts)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)
    betas = lr.coef_
    copes = contrasts @ betas
    varcopes = get_varcopes(X, y, contrasts, lr)

    return betas, copes, varcopes


class Feature:
    def __init__(self, name, values, feature_type="continuous"):
        self.name = self._validate_name(name)
        self.values = self._validate_values(values)
        self.feature_type = self._validate_type(feature_type)

    def _validate_name(self, name):
        if not isinstance(name, str):
            raise ValueError(f"name must be a string, got {type(name)}")
        return name

    def _validate_values(self, values):
        if isinstance(values, list):
            values = np.array(values)

        if not isinstance(values, np.ndarray):
            raise ValueError(f"values must be a list or np.ndarray, got {type(values)}")

        if values.ndim != 1:
            raise ValueError(f"values must be 1D, got {values.ndim}D")

        self.n_samples = len(values)
        return values

    def _validate_type(self, feature_type):
        if feature_type not in ["continuous", "categorical"]:
            raise ValueError(
                f"type must be 'continuous' or 'categorical', got {feature_type}"
            )
        return feature_type

    def get_name(self):
        return self.name

    def get_n_samples(self):
        return self.n_samples

    def get_values(self):
        return self.values

    def get_feature_type(self):
        return self.feature_type

    def summary(self):
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "feature_type": self.feature_type,
        }


class Covariates:
    def __init__(self):
        self.covariates = []
        self.n_covariates = 0
        self.n_samples = None

    def add_covariates(self, covariates, covariate_types=None):
        covariates, covariate_types = self._validate_inputs(covariates, covariate_types)
        for name, values, feature_type in zip(
            covariates.keys(), covariates.values(), covariate_types
        ):
            feature = Feature(name, values, feature_type)
            self.covariates.append(feature)
            self.n_covariates += 1

    def get_covariates(self):
        return self.covariates

    def get_covariate_names(self):
        return [feature.get_name() for feature in self.covariates]

    def get_covariate_types(self):
        return [feature.get_feature_type() for feature in self.covariates]

    def _validate_inputs(self, covariates, covariate_types):
        if not isinstance(covariates, dict):
            raise ValueError(f"covariates must be a dictionary, got {type(covariates)}")

        n_covariates = len(covariates)
        if covariate_types is None:
            covariate_types = ["continuous"] * n_covariates
        else:
            if not isinstance(covariate_types, list):
                raise ValueError(
                    f"covariate_types must be a list, got {type(covariate_types)}"
                )
            if len(covariate_types) != n_covariates:
                raise ValueError(
                    f"Number of covariate_types ({len(covariate_types)}) must match number of covariates ({n_covariates})."
                )

        for name, values in covariates.items():
            n_samples = len(values)
            if self.n_samples is None:
                self.n_samples = n_samples
            elif n_samples != self.n_samples:
                raise ValueError(
                    f"Number of samples in {name} ({n_samples}) must match number of samples in the first covariate ({self.n_samples})."
                )

        return covariates, covariate_types


class OLS:
    """
    Ordinary Least Squares (OLS) analysis.

    Parameters
    ----------
    contrasts : np.ndarray
        Contrasts matrix. Shape is (n_features, ) or (n_contrasts, n_features).
    include_mean : bool, optional
        Whether to include a mean in the design matrix. Default is True.
    covariates : Covariates or dict, optional
        Covariates. Keys are the names of the covariates and values are 1D array-like.
        Default is no covariates.
    """

    def __init__(self, contrasts, include_mean=True, covariates=None):
        self.n_targets = None
        _, _, self.contrasts = _validate_dimensions(contrasts=contrasts)
        self.n_contrasts, self.n_features = contrasts.shape

        self.include_mean = include_mean
        self.covariates = self._validate_covariates(covariates)
        self.n_samples = self.covariates.n_samples

        if self._get_X_n_features() != self.n_features:
            raise ValueError(
                f"Number of features in X ({self._get_X_n_features()}) must match number of features in contrasts ({self.n_features})."
            )

    def _validate_covariates(self, covariates):
        if covariates is None:
            covariates = {}
            return covariates

        if isinstance(covariates, dict):
            _covariates = Covariates()
            _covariates.add_covariates(covariates)
            return _covariates

        if isinstance(covariates, Covariates):
            return covariates

        raise ValueError(
            f"covariates must be a dictionary or Covariates object, got {type(covariates)}"
        )

    def _get_X_n_features(self):
        n_features = self.covariates.n_covariates
        if self.include_mean:
            n_features += 1
        return n_features

    def _validate_y(self, y):
        # TODO: Deal with more than 2D y
        y = np.array(y)
        _, y, _ = _validate_dimensions(y=y)
        if self.n_samples is None:
            self.n_samples = y.shape[0]

        elif y.shape[0] != self.n_samples:
            raise ValueError(
                f"Number of samples in y ({y.shape[0]}) must match number of samples in covariates ({self.n_samples})."
            )

        self.n_targets = y.shape[1]
        return y

    def fit(self, y, standardize_features=True):
        # Validate y
        self._validate_y(y)
        X = self.build_X()
        if standardize_features:
            X = self.standardize_features(X)

        betas, copes, varcopes = osl_fit(X, y, self.contrasts)
        self.betas = betas
        self.copes = copes
        self.varcopes = varcopes
        self.tstats = self.get_tstats(copes, varcopes)

    def build_X(self):
        """
        Build design matrix X.

        Returns
        -------
        X : np.ndarray
            Design matrix. Shape is (n_samples, n_features).
        """
        # Validate
        if self.n_samples is None:
            raise ValueError(
                "Number of samples is not defined. This happens when 'self.fit' has not been called and empty covariates."
            )

        # Initialize X
        X = np.zeros((self.n_samples, self.n_features))

        if self.include_mean:
            X[:, 0] = 1

        for i, feature in enumerate(self.covariates.get_covariates()):
            X[:, i + 1] = feature.get_values()

        return X

    def get_feature_types(self):
        feature_types = self.covariates.get_covariate_types()
        if self.include_mean:
            feature_types = ["categorical"] + feature_types
        return feature_types

    def standardize_features(self, X):
        """
        Standardize features.

        Parameters
        ----------
        X : np.ndarray
            Design matrix. Shape is (n_samples, n_features).

        Returns
        -------
        X : np.ndarray
            Standardized design matrix. Shape is (n_samples, n_features).
        """
        # Only standardize continuous features
        cts_indx = np.where(np.array(self.get_feature_types()) == "continuous")[0]
        X_mean = np.mean(X[:, cts_indx], axis=0)
        X_std = np.std(X[:, cts_indx], axis=0)
        X[:, cts_indx] = (X[:, cts_indx] - X_mean) / X_std
        return X

    def get_tstats(self, copes, varcopes):
        """
        Get t-statistics.

        Parameters
        ----------
        copes : np.ndarray
            Contrast parameter estimates. Shape is (n_contrasts, n_targets).
        varcopes : np.ndarray
            Variance of the copes. Shape is (n_contrasts, n_targets).

        Returns
        -------
        tstats : np.ndarray
            T-statistics. Shape is (n_contrasts, n_targets).
        """
        # TODO: might need to deal with division by zero and NaNs
        return copes / np.sqrt(varcopes)

    def get_feature_names(self):
        """
        Get the names of the features.
        """
        names = list(self.covariates.keys())
        if self.include_mean:
            names = ["Mean"] + names
        return names

    @property
    def design_matrix(self):
        """
        Design matrix X.
        """
        return self.build_X()

    @property
    def summary(self):
        """
        Summary of the OSL analysis.
        """
        return {
            "n_samples": self.n_samples,
            "n_targets": self.n_targets,
            "n_contrasts": self.n_contrasts,
            "n_features": self.n_features,
            "features": self.get_feature_names(),
            "betas": self.betas,
            "copes": self.copes,
            "varcopes": self.varcopes,
            "tstats": self.tstats,
        }


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
