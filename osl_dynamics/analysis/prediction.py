"""
This module provides classes for building machine learning pipelines,
performing hyperparameter tuning, and evaluating model performance.
"""

import numpy as np
from tqdm.auto import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    KFold,
    StratifiedKFold,
)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class PipelineBuilder:
    """
    A class to handle the design of machine learning pipelines with options for
    scaling, dimensionality reduction, and model selection.
    """

    DEFAULT_SCALER = None
    DEFAULT_DIM_REDUCTION = None
    DEFAULT_PREDICTOR = "ols"

    def __init__(self):
        """
        Initializes the Design object with predefined dictionaries of available
        scalers, dimensionality reduction techniques, and models.
        """
        self.scaler_dict = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
        }
        self.dim_reduction_dict = {
            "pca": PCA(),
            "ica": FastICA(),
        }
        self.predictor_dict = {
            "ols": LinearRegression(),
            "ridge": Ridge(),
            "lasso": Lasso(),
            "elastic_net": ElasticNet(),
            "svm": SVC(),
            "knn": KNeighborsClassifier(),
            "random_forest": RandomForestClassifier(),
            "gradient_boosted": GradientBoostingClassifier(),
            "logistic_regression": LogisticRegression(),
        }

    @property
    def available_scalers(self):
        return list(self.scaler_dict.keys())

    @property
    def available_dim_reductions(self):
        return list(self.dim_reduction_dict.keys())

    @property
    def available_predictors(self):
        return list(self.predictor_dict.keys())

    def _check_model(self, model_type, model_dict, model_name):
        """
        Checks if the provided model type exists in the given model dictionary.

        Parameters
        ----------
        model_type : str or None
            The type of the model to check.
        model_dict : dict
            The dictionary of available models.
        model_name : str
            The name of the model (for error messaging purposes).

        Raises
        ------
        ValueError
            If the model type is not found in the model dictionary.
        """
        if model_type is not None and model_type not in model_dict:
            raise ValueError(
                f"Invalid {model_name} type. Must be one of: {list(model_dict.keys())}"
            )

    def validate_model(self, scaler=None, dim_reduction=None, predictor=None):
        """
        Validates the provided model components (scaler, dimensionality reduction, and predictor).

        Parameters
        ----------
        scaler : str or None, optional
            The scaler name to use. If None, the default scaler is used.
        dim_reduction : str or None, optional
            The dimensionality reduction technique to use. If None, no reduction is applied.
        predictor : str or None, optional
            The model name to use. If None, the default predictor is used.

        Raises
        ------
        ValueError
            If any of the provided models or techniques are invalid.
        """
        self._check_model(scaler, self.scaler_dict, "scaler")
        self._check_model(dim_reduction, self.dim_reduction_dict, "dim_reduction")
        self._check_model(predictor, self.predictor_dict, "predictor")

    def build_model(self, scaler=None, dim_reduction=None, predictor=None):
        """
        Constructs and returns a scikit-learn pipeline with the specified components.

        Parameters
        ----------
        scaler : str or None, optional
            The scaler name to use. If None, the default scaler is used.
        dim_reduction : str or None, optional
            The dimensionality reduction technique to use. If None, no reduction is applied.
        predictor : str or None, optional
            The model name to use. If None, the default predictor is used.

        Returns
        -------
        Pipeline
            A scikit-learn Pipeline object with the specified components.
        """
        scaler = scaler or self.DEFAULT_SCALER
        dim_reduction = dim_reduction or self.DEFAULT_DIM_REDUCTION
        predictor = predictor or self.DEFAULT_PREDICTOR
        self.validate_model(scaler, dim_reduction, predictor)

        steps = []
        if scaler:
            steps.append(("scaler", self.scaler_dict[scaler]))

        if dim_reduction:
            steps.append(("dim_reduction", self.dim_reduction_dict[dim_reduction]))

        steps.append(("predictor", self.predictor_dict[predictor]))

        return Pipeline(steps)

    def get_params_grid(
        self, scalar_params=None, dim_reduction_params=None, predictor_params=None
    ):
        """
        Returns a combined parameter grid for use in hyperparameter optimization (e.g., GridSearchCV).

        Parameters
        ----------
        scalar_params : dict, optional
            A dictionary of parameters to be passed to the scaler.
        dim_reduction_params : dict, optional
            A dictionary of parameters to be passed to the dimensionality reduction technique.
        predictor_params : dict, optional
            A dictionary of parameters to be passed to the model.

        Returns
        -------
        dict
            A dictionary combining the parameters for all the components in the pipeline.
        """
        params = {}

        if scalar_params:
            params.update({f"scaler__{k}": v for k, v in scalar_params.items()})

        if dim_reduction_params:
            params.update(
                {f"dim_reduction__{k}": v for k, v in dim_reduction_params.items()}
            )

        if predictor_params:
            params.update({f"predictor__{k}": v for k, v in predictor_params.items()})

        return params


class ModelSelection:
    def __init__(
        self,
        model,
        params_grid=None,
        search_type="grid",
        cv=5,
        scoring=None,
        n_iter=10,
        random_state=None,
        n_jobs=1,
        verbose=0,
    ):
        """
        Initializes the pipeline with model selection and cross-validation settings.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            The machine learning model to be used.
        params_grid : dict, optional
            The hyperparameter grid for tuning.
        search_type : str, default='grid'
            The type of search: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
        cv : int, optional
            Number of cross-validation folds. Defaults to 5.
        scoring : str, optional
            Scoring metric to optimize. Defaults to None.
        n_iter : int, optional
            Number of iterations for RandomizedSearchCV. Defaults to 10.
        random_state : int, optional
            Random seed for reproducibility.
        n_jobs : int, optional
            Number of CPU cores to use for parallel processing. Defaults to 1.
        verbose : int, optional
            Verbosity level for model selection methods. Defaults to 0.
        """
        self.model = model
        self.params_grid = params_grid
        self.search_type = search_type
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.best_model = None
        self.best_params = None

    def set_params_grid(self, params_grid):
        """
        Sets the hyperparameter grid for tuning.

        Parameters
        ----------
        params_grid : dict
            The hyperparameter grid for tuning.
        """
        if not isinstance(params_grid, dict):
            raise ValueError("params_grid must be a dictionary.")
        self.params_grid = params_grid

    def set_cv(self, cv):
        """Sets the number of cross-validation folds.

        Parameters
        ----------
        cv : int
            Number of cross-validation folds.
        """
        if not isinstance(cv, int) or cv < 2:
            raise ValueError("cv must be an integer greater than or equal to 2.")
        self.cv = cv

    def set_scoring(self, scoring):
        """
        Sets the scoring metric.

        Parameters
        ----------
        scoring : str
            The scoring metric to use.
        """
        if scoring is not None and not isinstance(scoring, str):
            raise ValueError("scoring must be a string representing a valid metric.")
        self.scoring = scoring

    def set_n_iter(self, n_iter):
        """
        Sets the number of iterations for RandomizedSearchCV.

        Parameters
        ----------
        n_iter : int
            Number of iterations (must be positive integer).
        """
        if not isinstance(n_iter, int) or n_iter < 1:
            raise ValueError("n_iter must be a positive integer.")
        self.n_iter = n_iter

    def validate_data(self, X, y):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays or lists.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

    def model_selection(self, X, y, override_best_model=True):
        """
        Performs hyperparameter tuning using cross-validation.

        Parameters
        ----------
        X : array-like
            Feature matrix of shape (n_samples, n_features).
        y : array-like
            Target variable of shape (n_samples,).

        Returns
        -------
        best_params_ : dict
            Best hyperparameters found.
        best_score_ : float
            Best cross-validation score achieved.
        """
        self.validate_data(X, y)

        if self.params_grid is None:
            raise ValueError("params_grid must be provided for model selection.")

        if self.search_type == "grid":
            search = GridSearchCV(
                self.model,
                self.params_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        elif self.search_type == "random":
            search = RandomizedSearchCV(
                self.model,
                self.params_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_iter=self.n_iter,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        else:
            raise ValueError("Invalid search type. Must be 'grid' or 'random'.")

        search.fit(X, y)
        if override_best_model:
            self.best_model = search.best_estimator_
            self.best_params = search.best_params_

        return search.best_estimator_

    def nested_cross_validation(
        self, X, y, split_type="kfold", outer_cv=5, shuffle=True
    ):
        """
        Performs nested cross-validation to evaluate model performance.

        Parameters
        ----------
        X : array-like
            Feature matrix of shape (n_samples, n_features).
        y : array-like
            Target variable of shape (n_samples,).
        split_type : str, optional
            Type of cross-validation split to use. Must be 'kfold' or 'stratified_kfold'. Defaults to 'kfold'.
        outer_cv : int, optional
            Number of outer cross-validation folds. Defaults to 5.
        shuffle : bool, optional
            Whether to shuffle the data before splitting. Defaults to True.

        Returns
        -------
        outer_scores : np.ndarray
            Array of test scores for each outer fold.
        """
        self.validate_data(X, y)

        if split_type == "kfold":
            outer_split = KFold(
                n_splits=outer_cv, shuffle=shuffle, random_state=self.random_state
            )
        elif split_type == "stratified_kfold":
            outer_split = StratifiedKFold(
                n_splits=outer_cv, shuffle=shuffle, random_state=self.random_state
            )
        else:
            raise ValueError(
                "Invalid split type. Must be 'kfold' or 'stratified_kfold'."
            )

        outer_scores = []
        for train_idx, test_idx in tqdm(
            outer_split.split(X, y), desc="Nested CV", total=outer_cv
        ):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            best_model = self.model_selection(
                X_train, y_train, override_best_model=False
            )
            test_score = best_model.score(X_test, y_test)
            outer_scores.append(test_score)

        return np.array(outer_scores)

    def cross_validation_scores(self, X, y, cv=None, scoring=None):
        """
        Computes cross-validation scores for the best model.

        Parameters
        ----------
        X : array-like
            Feature matrix of shape (n_samples, n_features).
        y : array-like
            Target variable of shape (n_samples,).
        cv : int, optional
            Number of cross-validation folds. Defaults to the instance's cv attribute.
        scoring : str, optional
            Scoring metric to use. Defaults to the instance's scoring attribute.

        Returns
        -------
        scores : np.ndarray
            Cross-validation scores.
        """
        self.validate_data(X, y)
        model = self.best_model or self.model
        cv = cv or self.cv
        scoring = scoring or self.scoring

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return scores
