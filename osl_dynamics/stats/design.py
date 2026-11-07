"""GLM design matrices and contrasts."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

FEATURE_TYPES = ("constant", "continuous", "categorical")


def _validate_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"name must be a string, got {type(name)}")
    return name


def _validate_values(values: Union[np.ndarray, list]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"values must be 1D, got {values.ndim}D")
    return values


def _validate_type(feature_type: str) -> str:
    if feature_type not in FEATURE_TYPES:
        raise ValueError(
            "feature_type must be 'constant', 'continuous' or 'categorical', "
            f"got {feature_type}"
        )
    return feature_type


class Feature:
    """A regressor (column) of the design matrix.

    Parameters
    ----------
    name : str
        Feature name.
    values : np.ndarray or list
        Feature values. Shape is (n_samples,).
    feature_type : str
        :code:`'constant'`, :code:`'continuous'` or :code:`'categorical'`.
        Categorical features are indicator (0/1) regressors. Continuous
        features are z-scored when the design matrix is built if the design
        has :code:`standardize_features=True`.
    """

    def __init__(
        self, name: str, values: Union[np.ndarray, list], feature_type: str
    ) -> None:
        self.name = _validate_name(name)
        self.values = _validate_values(values)
        self.feature_type = _validate_type(feature_type)
        self.n_samples = len(self.values)

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "feature_type": self.feature_type,
        }


class Contrast:
    """A t-contrast: a weighted sum of the regression coefficients.

    Parameters
    ----------
    name : str
        Contrast name.
    values : np.ndarray or list
        Contrast weights. Shape is (n_features,).
    """

    statistic = "t"

    def __init__(self, name: str, values: Union[np.ndarray, list]) -> None:
        self.name = _validate_name(name)
        self.values = _validate_values(values)
        self.n_features = len(self.values)
        self.contrast_type = self._get_contrast_type()

    def _get_contrast_type(self) -> str:
        if np.allclose(self.matrix.sum(axis=1), 0):
            return "differential"
        return "main_effect"

    @property
    def matrix(self) -> np.ndarray:
        """Contrast weights as a matrix. Shape is (n_rows, n_features)."""
        return self.values[None, :]

    @property
    def n_rows(self) -> int:
        """Number of rows (1 for a t-contrast)."""
        return self.matrix.shape[0]

    @property
    def feature_indices(self) -> np.ndarray:
        """Indices of the features with a non-zero weight."""
        return np.where(np.any(self.matrix != 0, axis=0))[0]

    @property
    def nuisance_indices(self) -> np.ndarray:
        """Indices of the features with a zero weight in every row."""
        return np.where(np.all(self.matrix == 0, axis=0))[0]

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "n_features": self.n_features,
            "contrast_type": self.contrast_type,
            "statistic": self.statistic,
        }


class FContrast(Contrast):
    """An F-contrast: a set of t-contrasts tested jointly.

    The F-test asks whether any linear combination of the rows differs from
    zero, e.g. whether any of three groups differ from each other, without
    specifying a direction. It is the GLM form of an ANOVA.

    Parameters
    ----------
    name : str
        Contrast name.
    values : np.ndarray or list of lists
        Contrast weights, one t-contrast per row. Shape is
        (n_rows, n_features). The rows must be linearly independent.
    """

    statistic = "F"

    def __init__(self, name: str, values: Union[np.ndarray, list]) -> None:
        self.name = _validate_name(name)
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            raise ValueError(
                f"values of an F-contrast must be 2D (n_rows, n_features), got "
                f"{values.ndim}D."
            )
        if np.linalg.matrix_rank(values) < values.shape[0]:
            raise ValueError(
                f"The rows of F-contrast {name} are not linearly independent."
            )
        self.values = values
        self.n_features = values.shape[1]
        self.contrast_type = self._get_contrast_type()

    @property
    def matrix(self) -> np.ndarray:
        """Contrast weights. Shape is (n_rows, n_features)."""
        return self.values

    def summary(self) -> Dict:
        summary = super().summary()
        summary["n_rows"] = self.n_rows
        return summary


class Design:
    """A GLM design: the regressors and the contrasts.

    Parameters
    ----------
    features : list of Feature, optional
        Regressors. Default is an empty list.
    contrasts : list of Contrast, optional
        t-contrasts. Default is an empty list.
    f_contrasts : list of FContrast, optional
        F-contrasts. Default is an empty list.
    standardize_features : bool, optional
        Whether to z-score continuous features. Default is :code:`True`,
        in which case contrast estimates for a continuous feature are per
        standard deviation of that feature.
    """

    def __init__(
        self,
        features: Optional[List[Feature]] = None,
        contrasts: Optional[List[Contrast]] = None,
        f_contrasts: Optional[List[FContrast]] = None,
        standardize_features: bool = True,
    ) -> None:
        self.features = [] if features is None else list(features)
        self.contrasts = [] if contrasts is None else list(contrasts)
        self.f_contrasts = [] if f_contrasts is None else list(f_contrasts)
        self.standardize_features = standardize_features

    def add_feature(
        self, name: str, values: Union[np.ndarray, list], feature_type: str
    ) -> None:
        """Add a feature to the design.

        Parameters
        ----------
        name : str
            Feature name.
        values : np.ndarray or list
            Feature values. Shape is (n_samples,).
        feature_type : str
            :code:`'constant'`, :code:`'continuous'` or :code:`'categorical'`.
        """
        self.features.append(Feature(name, values, feature_type))

    def add_contrast(self, name: str, values: Union[np.ndarray, list]) -> None:
        """Add a t-contrast to the design.

        Parameters
        ----------
        name : str
            Contrast name.
        values : np.ndarray or list
            Contrast weights. Shape is (n_features,).
        """
        self.contrasts.append(Contrast(name, values))

    def add_f_contrast(
        self,
        name: str,
        values: Optional[Union[np.ndarray, list]] = None,
        contrasts: Optional[Sequence[Union[str, int]]] = None,
    ) -> None:
        """Add an F-contrast to the design.

        Parameters
        ----------
        name : str
            Contrast name.
        values : np.ndarray or list of lists, optional
            Contrast weights, one t-contrast per row. Shape is
            (n_rows, n_features).
        contrasts : list of str or int, optional
            Alternatively, the names (or indices) of t-contrasts already in
            the design whose rows make up the F-contrast.
        """
        if (values is None) == (contrasts is None):
            raise ValueError("Pass exactly one of values or contrasts.")
        if contrasts is not None:
            if len(contrasts) == 0:
                raise ValueError("contrasts must name at least one t-contrast.")
            rows = []
            for contrast in contrasts:
                if isinstance(contrast, str):
                    if contrast not in self.contrast_names:
                        raise ValueError(f"No t-contrast named {contrast}.")
                    contrast = self.contrast_names.index(contrast)
                rows.append(self.contrasts[contrast].values)
            values = np.stack(rows)
        self.f_contrasts.append(FContrast(name, values))

    def validate(self) -> None:
        """Check the features and contrasts are consistent."""
        if len(self.features) == 0:
            raise ValueError("The design has no features.")
        if len(self.contrasts) == 0 and len(self.f_contrasts) == 0:
            raise ValueError("The design has no contrasts.")

        for feature in self.features:
            if not isinstance(feature, Feature):
                raise TypeError(f"Expected a Feature, got {type(feature)}.")
            if feature.n_samples != self.n_samples:
                raise ValueError(
                    "All features must have the same number of samples, got "
                    f"{feature.n_samples} in {feature.name} and {self.n_samples} "
                    f"in {self.features[0].name}."
                )

        for contrast in self.contrasts:
            if not isinstance(contrast, Contrast) or isinstance(contrast, FContrast):
                raise TypeError(f"Expected a Contrast, got {type(contrast)}.")
        for contrast in self.f_contrasts:
            if not isinstance(contrast, FContrast):
                raise TypeError(f"Expected an FContrast, got {type(contrast)}.")
        for contrast in self.contrasts + self.f_contrasts:
            if contrast.n_features != self.n_features:
                raise ValueError(
                    f"Contrast {contrast.name} has {contrast.n_features} weights "
                    f"but the design has {self.n_features} features."
                )

    def build_X(self) -> np.ndarray:
        """Build the design matrix.

        Returns
        -------
        X : np.ndarray
            Design matrix. Shape is (n_samples, n_features).
        """
        self.validate()
        X = np.stack([feature.values for feature in self.features], axis=1)
        if self.standardize_features:
            X = self._standardize_features(X)
        return X

    def build_contrast_array(self) -> np.ndarray:
        """Build the t-contrast matrix.

        Returns
        -------
        contrasts : np.ndarray
            Contrast matrix. Shape is (n_contrasts, n_features).
        """
        self.validate()
        if len(self.contrasts) == 0:
            return np.zeros((0, self.n_features))
        return np.stack([contrast.values for contrast in self.contrasts], axis=0)

    def build_f_contrast_arrays(self) -> List[np.ndarray]:
        """Build the F-contrast matrices.

        Returns
        -------
        f_contrasts : list of np.ndarray
            One matrix per F-contrast. Shape of each is (n_rows, n_features).
        """
        self.validate()
        return [contrast.matrix for contrast in self.f_contrasts]

    def _standardize_features(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        cts = np.array(self.feature_types) == "continuous"
        if cts.any():
            mean = np.nanmean(X[:, cts], axis=0)
            std = np.nanstd(X[:, cts], axis=0)
            if np.any(std == 0):
                names = np.array(self.feature_names)[cts][std == 0]
                raise ValueError(
                    f"Continuous feature(s) {list(names)} are constant and "
                    "cannot be standardized."
                )
            X[:, cts] = (X[:, cts] - mean) / std
        return X

    def contrast_feature_indices(self, contrast_index: int) -> np.ndarray:
        """Indices of the features with a non-zero weight in a t-contrast."""
        return self.contrasts[contrast_index].feature_indices

    def nuisance_feature_indices(self, contrast_index: int) -> np.ndarray:
        """Indices of the features with a zero weight in a t-contrast."""
        return self.contrasts[contrast_index].nuisance_indices

    @property
    def n_samples(self) -> int:
        return self.features[0].n_samples

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_contrasts(self) -> int:
        """Number of t-contrasts."""
        return len(self.contrasts)

    @property
    def n_f_contrasts(self) -> int:
        """Number of F-contrasts."""
        return len(self.f_contrasts)

    @property
    def feature_names(self) -> List[str]:
        return [f.name for f in self.features]

    @property
    def feature_types(self) -> List[str]:
        return [f.feature_type for f in self.features]

    @property
    def contrast_names(self) -> List[str]:
        return [c.name for c in self.contrasts]

    @property
    def contrast_types(self) -> List[str]:
        return [c.contrast_type for c in self.contrasts]

    @property
    def f_contrast_names(self) -> List[str]:
        return [c.name for c in self.f_contrasts]

    @property
    def dof(self) -> int:
        """Degrees of freedom of the design, for the samples without missing
        values in the design matrix."""
        X = self.build_X()
        X = X[~np.isnan(X).any(axis=1)]
        return X.shape[0] - np.linalg.matrix_rank(X)

    def summary(self) -> Dict:
        """Summary of the design."""
        summary = {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "n_contrasts": self.n_contrasts,
            "contrast_names": self.contrast_names,
            "contrast_types": self.contrast_types,
        }
        if self.n_f_contrasts > 0:
            summary["n_f_contrasts"] = self.n_f_contrasts
            summary["f_contrast_names"] = self.f_contrast_names
        summary["dof"] = self.dof
        return summary

    def plot(self, **kwargs):
        """Plot the design matrix and contrasts.

        Parameters
        ----------
        kwargs : keyword arguments, optional
            Passed to :func:`osl_dynamics.utils.plotting.plot_design_matrix`.
        """
        from osl_dynamics.utils.plotting import plot_design_matrix

        return plot_design_matrix(self, **kwargs)


@dataclass
class DesignConfig:
    """Build a :class:`Design` from dictionaries.

    Parameters
    ----------
    features : list of dict
        One dictionary per regressor with keys:

        - :code:`name` (str): feature name.
        - :code:`values` (np.ndarray or list): values, shape (n_samples,).
        - :code:`feature_type` (str): :code:`'constant'`,
          :code:`'continuous'` or :code:`'categorical'`.

    contrasts : list of dict, optional
        One dictionary per t-contrast with keys:

        - :code:`name` (str): contrast name.
        - :code:`values` (np.ndarray or list): weights, shape (n_features,).

    f_contrasts : list of dict, optional
        One dictionary per F-contrast with keys:

        - :code:`name` (str): contrast name.
        - :code:`values` (np.ndarray or list of lists): weights, one
          t-contrast per row, shape (n_rows, n_features); or
        - :code:`contrasts` (list of str or int): names or indices of the
          t-contrasts in :code:`contrasts` that make up the F-contrast.

        At least one t- or F-contrast is needed.
    standardize_features : bool, optional
        Whether to z-score continuous features. Default is :code:`True`.

    Examples
    --------
    Two groups with age as a nuisance covariate::

        config = DesignConfig(
            features=[
                {"name": "Group1", "values": group == 1, "feature_type": "categorical"},
                {"name": "Group2", "values": group == 2, "feature_type": "categorical"},
                {"name": "Age", "values": age, "feature_type": "continuous"},
            ],
            contrasts=[{"name": "Group1-Group2", "values": [1, -1, 0]}],
        )
        design = config.create_design()

    Three groups, testing for any difference between them (an F-test)::

        config = DesignConfig(
            features=[
                {"name": "Group1", "values": group == 1, "feature_type": "categorical"},
                {"name": "Group2", "values": group == 2, "feature_type": "categorical"},
                {"name": "Group3", "values": group == 3, "feature_type": "categorical"},
            ],
            f_contrasts=[{"name": "Groups", "values": [[1, -1, 0], [0, 1, -1]]}],
        )
    """

    features: List[Dict] = None
    contrasts: List[Dict] = None
    f_contrasts: List[Dict] = None
    standardize_features: bool = True

    def create_design(self) -> Design:
        """Create the :class:`Design`.

        Returns
        -------
        design : Design
            Validated design.
        """
        if not self.features:
            raise ValueError("features must be a non-empty list of dicts.")
        if not self.contrasts and not self.f_contrasts:
            raise ValueError(
                "contrasts or f_contrasts must be a non-empty list of dicts."
            )
        design = Design(standardize_features=self.standardize_features)
        for feature in self.features:
            design.add_feature(
                feature["name"], feature["values"], feature["feature_type"]
            )
        for contrast in self.contrasts or []:
            design.add_contrast(contrast["name"], contrast["values"])
        for contrast in self.f_contrasts or []:
            design.add_f_contrast(
                contrast["name"],
                values=contrast.get("values"),
                contrasts=contrast.get("contrasts"),
            )
        design.validate()
        return design
