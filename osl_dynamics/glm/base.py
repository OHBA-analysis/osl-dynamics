from typing import List, Dict
from dataclasses import dataclass
import numpy as np

from osl_dynamics.glm.ols import osl_fit, get_degree_of_freedom


# Helper functions for validation
def _validate_name(name):
    if not isinstance(name, str):
        raise ValueError(f"name must be a string, got {type(name)}")
    return name


def _validate_values(values):
    if isinstance(values, list):
        values = np.array(values)

    if not isinstance(values, np.ndarray):
        raise ValueError(f"values must be a list or np.ndarray, got {type(values)}")

    if values.ndim != 1:
        raise ValueError(f"values must be 1D, got {values.ndim}D")

    return values


def _validate_type(feature_type):
    if feature_type not in ["constant", "continuous", "categorical"]:
        raise ValueError(
            f"type must be 'constant' or 'continuous' or 'categorical', got {feature_type}"
        )
    return feature_type


@dataclass
class DesignConfig:
    """
    Configuration class for Design.

    Parameters
    ----------
    features : List[Dict]
        List of dictionaries containing feature information.
        Each dictionary should contain the following keys:
            - name: str
                Feature name.
            - values: np.ndarray or list
                Feature values. Must be 1D.
            - feature_type: str
                Feature type. Must be 'constant', 'continuous', or 'categorical'.
    contrasts : List[Dict]
        List of dictionaries containing contrast information.
        Each dictionary should contain the following keys:
            - name: str
                Contrast name.
            - values: np.ndarray or list
                Contrast values. Must be 1D.
    standardize_features : bool
        Whether to standardize continuous features. Default is True.
    """

    features: List[Dict] = None
    contrasts: List[Dict] = None
    standardize_features: bool = True

    def create_design(self):
        """
        Create a Design object from the configuration.

        Returns
        -------
        design : osl_dynamics.glm.base.Design
            Design object.
        """
        design = Design(standardize_features=self.standardize_features)
        for feature in self.features:
            design.add_feature(
                feature["name"], feature["values"], feature["feature_type"]
            )

        for contrast in self.contrasts:
            design.add_contrast(contrast["name"], contrast["values"])

        design.validate()
        return design


class Feature:
    """
    Base class for feature objects.

    Parameters
    ----------
    values : np.ndarray or list
        Feature values. Must be 1D.
    name : str
        Feature name.
    feature_type : str
        Feature type. Must be 'constant', 'continuous', or 'categorical'.
    """

    def __init__(self, name, values, feature_type):
        self.name = _validate_name(name)
        self.values = _validate_values(values)
        self.feature_type = _validate_type(feature_type)
        self.n_samples = len(self.values)

    def summary(self):
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "feature_type": self.feature_type,
        }


class Contrast:
    """
    Base class for contrast objects.

    Parameters
    ----------
    values : np.ndarray or list
        Contrast values. Must be 1D.
    name : str
        Contrast name.
    """

    def __init__(self, name, values):
        self.name = _validate_name(name)
        self.values = _validate_values(values)
        self.n_features = len(self.values)
        self.contrast_type = self._get_contrast_type()

    def _get_contrast_type(self):
        if self.values.sum() == 0:
            return "differential"
        else:
            return "main_effect"

    def summary(self):
        return {
            "name": self.name,
            "n_features": self.n_features,
            "contrast_type": self.contrast_type,
        }


class Design:
    """
    Base class for design objects.

    Parameters
    ----------
    features : List[Feature], optional
        List of Feature objects. Default is None.
    contrasts : List[Contrast], optional
        List of Contrast objects. Default is None.
    standardize_features : bool, optional
        Whether to standardize continuous features. Default is True.
    """

    def __init__(self, features=None, contrasts=None, standardize_features=True):
        self.features = [] if features is None else features
        self.contrasts = [] if contrasts is None else contrasts
        self.n_samples = None
        self.n_features = None
        self.standardize_features = standardize_features

    def add_feature(self, name, values, feature_type):
        """
        Add a feature to the design.

        Parameters
        ----------
        name : str
            Feature name.
        values : np.ndarray or list
            Feature values. Must be 1D.
        feature_type : str
            Feature type. Must be 'constant', 'continuous', or 'categorical'.
        """
        feature = Feature(name, values, feature_type)
        self.features.append(feature)

    def add_contrast(self, name, values):
        """
        Add a contrast to the design.

        Parameters
        ----------
        name : str
            Contrast name.
        values : np.ndarray or list
            Contrast values. Must be 1D.
        """
        contrast = Contrast(name, values)
        self.contrasts.append(contrast)

    def validate(self):
        """
        Validate the design.
        """
        self._validate_features()
        self._validate_contrasts()

        # n_features (contrast length) should match the number of features
        if self.n_features != len(self.features):
            raise ValueError(
                f"Number of features {len(self.features)} must match the length of the contrasts {self.n_features}."
            )

    def build_X(self):
        """
        Build the design matrix.

        Returns
        -------
        X : np.ndarray
            Design matrix. Shape is (n_samples, n_features).
        """
        self.validate()
        # Initialise X
        X = np.zeros((self.n_samples, self.n_features))
        for i, feature in enumerate(self.features):
            X[:, i] = feature.values

        if self.standardize_features:
            X = self._standardize_features(X)

        return X

    def build_contrast_array(self):
        """
        Build the contrast array.

        Returns
        -------
        contrast_array : np.ndarray
            Contrast array. Shape is (n_contrasts, n_features).
        """
        self.validate()
        # Initialise contrasts array
        contrast_array = np.zeros((self.n_contrasts, self.n_features))
        for i, contrast in enumerate(self.contrasts):
            contrast_array[i] = contrast.values
        return contrast_array

    def _standardize_features(self, X):
        """
        Standardize continuous features.

        Parameters
        ----------
        X : np.ndarray
            Design matrix. Shape is (n_samples, n_features).

        Returns
        -------
        X_copy : np.ndarray
            Standardized design matrix.
        """
        X_copy = X.copy()
        # Standardise continuous features
        cts_indx = np.where(np.array(self.feature_types) == "continuous")[0]
        if len(cts_indx) > 0:
            X_mean = np.mean(X_copy[:, cts_indx], axis=0)
            X_std = np.std(X_copy[:, cts_indx], axis=0)
            X_copy[:, cts_indx] = (X_copy[:, cts_indx] - X_mean) / X_std
        return X_copy

    def _validate_features(self):
        # TODO: Deal with rows with NaNs.
        if self.features is None:
            raise ValueError("No features found.")

        # all features should have the same length
        for feature in self.features:
            if not isinstance(feature, Feature):
                raise ValueError(
                    f"Expected Feature object, got {type(feature)} in features."
                )
            if self.n_samples is None:
                self.n_samples = feature.n_samples
            elif feature.n_samples != self.n_samples:
                raise ValueError(
                    f"All features must have the same number of samples, got {feature.n_samples} samples in {feature.name} and {self.n_samples} samples in the first feature."
                )

    def _validate_contrasts(self):
        if self.contrasts is None:
            raise ValueError("No contrasts found.")

        # all contrasts should have the same length
        for contrast in self.contrasts:
            if not isinstance(contrast, Contrast):
                raise ValueError(
                    f"Expected Contrast object, got {type(contrast)} in contrasts."
                )
            if self.n_features is None:
                self.n_features = contrast.n_features
            elif contrast.n_features != self.n_features:
                raise ValueError(
                    f"All contrasts must have the same number of features as the design matrix, got {contrast.n_features} features in {contrast.name} and {self.n_features} features in the design matrix."
                )

    @property
    def n_contrasts(self):
        return len(self.contrasts)

    @property
    def feature_names(self):
        return [f.name for f in self.features]

    @property
    def feature_types(self):
        return [f.feature_type for f in self.features]

    @property
    def contrast_names(self):
        return [c.name for c in self.contrasts]

    @property
    def contrast_types(self):
        return [c.contrast_type for c in self.contrasts]

    @property
    def dof(self):
        return get_degree_of_freedom(self.build_X())

    def summary(self):
        """Get a summary of the design."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "n_contrasts": self.n_contrasts,
            "contrast_names": self.contrast_names,
            "contrast_types": self.contrast_types,
            "dof": self.dof,
        }


class GLM:
    """
    Base class for GLM objects.

    Parameters
    ----------
    design : osl_dynamics.glm.base.Design
        Design object.
    """

    def __init__(self, design):
        self.design = design
        self.X = self.design.build_X()
        self.c = self.design.build_contrast_array()
        self.n_targets = None
        self.target_dims = None

    def fit(self, y):
        """
        Fit the GLM model.

        Parameters
        ----------
        y : np.ndarray or list
            Target values. Shape is (n_samples, *target_dims).
        """
        y_flatten = self._validate_y(y)
        betas, copes, varcopes = osl_fit(X=self.X, y=y_flatten, contrasts=self.c)
        self.betas = betas.reshape((self.n_features, *self.target_dims))
        self.copes = copes.reshape((self.n_contrasts, *self.target_dims))
        self.varcopes = varcopes.reshape((self.n_contrasts, *self.target_dims))

    def _validate_y(self, y):
        """
        Validate the target values and flatten the target dimensions.
        """
        if isinstance(y, list):
            y = np.array(y).copy()

        if not isinstance(y, np.ndarray):
            raise ValueError(f"y must be a list or np.ndarray, got {type(y)}")

        if y.ndim == 1:
            # Add target dimension
            y = y[:, None]

        if y.shape[0] != self.n_samples:
            raise ValueError(
                f"Number of samples in y ({y.shape[0]}) must match the number of samples in the design matrix ({self.n_samples})."
            )

        y_copy = y.copy()
        self.target_dims = y_copy.shape[1:]

        # Flatten the target dimensions
        y_copy = np.reshape(y_copy, (self.n_samples, -1))
        self.n_targets = y_copy.shape[1]
        return y_copy

    def get_tstats(self, copes=None, varcopes=None):
        """
        Get t-statistics.

        Parameters
        ----------
        copes : np.ndarray
            Contrast parameter estimates. Shape is (n_contrasts, *target_dims).
        varcopes : np.ndarray
            Variance of contrast parameter estimates. Shape is (n_contrasts, *target_dims).

        Returns
        -------
        tstats : np.ndarray
            t-statistics. Shape is (n_contrasts, *target_dims).
        """
        copes = copes or self.copes
        varcopes = varcopes or self.varcopes
        return copes / np.sqrt(varcopes)

    @property
    def n_samples(self):
        return self.design.n_samples

    @property
    def n_features(self):
        return self.design.n_features

    @property
    def feature_names(self):
        return self.design.feature_names

    @property
    def feature_types(self):
        return self.design.feature_types

    @property
    def n_contrasts(self):
        return self.design.n_contrasts

    @property
    def contrast_names(self):
        return self.design.contrast_names

    @property
    def contrast_types(self):
        return self.design.contrast_types

    @property
    def dof(self):
        return self.design.dof

    @property
    def tstats(self):
        return self.get_tstats(self.copes, self.varcopes)

    def summary(self):
        """Get a summary of the GLM."""
        sum = self.design.summary()
        sum["n_targets"] = self.n_targets
        sum["target_dims"] = self.target_dims
        return sum
