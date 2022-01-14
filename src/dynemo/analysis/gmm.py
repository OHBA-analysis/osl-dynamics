"""Functions for fitting a Gaussian mixture model.

"""

import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from dynemo.data.manipulation import standardize
from dynemo.utils import plotting


def fit_gaussian_mixture(
    X: np.ndarray,
    plot_filename: str = None,
    print_message: bool = True,
    bayesian: bool = True,
    label_order: str = "mean",
    **kwargs,
) -> np.ndarray:
    """Fits a two component Bayesian Gaussian mixture model.

    Parameters
    ----------
    X : np.ndarray
        Data to fit Gaussian mixture model to.
    plot_filename : str
        Filename to save a plot of the Gaussian mixture model. Optional.
    print_message : bool
        Should we print a message? Optional.
    bayesian : bool
        Should we fit a Bayesian GMM? Optional.
    label_order: str
        How do we order the inferred classes?
    **kwargs
        Keyword argument to pass to the sklearn class. Optional.

    Returns
    -------
    np.ndarray
        Class of each data point.
    """
    if print_message:
        print("Fitting GMM")

    # Validation
    if X.ndim == 1:
        X = X[:, np.newaxis]
    elif X.ndim != 2:
        raise ValueError("X must be a 1D or 2D numpy array.")

    # Standardise the data
    X = standardize(X)

    # Fit a Gaussian mixture model
    if bayesian:
        gm = BayesianGaussianMixture(n_components=2, **kwargs)
    else:
        gm = GaussianMixture(n_components=2, **kwargs)
    gm.fit(X)

    # Inferred parameters
    amplitudes = np.squeeze(gm.weights_) / np.sqrt(
        2 * np.pi * np.squeeze(gm.covariances_)
    )
    means = np.squeeze(gm.means_)
    variances = np.sqrt(np.squeeze(gm.covariances_))
    if label_order == "mean":
        order = np.argsort(means)
    elif label_order == "variance":
        order = np.argsort(variances)
    else:
        raise NotImplementedError(label_order)

    # Plots
    if plot_filename is not None:
        plotting.plot_gmm(X[:, 0], amplitudes[order], means[order], variances[order], filename=plot_filename)

    # Which component does each data point correspond to
    y = gm.predict(X)

    # Deal with label switching
    if label_order == "mean":
        if means[0] > means[1]:
            # 1 -> 0; 0 -> 1 
            y = (1-y).astype(int)
    if label_order == "variance":
        if variances[0] > variances[1]:
            y = (1-y).astype(int)

    return y
