"""Functions for fitting a Gaussian mixture model.

"""

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from dynemo.data.manipulation import standardize
from dynemo.utils import plotting


def fit_gaussian_mixture(
    X: np.ndarray,
    plot_filename: str = None,
    print_message: bool = True,
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
    bgm = BayesianGaussianMixture(n_components=2, max_iter=5000, n_init=10)
    bgm.fit(X)

    # Inferred parameters
    amplitudes = np.squeeze(bgm.weights_) / np.sqrt(
        2 * np.pi * np.squeeze(bgm.covariances_)
    )
    means = np.squeeze(bgm.means_)
    variances = np.sqrt(np.squeeze(bgm.covariances_))

    # Plots
    if plot_filename is not None:
        plotting.plot_gmm(X[:, 0], amplitudes, means, variances, filename=plot_filename)

    # Which component does each data point correspond to
    y = bgm.predict(X)

    return y
