"""Functions for fitting a Gaussian mixture model.

"""

from typing import Tuple

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from dynemo.data.manipulation import standardize
from dynemo.utils import plotting


def fit_gmm(
    X: np.ndarray, n_fits: int = 1, max_iter: int = 1000, plot_filename: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fits a two component Bayesian Gaussian mixture model.

    Parameters
    ----------
    X : np.ndarray
        Data to fit Gaussian mixture model to.
    n_fits : int
        How many times should we fit the Gaussian mixture model. Optional.
    max_iter : int
        Maximum number of iterations. Optional.
    plot_filename : str
        Filename to save a plot of the Gaussian mixture model. Optional.

    Returns
    -------
    amplitudes : np.ndarray
        Amplitude of Gaussian components.
    means : np.ndarray
        Mean of Gaussian components.
    variances : np.ndarray
        Variance of Gaussian components.
    """
    print("Fitting GMM")

    # Validation
    if X.ndim == 1:
        X = X[:, np.newaxis]
    elif X.ndim != 2:
        raise ValueError("X must be a 1D or 2D numpy array.")

    # Standardise the data
    X = standardize(X)

    # Fit a Gaussian mixture model
    lower_bound = np.Inf
    for i in range(n_fits):
        bgm = BayesianGaussianMixture(n_components=2, max_iter=max_iter)
        bgm.fit(X)
        if bgm.lower_bound_ < lower_bound:
            # Inferred parameters
            amplitudes = np.squeeze(bgm.weights_) / np.sqrt(
                2 * np.pi * np.squeeze(bgm.covariances_)
            )
            means = np.squeeze(bgm.means_)
            variances = np.sqrt(np.squeeze(bgm.covariances_))

            # Update best lower bound
            lower_bound = bgm.lower_bound_

    # Plots
    if plot_filename:
        plotting.plot_gmm(X[:, 0], amplitudes, means, variances, filename=plot_filename)

    return amplitudes, means, variances
