"""Functions related to Gaussian Mixture Models (GMMs).

"""

import numpy as np
from scipy import stats
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from osl_dynamics.data import processing
from osl_dynamics.utils import plotting


def fit_gaussian_mixture(
    X,
    bayesian=True,
    standardize=True,
    label_order="mean",
    sklearn_kwargs=None,
    min_percentile=0,
    max_percentile=100,
    plot_filename=None,
    plot_kwargs=None,
    print_message=True,
):
    """Fits a two component Bayesian Gaussian mixture model.

    Parameters
    ----------
    X : np.ndarray
        Data to fit Gaussian mixture model to.
    bayesian : bool
        Should we fit a Bayesian GMM?
    standardize : bool
        Should we standardize X?
    label_order: str
        How do we order the inferred classes?
    sklearn_kwargs : dict
        Keyword arguments to pass to the sklearn class.
    min_percentile : float
        Minimum percentile for the threshold. Should be between 0 and 100.
        E.g. for the 90th percentile, max_percentile=90.
    max_percentile : float
        Maximum percentile for the threshold. Should be a between 0 and 100.
        E.g. for the 95th percentile, max_percentile=95.
    plot_filename : str
        Filename to save a plot of the Gaussian mixture model.
    plot_kwargs : dict
        Keyword arguments to pass to plotting function.
        Only used if plot_filename is not None.
    print_message : bool
        Should we print a message?

    Returns
    -------
    y : np.ndarray
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
    if standardize:
        X = processing.standardize(X)

    # Fit a Gaussian mixture model
    if bayesian:
        gm = BayesianGaussianMixture(n_components=2, **sklearn_kwargs)
    else:
        gm = GaussianMixture(n_components=2, **sklearn_kwargs)
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

    # Which component does each data point correspond to
    y = gm.predict(X)

    # Deal with label switching
    if label_order == "mean":
        if means[0] > means[1]:
            # 1 -> 0; 0 -> 1
            y = (1 - y).astype(int)
    if label_order == "variance":
        if variances[0] > variances[1]:
            y = (1 - y).astype(int)

    # Percentile threshold
    percentile = get_percentile_threshold(X[:, 0], y, min_percentile, max_percentile)

    # Plots
    if plot_filename is not None:
        fig, ax = plotting.plot_gmm(
            X[:, 0],
            amplitudes[order],
            means[order],
            variances[order],
            title=f"Percentile = {round(percentile)}",
            **plot_kwargs,
        )
        threshold = np.percentile(X[:, 0], percentile)
        ax.axvline(threshold, color="black", linestyle="--")
        plotting.save(fig, plot_filename)

    return percentile


def get_percentile_threshold(X, y, min_percentile=0, max_percentile=100):
    """Calculate the percentile threshold for determining class labels
    from a two component GMM.

    Parameters
    ----------
    X : np.ndarray
        Data used to fit a GMM.
    y : np.ndarray
        Class labels. This must be an array of 0s and 1s, where 0 indicates an
        'off' component and 1 indicates an 'on' component.
    min_percentile : float
        Minimum percentile for the threshold. Should be between 0 and 100.
        E.g. for the 90th percentile, max_percentile=90.
    max_percentile : float
        Maximum percentile for the threshold. Should be a between 0 and 100.
        E.g. for the 95th percentile, max_percentile=95.

    Returns
    -------
    threshold : float
        Largest value out of the two options: smallest value in the X array that
        belongs to the 'on' class and largest value in the X array that belongs to
        the 'off' class. Value is returned as a percentile of X.
    """

    # Validation
    if min_percentile > 100 or min_percentile < 0:
        raise ValueError("min_percentile must be between 0 and 100.")

    if max_percentile > 100 or max_percentile < 0:
        raise ValueError("max_percentile must be between 0 and 100.")

    if min_percentile >= max_percentile:
        raise ValueError("min_percentile must be less than max_percentile.")

    # Get the threshold for determining the class
    threshold = np.max([np.min(X[y == 1]), np.max(X[y == 0])])

    # What percentile of the full distribution is the threshold?
    percentile = stats.percentileofscore(X, threshold)

    return max(min(percentile, max_percentile), min_percentile)
