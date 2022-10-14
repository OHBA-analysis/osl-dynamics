"""Functions related to Gaussian Mixture Models (GMMs).

"""

import numpy as np
from scipy import stats, special
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from osl_dynamics.utils import plotting


def fit_gaussian_mixture(
    X,
    bayesian=True,
    logit_transform=False,
    standardize=True,
    label_order="mean",
    sklearn_kwargs={},
    one_component_percentile=None,
    n_sigma=0,
    plot_filename=None,
    plot_kwargs={},
    print_message=True,
    return_labels=False,
):
    """Fits a two component Bayesian Gaussian mixture model.

    Parameters
    ----------
    X : np.ndarray
        Data to fit Gaussian mixture model to.
    bayesian : bool
        Should we fit a Bayesian GMM?
    logit_transform : bool
        Should we logit transform the X?
    standardize : bool
        Should we standardize X?
    label_order: str
        How do we order the inferred classes?
    sklearn_kwargs : dict
        Keyword arguments to pass to the sklearn class.
    one_component_percentile : float
        Percentile threshold if only one component is found.
        Should be a between 0 and 100. E.g. for the 95th percentile,
        one_component_percentile=95.
    n_sigma : float
        Number of standard deviations of the 'off' component the mean
        of the 'on' component must be for the fit to be considered to
        have two components.
    plot_filename : str
        Filename to save a plot of the Gaussian mixture model.
    plot_kwargs : dict
        Keyword arguments to pass to plotting function.
        Only used if plot_filename is not None.
    print_message : bool
        Should we print a message?
    return_labels : bool
        Should we return the labels?

    Returns
    -------
    y : float or np.ndarray
        Percentile for thresholding or class of each data point if
        return_labels=True.
    """
    if print_message:
        print("Fitting GMM")

    # Copy the data so we don't modify it
    X = np.copy(X)

    # Validation
    if X.ndim != 1:
        raise ValueError("X must be a 1D numpy array.")
    else:
        X = X[:, np.newaxis]

    # Logit transform
    if logit_transform:
        X = special.logit(X)
        X[np.isinf(X[:, 0]), :] = np.mean(X[~np.isinf(X[:, 0]), 0])

    # Standardise the data
    if standardize:
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)

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
    amplitudes = amplitudes[order]
    means = means[order]
    variances = variances[order]

    # Percentile threshold
    if (
        abs(means[1] - means[0]) < n_sigma * np.sqrt(variances[0])
        and one_component_percentile is not None
    ):
        percentile = one_component_percentile
    else:
        percentile = get_percentile_threshold(X[:, 0], y, means)

    # Plots
    if plot_filename is not None:
        fig, ax = plotting.plot_gmm(
            X[:, 0],
            amplitudes,
            means,
            variances,
            title=f"Percentile = {round(percentile)}",
            **plot_kwargs,
        )
        threshold = np.percentile(X[:, 0], percentile)
        ax.axvline(threshold, color="black", linestyle="--")
        plotting.save(fig, plot_filename)

    if return_labels:
        return y
    else:
        return percentile


def get_percentile_threshold(X, y, mu):
    """Calculate the percentile threshold for determining class labels
    from a two component GMM.

    Parameters
    ----------
    X : np.ndarray
        Data used to fit a GMM.
    y : np.ndarray
        Class labels. This must be an array of 0s and 1s, where 0 indicates an
        'off' component and 1 indicates an 'on' component.
    mu : np.ndarray
        Mean of each class.

    Returns
    -------
    threshold : float
        Largest value out of the two options: smallest value in the X array that
        belongs to the 'on' class and largest value in the X array that belongs to
        the 'off' class. Value is returned as a percentile of X.
    """

    # Get the threshold for determining the class
    min_threshold = np.min([np.min(X[y == 1]), np.max(X[y == 0])])
    max_threshold = np.max([np.min(X[y == 1]), np.max(X[y == 0])])

    # Pick the threshold that is between the means
    if mu[0] < min_threshold < mu[1]:
        threshold = min_threshold
    else:
        threshold = max_threshold

    # What percentile of the full distribution is the threshold?
    percentile = stats.percentileofscore(X, threshold)

    return percentile
