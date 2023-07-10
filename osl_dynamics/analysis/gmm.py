"""Functions related to Gaussian Mixture Models (GMMs).

"""

import logging

import numpy as np
from scipy import special, stats
from sklearn.mixture import GaussianMixture

from osl_dynamics.utils import plotting

_logger = logging.getLogger("osl-dynamics")


def fit_gaussian_mixture(
    X,
    logit_transform=False,
    standardize=True,
    p_value=None,
    one_component_percentile=None,
    n_sigma=0,
    label_order="mean",
    sklearn_kwargs=None,
    return_statistics=False,
    show_plot=False,
    plot_filename=None,
    plot_kwargs=None,
    log_message=True,
):
    """Fits a two-component Gaussian Mixture Model (GMM).

    Parameters
    ----------
    X : np.ndarray
        Data to fit GMM to. Must be 1D.
    logit_transform : bool, optional
        Should we logit transform the :code:`X`?
    standardize : bool, optional
        Should we standardize :code:`X`?
    p_value : float, optional
        Used to determine a threshold. We ensure the data points assigned to
        the 'on' component have a probability of less than :code:`p_value` of
        belonging to the 'off' component.
    one_component_percentile : float, optional
        Percentile threshold if only one component is found. Should be between
        0 and 100. E.g. for the 95th percentile,
        :code:`one_component_percentile=95`.
    n_sigma : float, optional
        Number of standard deviations of the 'off' component the mean of the
        'on' component must be for the fit to be considered to have two
        components.
    label_order: str, optional
        How do we order the inferred classes?
    sklearn_kwargs : dict, optional
        Dictionary of keyword arguments to pass to
        `sklearn.mixture.GaussianMixture <https://scikit-learn.org/stable\
        /modules/generated/sklearn.mixture.GaussianMixture.html>`_.
    return_statistics: bool, optional
        Should we return statistics of the Gaussian mixture components?
    show_plot : bool, optional
        Should we show the GMM fit to the distribution of :code:`X`.
    plot_filename : str, optional
        Filename to save a plot of the Gaussian mixture model.
    plot_kwargs : dict, optional
        Keyword arguments to pass to `utils.plotting.plot_gmm
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /utils/plotting/index.html#osl_dynamics.utils.plotting.plot_gmm>`_
        Only used if :code:`plot_filename` is not :code:`None`.
    log_message : bool
        Should we log a message?

    Returns
    -------
    threshold : float
        Threshold for the on class.
    """
    if sklearn_kwargs is None:
        sklearn_kwargs = {}

    if plot_kwargs is None:
        plot_kwargs = {}

    if log_message:
        _logger.info("Fitting GMM")

    # Copy the data so we don't modify it
    X_ = np.copy(X)

    # Validation
    if X.ndim != 1:
        raise ValueError("X must be a 1D numpy array.")
    else:
        X_ = X_[:, np.newaxis]
        X = X[:, np.newaxis]

    # Logit transform
    if logit_transform:
        X_ = special.logit(X)
        X_[np.isinf(X_[:, 0]), :] = np.mean(X_[~np.isinf(X_[:, 0]), 0])

    # Standardise the data
    if standardize:
        std = np.std(X_, axis=0)
        if std == 0:
            return max(X)
        mu = np.mean(X_, axis=0)
        X_ -= mu
        X_ /= std

    # Fit a Gaussian mixture model
    gm = GaussianMixture(n_components=2, **sklearn_kwargs)
    gm.fit(X_)

    # Inferred parameters
    amplitudes = np.squeeze(gm.weights_) / np.sqrt(
        2 * np.pi * np.squeeze(gm.covariances_)
    )
    means = np.squeeze(gm.means_)
    stddevs = np.sqrt(np.squeeze(gm.covariances_))
    if label_order == "mean":
        order = np.argsort(means)
    elif label_order == "variance":
        order = np.argsort(stddevs)
    else:
        raise NotImplementedError(label_order)

    # Order the components
    amplitudes = amplitudes[order]
    means = means[order]
    stddevs = stddevs[order]

    # Calculate a threshold to distinguish between components
    if (
        abs(means[1] - means[0]) < n_sigma * stddevs[0]
        and one_component_percentile is not None
    ):
        # Reorder data in an ascending order
        ascending = np.argsort(X_[:, 0])
        X_ = X_[ascending]
        X = X[ascending]

        # The Gaussians are not sufficiently distinct to define a threshold
        index = one_component_percentile * len(X) // 100

    elif p_value is not None:
        # We decide the threshold based on the probability of a data point
        # belonging to the 'off' component. We assign a data point to the 'on'
        # component if its probability of belonging to the 'off' component is
        # less than the p_value

        # Calculate the probability of each data point belonging to each
        # component. The variable 'a' is the 'activation'
        dX = max(X_) / 100
        x = np.arange(means[0], max(X_) + dX, dX)
        a = np.array(
            [stats.norm.pdf(x, loc, scale) for loc, scale in zip(means, stddevs)]
        ).T
        a *= gm.weights_

        # Find the index of the data point closest to the desired p-value
        # This defines the threshold in the standardised/logit transformed space
        x_threshold = x[np.argmin(np.abs(a[:, 0] - p_value / X_.shape[0]))]
        index = np.argmin(np.abs(X_[:, 0] - x_threshold))

    else:
        # Calculate the probability of each data point belonging to each
        # component
        ascending = np.argsort(X_[:, 0])
        X_ = X_[ascending]
        X = X[ascending]
        y = gm.predict_proba(X_)
        y = y[:, order]

        # Get the index of the first data point classified as the 'on' component
        on_prob_higher = y[:, 0] < y[:, 1]
        on_prob_higher[X_[:, 0] < means[0]] = False
        index = np.argmax(on_prob_higher)

    # Get the threshold in the standardised/logit transform and original space
    threshold_ = X_[index, 0]
    threshold = X[index, 0]

    # Plots
    if show_plot or plot_filename is not None:
        fig, ax = plotting.plot_gmm(
            X_[:, 0],
            amplitudes,
            means,
            stddevs,
            title=f"Threshold = {threshold_:.3}",
            **plot_kwargs,
        )
        ax.axvline(threshold_, color="black", linestyle="--")
        if plot_filename is not None:
            plotting.save(fig, plot_filename)
            plotting.close()

    # Return Gaussian component metrics
    if return_statistics:
        metrics = dict(
            threshold=threshold_,
            data=X_[:, 0],
            amplitudes=amplitudes,
            means=means,
            stddevs=stddevs,
        )
        return threshold, metrics

    return threshold
