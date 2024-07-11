"""Functions to calculate and plot network connectivity.

"""

import os
import logging
from pathlib import Path

import numpy as np
from nilearn import plotting
from scipy import stats
from tqdm.auto import trange
from pqdm.threads import pqdm
import matplotlib.pyplot as plt

from osl_dynamics import array_ops
from osl_dynamics.analysis import gmm
from osl_dynamics.analysis.spectral import get_frequency_args_range
from osl_dynamics.utils.misc import override_dict_defaults
from osl_dynamics.utils.parcellation import Parcellation

_logger = logging.getLogger("osl-dynamics")


def sliding_window_connectivity(
    data,
    window_length,
    step_size=None,
    conn_type="corr",
    concatenate=False,
    n_jobs=1,
):
    """Calculate sliding window connectivity.

    Parameters
    ----------
    data : list or np.ndarray
        Time series data. Shape must be (n_sessions, n_samples, n_channels)
        or (n_samples, n_channels).
    window_length : int
        Window length in samples.
    step_size : int, optional
        Number of samples to slide the window along the time series.
        If :code:`None`, then :code:`step_size=window_length // 2`.
    conn_type : str, optional
        Metric to use to calculate pairwise connectivity in the network.
        Should use :code:`"corr"` for Pearson correlation or :code:`"cov"`
        for covariance.
    concatenate : bool, optional
        Should we concatenate the sliding window connectivities from each
        array into one big time series?
    n_jobs : int, optional
        Number of parallel jobs to run. Default is 1.

    Returns
    -------
    sliding_window_conn : list or np.ndarray
        Time series of connectivity matrices. Shape is (n_sessions, n_windows,
        n_channels, n_channels) or (n_windows, n_channels, n_channels).
    """
    # Validation
    if conn_type not in ["corr", "cov"]:
        raise ValueError("conn_type must be 'corr' or 'cov'.")

    if conn_type == "cov":
        metric = np.cov
    else:
        metric = np.corrcoef

    if step_size is None:
        step_size = window_length // 2

    if isinstance(data, np.ndarray):
        if data.ndim != 3:
            data = [data]

    # Helper function to calculate connectivity
    def _swc(x):
        n_samples = x.shape[0]
        n_channels = x.shape[1]
        n_windows = (n_samples - window_length - 1) // step_size + 1

        # Preallocate an array to hold moving average values
        swc = np.empty([n_windows, n_channels, n_channels], dtype=np.float32)

        # Compute connectivity matrix for each window
        for i in range(n_windows):
            window_ts = x[i * step_size : i * step_size + window_length]
            swc[i] = metric(window_ts, rowvar=False)

        return swc

    # Setup keyword arguments to pass to the helper function
    kwargs = [{"x": x} for x in data]

    if len(data) == 1:
        _logger.info("Sliding window connectivity")
        results = [_swc(**kwargs[0])]

    elif n_jobs == 1:
        results = []
        for i in trange(len(data), desc="Sliding window connectivity"):
            results.append(_swc(**kwargs[i]))

    else:
        _logger.info("Sliding window connectivity")
        results = pqdm(
            kwargs,
            _swc,
            argument_type="kwargs",
            n_jobs=n_jobs,
        )

    if concatenate or len(results) == 1:
        results = np.concatenate(results)

    return results


def covariance_from_spectra(
    f,
    cpsd,
    components=None,
    frequency_range=None,
):
    """Calculates covariance from cross power spectra.

    Parameters
    ----------
    f : np.ndarray
        Frequency axis of the spectra. Only used if :code:`frequency_range`
        is given. Shape must be (n_freq,).
    cpsd : np.ndarray
        Cross power spectra. Shape must be
        (n_modes, n_channels, n_channels, n_freq).
    components : np.ndarray, optional
        Spectral components. Shape must be (n_components, n_freq).
    frequency_range : list, optional
        Frequency range to integrate the PSD over (Hz).
        Default is the full range.

    Returns
    -------
    covar : np.ndarray
        Covariance over a frequency band for each component of each mode.
        Shape is (n_components, n_modes, n_channels, n_channels).
    """

    # Validation
    error_message = (
        "A (n_channels, n_channels, n_freq), "
        + "(n_modes, n_channels, n_channels, n_freq) or "
        + "(n_sessions, n_modes, n_channels, n_channels, n_freq) "
        + "array must be passed."
    )
    cpsd = array_ops.validate(
        cpsd,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )

    if components is not None and frequency_range is not None:
        raise ValueError(
            "Only one of the arguments components or frequency range can be passed."
        )

    if frequency_range is not None and f is None:
        raise ValueError(
            "If frequency_range is passed, frequenices must also be passed."
        )

    # Dimensions
    n_sessions, n_modes, n_channels, n_channels, n_freq = cpsd.shape
    if components is None:
        n_components = 1
    else:
        n_components = components.shape[0]

    # Calculate connectivity maps for each array
    covar = []
    for i in range(n_sessions):
        # Cross spectral densities
        csd = cpsd[i].reshape(-1, n_freq)
        csd = abs(csd)

        if components is not None:
            # Calculate PSD for each spectral component
            c = components @ csd.T
            for j in range(n_components):
                c[j] /= np.sum(components[j])

        else:
            # Integrate over the given frequency range
            if frequency_range is None:
                c = np.sum(csd, axis=-1)
            else:
                [min_arg, max_arg] = get_frequency_args_range(f, frequency_range)
                c = np.sum(csd[..., min_arg:max_arg], axis=-1)

        c = c.reshape(n_components, n_modes, n_channels, n_channels)
        covar.append(c)

    return np.squeeze(covar)


def mean_coherence_from_spectra(
    f,
    coh,
    components=None,
    frequency_range=None,
):
    """Calculates mean coherence from spectra.

    Parameters
    ----------
    f : np.ndarray
        Frequency axis of the spectra. Only used if :code:`frequency_range` is
        given. Shape must be (n_freq,).
    coh : np.ndarray
        Coherence for each channel. Shape must be (n_modes, n_channels,
        n_channels, n_freq).
    components : np.ndarray, optional
        Spectral components. Shape must be (n_components, n_freq).
    frequency_range : list, optional
        Frequency range to integrate the PSD over (Hz).

    Returns
    -------
    mean_coh : np.ndarray
        Mean coherence over a frequency band for each component of each mode.
        Shape is (n_components, n_modes, n_channels, n_channels) or
        (n_modes, n_channels, n_channels) or (n_channels, n_channels).
    """

    # Validation
    error_message = (
        "a 3D numpy array (n_channels, n_channels, n_freq) "
        + "or 4D numpy array (n_modes, n_channels, n_channels, "
        + "n_freq) must be passed for spectra."
    )
    coh = array_ops.validate(
        coh,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )

    if components is not None and frequency_range is not None:
        raise ValueError(
            "Only one of the arguments components or frequency range can be passed."
        )

    if frequency_range is not None and f is None:
        raise ValueError(
            "If frequency_range is passed, frequenices must also be passed."
        )

    # Dimensions
    n_sessions, n_modes, n_channels, n_channels, n_freq = coh.shape
    if components is None:
        n_components = 1
    else:
        n_components = components.shape[0]

    # Calculate mean coherence for each array
    mean_coh = []
    for i in range(n_sessions):
        # Concatenate over modes
        c = coh[i].reshape(-1, n_freq)

        if components is not None:
            # Coherence for each spectral component
            c = components @ c.T
            for j in range(n_components):
                c[j] /= np.sum(components[j])

        else:
            # Mean over the given frequency range
            if frequency_range is None:
                c = np.mean(c, axis=-1)
            else:
                [min_arg, max_arg] = get_frequency_args_range(f, frequency_range)
                c = np.mean(c[..., min_arg:max_arg], axis=-1)

        c = c.reshape(n_components, n_modes, n_channels, n_channels)
        mean_coh.append(c)

    return np.squeeze(mean_coh)


def mean_connections(conn_map):
    """Average the edges for each node.

    Parameters
    ----------
    conn_map : np.ndarray
        A (..., n_channels, n_channels) connectivity matrix.

    Returns
    -------
    mean_connections : np.ndarray
        A (..., n_channels) matrix.
    """
    return np.mean(conn_map, axis=-1)


def eigenvectors(
    conn_map,
    n_eigenvectors=1,
    absolute_value=False,
    as_network=False,
):
    """Calculate eigenvectors of a connectivity matrix.

    Parameters
    ----------
    conn_map : np.ndarray
        Connectivity matrix. Shape must be (..., n_channels, n_channels).
    n_eigenvectors : int, optional
        Number of eigenvectors to include.
    absolute_value : bool, optional
        Should we take the absolute value of the connectivity matrix before
        calculating the eigen decomposition?
    as_network : bool, optional
        Should we return a matrix?

    Returns
    -------
    eigenvectors : np.ndarray.
        Eigenvectors. Shape is (n_eigenvectors, ..., n_channels, n_channels)
        if :code:`as_network=True`, otherwise it is
        (n_eigenvectors, ..., n_channels). If :code:`n_eigenvectors=1`,
        the first dimension is removed.
    """
    if absolute_value:
        # Take absolute value
        conn_map = abs(conn_map)

    # Calculate eigen decomposition
    _, eigenvectors = np.linalg.eigh(conn_map)

    # Reorder from ascending eigenvalues to descending
    eigenvectors = eigenvectors[..., ::-1]

    # Keep the requested number of eigenvectors and make the first axis
    # specify the eigenvector
    eigenvectors = np.rollaxis(eigenvectors[..., :n_eigenvectors], -1)

    if as_network:
        # Calculate the outer product using the eigenvectors
        eigenvectors = np.expand_dims(eigenvectors, axis=-1) @ np.expand_dims(
            eigenvectors, axis=-2
        )

    return np.squeeze(eigenvectors)


def gmm_threshold(
    conn_map,
    subtract_mean=False,
    mean_weights=None,
    standardize=False,
    p_value=None,
    keep_positive_only=False,
    one_component_percentile=0,
    n_sigma=0,
    sklearn_kwargs=None,
    show=False,
    filename=None,
    plot_kwargs=None,
):
    """Threshold a connectivity matrix using the GMM method.

    Wrapper for combining `connectivity.fit_gmm <https://osl-dynamics\
    .readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity\
    /index.html#osl_dynamics.analysis.connectivity.fit_gmm>`_ and
    `connectivity.threshold <https://osl-dynamics.readthedocs.io/en/latest\
    /autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics\
    .analysis.connectivity.threshold>`_.

    Parameters
    ----------
    conn_map : np.ndarray
        Connectivity matrix. Shape must be (n_components, n_modes,
        n_channels, n_channels) or (n_modes, n_channels, n_channels)
        or (n_channels, n_channels).
    subtract_mean : bool, optional
        Should we subtract the mean over modes before fitting a GMM?
    mean_weights: np.ndarray, optional
        Numpy array with weightings for each mode/state to use to calculate
        the mean. Default is equal weighting.
    standardize : bool, optional
        Should we standardize the input to the GMM?
    p_value : float, optional
        Used to determine a threshold. We ensure the data points assigned to
        the 'on' component have a probability of less than :code:`p_value` of
        belonging to the 'off' component.
    keep_positive_only : bool, optional
        Should we only keep positive values to fit a GMM to?
    one_component_percentile : float, optional
        Percentile threshold if only one component is found. Should be a
        between 0 and 100. E.g. for the 95th percentile,
        :code:`one_component_percentile=95`.
    n_sigma : float, optional
        Number of standard deviations of the 'off' component the mean of the
        'on' component must be for the fit to be considered to have two
        components.
    sklearn_kwargs : dict, optional
        Dictionary of keyword arguments to pass to
        `sklearn.mixture.GaussianMixture <https://scikit-learn.org/stable\
        /modules/generated/sklearn.mixture.GaussianMixture.html>`_.
    show : bool, optional
        Should we show the GMM fit to the distribution of :code:`conn_map`.
    filename : str, optional
        Filename to save fit to.
    plot_kwargs : dict, optional
        Dictionary of keyword arguments to pass to `utils.plotting.plot_gmm
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /utils/plotting/index.html#osl_dynamics.utils.plotting.plot_gmm>`_.

    Returns
    -------
    conn_map : np.ndarray
        Thresholded connectivity matrix. The shape is the same as the original
        :code:`conn_map`.
    """
    percentile = fit_gmm(
        conn_map,
        subtract_mean,
        mean_weights,
        standardize,
        p_value,
        keep_positive_only,
        one_component_percentile,
        n_sigma,
        sklearn_kwargs,
        show,
        filename,
        plot_kwargs,
    )
    conn_map = threshold(conn_map, percentile, subtract_mean)
    return conn_map


def fit_gmm(
    conn_map,
    subtract_mean=False,
    mean_weights=None,
    standardize=False,
    p_value=None,
    keep_positive_only=False,
    one_component_percentile=0,
    n_sigma=0,
    sklearn_kwargs=None,
    show=False,
    filename=None,
    plot_kwargs=None,
):
    """Fit a two component GMM to connections to identify a threshold.

    Parameters
    ----------
    conn_map : np.ndarray
        Connectivity map. Shape must be (n_components, n_modes, n_channels,
        n_channels) or (n_modes, n_channels, n_channels) or (n_channels,
        n_channels).
    subtract_mean : bool, optional
        Should we subtract the mean over modes before fitting a GMM?
    mean_weights: np.ndarray, optional
        Numpy array with weightings for each mode/state to use to calculate
        the mean. Default is equal weighting.
    standardize : bool, optional
        Should we standardize the input to the GMM?
    p_value : float, optional
        Used to determine a threshold. We ensure the data points assigned to
        the 'on' component have a probability of less than :code:`p_value` of
        belonging to the 'off' component.
    keep_positive_only : bool, optional
        Should we only keep positive values to fit a GMM to?
    one_component_percentile : float, optional
        Percentile threshold if only one component is found. Should be a
        between 0 and 100. E.g. for the 95th percentile,
        :code:`one_component_percentile=95`.
    n_sigma : float, optional
        Number of standard deviations of the 'off' component the mean of the
        'on' component must be for the fit to be considered to have two
        components.
    sklearn_kwargs : dict, optional
        Dictionary of keyword arguments to pass to
        `sklearn.mixture.GaussianMixture <https://scikit-learn.org/stable\
        /modules/generated/sklearn.mixture.GaussianMixture.html>`_
        Default is :code:`{"max_iter": 5000, "n_init": 10}`.
    show : bool, optional
        Should we show the GMM fit to the distribution of :code:`conn_map`.
    filename : str, optional
        Filename to save fit to.
    plot_kwargs : dict, optional
        Dictionary of keyword arguments to pass to `utils.plotting.plot_gmm
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /utils/plotting/index.html#osl_dynamics.utils.plotting.plot_gmm>`_.

    Returns
    -------
    percentile : np.ndarray
        Percentile threshold. Shape is (n_components, n_modes) or (n_modes,).
    """
    # Validation
    conn_map = array_ops.validate(
        conn_map,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message="conn_map must be (n_modes, n_channels, n_channels) "
        + "or (n_channels, n_channels).",
    )

    if sklearn_kwargs is None:
        sklearn_kwargs = {}
    default_sklearn_kwargs = {"max_iter": 5000, "n_init": 10}
    sklearn_kwargs = override_dict_defaults(default_sklearn_kwargs, sklearn_kwargs)

    # Number of components, modes and channels
    n_components = conn_map.shape[0]
    n_modes = conn_map.shape[1]
    n_channels = conn_map.shape[2]

    # Mean over modes
    mean_conn_map = np.average(conn_map, axis=1, weights=mean_weights)

    # Indices for off diagonal elements
    m, n = np.triu_indices(n_channels, k=1)

    # Calculate thresholds by fitting a GMM
    percentiles = np.empty([n_components, n_modes])
    for i in range(n_components):
        for j in range(n_modes):
            # Off diagonal connectivity values to fit a GMM to
            if subtract_mean:
                c = conn_map[i, j, m, n] - mean_conn_map[i, m, n]
            else:
                c = conn_map[i, j, m, n]

            if keep_positive_only:
                # Only keep positive entries
                # (this is what's done in MATLAB OSL's teh_graph_gmm_fit)
                c = c[c > 0]
                if len(c) == 0:
                    percentiles[i, j] = 100
                    continue

            # Output filename
            if filename is not None:
                plot_filename = (
                    "{fn.parent}/{fn.stem}{i:0{w1}d}_{j:0{w2}d}{fn.suffix}".format(
                        fn=Path(filename),
                        i=i,
                        j=j,
                        w1=len(str(n_components)),
                        w2=len(str(n_modes)),
                    )
                )
            else:
                plot_filename = None

            # Fit a GMM to get class labels
            threshold = gmm.fit_gaussian_mixture(
                c,
                standardize=standardize,
                p_value=p_value,
                one_component_percentile=one_component_percentile,
                n_sigma=n_sigma,
                sklearn_kwargs=sklearn_kwargs,
                show_plot=show,
                plot_filename=plot_filename,
                plot_kwargs=plot_kwargs,
                log_message=False,
            )

            # Calculate the percentile from the threshold
            percentiles[i, j] = stats.percentileofscore(c, threshold)

    return np.squeeze(percentiles)


def threshold(
    conn_map,
    percentile,
    subtract_mean=False,
    mean_weights=None,
    absolute_value=False,
    return_edges=False,
):
    """Return edges that exceed a threshold.

    Parameters
    ---------
    conn_map : np.ndarray
        Connectivity matrix to threshold. Shape must be (n_components, n_modes,
        n_channels, n_channels), (n_modes, n_channels, n_channels) or
        (n_channels, n_channels).
    percentile : float or np.ndarray
        Percentile to threshold with. Should be between 0 and 100.
        Shape must be (n_components, n_modes), (n_modes,) or a float.
    subtract_mean : bool, optional
        Should we subtract the mean over modes before thresholding? The
        thresholding is only done to identify edges, the values returned in
        :code:`conn_map` are not mean subtracted.
    mean_weights : np.ndarray, optional
        Weights when calculating the mean over modes.
    absolute_value : bool, optional
        Should we take the absolute value before thresholding? The thresholding
        is only done to identify edges, the values returned in :code:`conn_map`
        are not absolute values. If :code:`subtract_mean=True`, the mean is
        subtracted before the absolute value.
    return_edges : bool, optional
        Should we return a boolean array for whether edges are above the
        threshold?

    Returns
    -------
    conn_map : np.ndarray
        Connectivity matrix with connections below the threshold set to zero.
        Or a boolean array if :code:`return_edges=True`. Shape is the same as
        the original :code:`conn_map`.
    """
    # Validation
    conn_map = array_ops.validate(
        conn_map,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message="conn_map must be of shape "
        + "(n_components, n_modes, n_channels, n_channels), "
        + "(n_modes, n_channels, n_channels) or (n_channels, n_channels)",
    )

    # Number of components and modes
    n_components = conn_map.shape[0]
    n_modes = conn_map.shape[1]
    n_channels = conn_map.shape[2]

    # Validatation
    if isinstance(percentile, float) or isinstance(percentile, int):
        percentile = percentile * np.ones([n_components, n_modes])

    percentile = array_ops.validate(
        percentile,
        correct_dimensionality=2,
        allow_dimensions=[0, 1],
        error_message="percentile must be of shape "
        + "(n_components, n_modes), (n_modes,) or float",
    )

    # Copy the original connectivity map
    c = conn_map.copy()

    # Subtract the mean
    if n_modes == 1:
        subtract_mean = False
    if subtract_mean:
        c -= np.average(c, axis=1, weights=mean_weights, keepdims=True)

    # Take absolute value
    if absolute_value:
        c = abs(c)

    # Set diagonal to nan
    c[:, :, range(n_channels), range(n_channels)] = np.nan

    # Are the connectivity matrices symmetric?
    c_is_symmetric = array_ops.check_symmetry(c, precision=1e-6)
    m, n = np.triu_indices(n_channels, k=1)

    # Which edges are greater than the threshold?
    edges = np.zeros(
        [n_components, n_modes, n_channels, n_channels],
        dtype=bool,
    )
    for i in range(n_components):
        for j in range(n_modes):
            if c_is_symmetric[i, j]:
                # We have a symmetric connectivity matrix
                # Threshold the upper triangle and copy to the lower triangle
                edges[i, j, m, n] = c[i, j, m, n] > np.nanpercentile(
                    c[i, j, m, n], percentile[i, j]
                )
                edges[i, j, n, m] = edges[i, j, m, n]
            else:
                # We have a directed connectivity matrix
                # Threshold each entry independently
                edges[i, j] = c[i, j] > np.nanpercentile(c[i, j], percentile[i, j])

    if return_edges:
        return np.squeeze(edges)

    # Zero the connections that are below the threshold
    conn_map[~edges] = 0

    return np.squeeze(conn_map)


def separate_edges(conn_map):
    """Separate positive and negative edges in a connectivity map.

    Parameters
    ----------
    conn_map : np.ndarray
        Connectivity map. Any shape.

    Returns
    -------
    pos_conn_map : np.ndarray
        Connectivity map with positive edges. Shape is the same as
        :code:`conn_map`.
    neg_conn_map : np.ndarray
        Connectivity map with negative edges. Shape is the same as
        :code:`conn_map`.
    """
    pos_conn_map = conn_map.copy()
    neg_conn_map = conn_map.copy()
    pos_conn_map[pos_conn_map < 0] = 0
    neg_conn_map[neg_conn_map > 0] = 0
    return pos_conn_map, neg_conn_map


def spectral_reordering(corr_mat):
    """Spectral re-ordering for correlation matrices.

    Parameters
    ----------
    corr_mat : np.ndarray
        Correlation matrix. Shape must be (n_channels, n_channels).

    Returns
    -------
    reorder_corr_mat : np.ndarray
        Re-ordered correlation matrix. Shape is (n_channels, n_channels).
    order : np.ndarray
        New ordering. Shape is (n_channels,).
    """
    # Add one to make all entries postive
    C = corr_mat + 1

    # Compute Q
    Q = -C
    np.fill_diagonal(Q, 0)
    Q -= np.sum(Q, axis=0)

    # Compute t
    t = np.diag(1.0 / np.sqrt(np.sum(C, axis=0)))

    # Compute D
    D = np.dot(np.dot(t, Q), t)

    # Eigevalue decomposition
    D, W = np.linalg.eig(D)
    v = W[:, 1]

    # Scale v
    v = np.dot(t, v)

    # Find permutations
    order = np.argsort(v)

    # Reorder
    reorder_corr_mat = corr_mat[order, :][:, order]

    return reorder_corr_mat, order


def save(
    connectivity_map,
    parcellation_file,
    filename=None,
    component=None,
    threshold=0,
    plot_kwargs=None,
    axes=None,
    combined=False,
    titles=None,
    n_rows=1,
):
    """Save connectivity maps as image files.

    This function is a wrapper for `nilearn.plotting.plot_connectome \
    <https://nilearn.github.io/stable/modules/generated/nilearn\
    .plotting.plot_connectome.html>`_.

    Parameters
    ----------
    connectivity_map : np.ndarray
        Matrices containing connectivity strengths to plot.
        Shape must be (n_components, n_modes, n_channels, n_channels),
        (n_modes, n_channels, n_channels) or (n_channels, n_channels).
    parcellation_file : str
        Name of parcellation file used.
    filename : str, optional
        Output filename. If :code:`None` is passed then the image is
        shown on screen. Must have extension :code:`.png`, :code:`.pdf`
        or :code:`.svg`.
    component : int, optional
        Spectral component to save.
    threshold : float or np.ndarray, optional
        Threshold to determine which connectivity to show. Should be between 0
        and 1. If a :code:`float` is passed the same threshold is used for all
        modes. Otherwise, threshold should be a numpy array of shape
        (n_modes,).
    plot_kwargs : dict, optional
        Keyword arguments to pass to the nilearn plotting function.
    axes : list, optional
        List of matplotlib axes to plot the connectivity maps on.
    combined : bool, optional
        Should the connectivity maps be combined on the same figure?
        The combined image is always shown on screen (for Juptyer notebooks).
        Note if :code:`True` is passed, the individual images will be deleted.
    titles : list, optional
        List of titles for each connectivity map. Only used if
        :code:`combined=True`.
    n_rows : int, optional
        Number of rows in the combined image. Only used if :code:`combined=True`.

    Examples
    --------
    Change colormap and views::

        connectivity.save(
            ...,
            plot_kwargs={
                "edge_cmap": "red_transparent_full_alpha_range",
                "display_mode": "lyrz",
            },
        )
    """
    # Suppress INFO messages from nibabel
    logging.getLogger("nibabel.global").setLevel(logging.ERROR)

    # Validation
    connectivity_map = np.copy(connectivity_map)
    error_message = (
        "Dimensionality of connectivity_map must be 3 or 4, "
        + f"got ndim={connectivity_map.ndim}."
    )
    connectivity_map = array_ops.validate(
        connectivity_map,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message=error_message,
    )

    if isinstance(threshold, float) or isinstance(threshold, int):
        threshold = np.array([threshold] * connectivity_map.shape[1])

    if np.any(threshold > 1) or np.any(threshold < 0):
        raise ValueError("threshold must be between 0 and 1.")

    if component is None:
        component = 0

    # Load parcellation file
    parcellation = Parcellation(parcellation_file)

    # Select the component we're plotting
    conn_map = connectivity_map[component]

    # Fill diagonal with zeros to help with the colorbar limits
    for c in conn_map:
        np.fill_diagonal(c, 0)

    # Default plotting settings
    default_plot_kwargs = {"node_size": 10, "node_color": "black"}

    # Loop through each connectivity map
    n_modes = conn_map.shape[0]
    axes = axes or [None] * n_modes
    output_files = []
    for i in trange(n_modes, desc="Saving images"):
        # Overwrite keyword arguments if passed
        kwargs = override_dict_defaults(default_plot_kwargs, plot_kwargs)

        # Output filename
        if filename is None:
            output_file = None
        else:
            output_file = "{fn.parent}/{fn.stem}{i:0{w}d}{fn.suffix}".format(
                fn=Path(filename), i=i, w=len(str(n_modes))
            )

        # If all connections are zero don't add a colourbar
        kwargs["colorbar"] = np.any(
            conn_map[i][~np.eye(conn_map[i].shape[-1], dtype=bool)] != 0
        )

        # Plot maps
        plotting.plot_connectome(
            conn_map[i],
            parcellation.roi_centers(),
            edge_threshold=f"{threshold[i] * 100}%",
            output_file=output_file,
            axes=axes[i],
            **kwargs,
        )
        output_files.append(output_file)

    if combined:
        # Combine the images
        if filename is None:
            raise ValueError("filename must be passed to save the combined image.")

        n_columns = -(n_modes // -n_rows)
        titles = titles or [None] * n_modes
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 5, n_rows * 5))
        for i, ax in enumerate(axes.flatten()):
            ax.axis("off")
            if i < n_modes:
                ax.imshow(plt.imread(output_files[i]))
                ax.set_title(titles[i], fontsize=20)
        fig.tight_layout()
        fig.savefig(filename)

        # Remove the individual images
        for output_file in output_files:
            os.remove(output_file)


def save_interactive(
    connectivity_map,
    parcellation_file,
    filename=None,
    component=None,
    threshold=0,
    plot_kwargs=None,
):
    """Save connectivity maps as interactive HTML plots.

    This function is a wrapper for `nilearn.plotting.view_connectome \
    <https://nilearn.github.io/stable/modules/generated/nilearn\
    .plotting.view_connectome.html>`_

    Parameters
    ----------
    connectivity_map : np.ndarray
        Matrices containing connectivity strengths to plot.
        Shape must be (n_components, n_modes, n_channels, n_channels),
        (n_modes, n_channels, n_channels) or (n_channels, n_channels).
    parcellation_file : str
        Name of parcellation file used.
    filename : str, optional
        Output filename. If :code:`None` is passed then the image is
        shown on screen. Must have extension :code:`.html`.
    component : int, optional
        Spectral component to save.
    threshold : float or np.ndarray, optional
        Threshold to determine which connectivity to show. Should be between 0
        and 1. If a :code:`float` is passed the same threshold is used for all
        modes. Otherwise, threshold should be a numpy array of shape
        (n_modes,).
    plot_kwargs : dict, optional
        Keyword arguments to pass to the nilearn plotting function.
    """

    # Validation
    connectivity_map = np.copy(connectivity_map)
    error_message = (
        "Dimensionality of connectivity_map must be 3 or 4, "
        + f"got ndim={connectivity_map.ndim}."
    )
    connectivity_map = array_ops.validate(
        connectivity_map,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message=error_message,
    )

    if isinstance(threshold, float) or isinstance(threshold, int):
        threshold = np.array([threshold] * connectivity_map.shape[1])

    if np.any(threshold > 1) or np.any(threshold < 0):
        raise ValueError("threshold must be between 0 and 1.")

    if component is None:
        component = 0

    # Load parcellation file
    parcellation = Parcellation(parcellation_file)

    # Select the component we're plotting
    conn_map = connectivity_map[component]

    # Fill diagonal with zeros to help with the colorbar limits
    for c in conn_map:
        np.fill_diagonal(c, 0)

    # Default plotting settings
    default_plot_kwargs = {"node_size": 10, "node_color": "black"}

    # Loop through each connectivity map
    n_modes = conn_map.shape[0]
    for i in trange(n_modes, desc="Saving images"):
        # Overwrite keyword arguments if passed
        kwargs = override_dict_defaults(default_plot_kwargs, plot_kwargs)

        # Output filename
        if filename is None:
            output_file = None
        else:
            output_file = "{fn.parent}/{fn.stem}{i:0{w}d}{fn.suffix}".format(
                fn=Path(filename), i=i, w=len(str(n_modes))
            )

        # The colour bar range is determined by the max value in the matrix
        # we zero the diagonal so it's not included
        np.fill_diagonal(conn_map[i], val=0)

        # Plot thick lines for the connections
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 12

        # Plot maps
        connectome = plotting.view_connectome(
            conn_map[i],
            parcellation.roi_centers(),
            edge_threshold=f"{threshold[i] * 100}%",
            **kwargs,
        )
        if filename is not None:
            connectome.save_as_html(output_file)
        else:
            return connectome
