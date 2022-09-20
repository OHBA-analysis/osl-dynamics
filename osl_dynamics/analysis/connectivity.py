"""Functions to calculate and plot network connectivity.

"""

from pathlib import Path

import numpy as np
from nilearn import plotting
from tqdm import trange

from osl_dynamics import array_ops
from osl_dynamics.analysis import gmm
from osl_dynamics.analysis.spectral import get_frequency_args_range
from osl_dynamics.utils.parcellation import Parcellation
from osl_dynamics.utils.misc import override_dict_defaults


def covariance_from_spectra(
    frequencies,
    power_spectra,
    components=None,
    frequency_range=None,
):
    """Calculates variance from power spectra.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis of the PSDs. Only used if frequency_range is given.
    power_spectra : np.ndarray
        Power/cross spectra for each channel. Shape is (n_modes, n_channels,
        n_channels, n_f).
    components : np.ndarray
        Spectral components. Shape is (n_components, n_f).
    frequency_range : list
        Frequency range to integrate the PSD over (Hz). Default is full range.

    Returns
    -------
    covar : np.ndarray
        Covariance over a frequency band for each component of each mode.
        Shape is (n_components, n_modes, n_channels, n_channels).
    """

    # Validation
    error_message = (
        "A (n_channels, n_channels, n_frequency_bins), "
        + "(n_modes, n_channels, n_channels, n_frequency_bins) or "
        + "(n_subjects, n_modes, n_channels, n_channels, n_frequency_bins) "
        + "array must be passed."
    )
    power_spectra = array_ops.validate(
        power_spectra,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )

    if components is not None and frequency_range is not None:
        raise ValueError(
            "Only one of the arguments components or frequency range can be passed."
        )

    if frequency_range is not None and frequencies is None:
        raise ValueError(
            "If frequency_range is passed, frequenices must also be passed."
        )

    # Dimensions
    n_subjects, n_modes, n_channels, n_channels, n_f = power_spectra.shape
    if components is None:
        n_components = 1
    else:
        n_components = components.shape[0]

    # Calculate connectivity maps for each subject
    covar = []
    for i in range(n_subjects):
        # Cross spectral densities
        csd = power_spectra[i].reshape(-1, n_f)
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
                [f_min, f_max] = get_frequency_args_range(frequencies, frequency_range)
                c = np.sum(csd[..., f_min:f_max], axis=-1)

        c = c.reshape(n_components, n_modes, n_channels, n_channels)
        covar.append(c)

    return np.squeeze(covar)


def mean_coherence_from_spectra(
    frequencies,
    coherence,
    components=None,
    frequency_range=None,
):
    """Calculates mean coherence from spectra.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis of the PSDs. Only used if frequency_range is given.
    coherence : np.ndarray
        Coherence for each channel. Shape is (n_modes, n_channels,
        n_channels, n_f).
    components : np.ndarray
        Spectral components. Shape is (n_components, n_f).
    frequency_range : list
        Frequency range to integrate the PSD over (Hz).

    Returns
    -------
    coh : np.ndarray
        Mean coherence over a frequency band for each component of each mode.
        Shape is (n_components, n_modes, n_channels, n_channels). Axis of
        length 1 are squeezed.
    """

    # Validation
    error_message = (
        "a 3D numpy array (n_channels, n_channels, n_frequency_bins) "
        + "or 4D numpy array (n_modes, n_channels, n_channels, "
        + "n_frequency_bins) must be passed for spectra."
    )
    coherence = array_ops.validate(
        coherence,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )

    if components is not None and frequency_range is not None:
        raise ValueError(
            "Only one of the arguments components or frequency range can be passed."
        )

    if frequency_range is not None and frequencies is None:
        raise ValueError(
            "If frequency_range is passed, frequenices must also be passed."
        )

    # Dimensions
    n_subjects, n_modes, n_channels, n_channels, n_f = coherence.shape
    if components is None:
        n_components = 1
    else:
        n_components = components.shape[0]

    # Calculate connectivity for each subject
    c = []
    for i in range(n_subjects):

        # Concatenate over modes
        coh = coherence[i].reshape(-1, n_f)

        # Same frequencies give nan coherences, replace with zero
        coh = np.nan_to_num(coh)

        if components is not None:
            # Coherence for each spectral component
            coh = components @ coh.T
            for j in range(n_components):
                coh[j] /= np.sum(components[j])

        else:
            # Mean over the given frequency range
            if frequency_range is None:
                coh = np.mean(coh, axis=-1)
            else:
                [f_min, f_max] = get_frequency_args_range(frequencies, frequency_range)
                coh = np.mean(coh[..., f_min:f_max], axis=-1)

        coh = coh.reshape(n_components, n_modes, n_channels, n_channels)
        c.append(coh)

    return np.squeeze(c)


def gmm_threshold(
    conn_map,
    subtract_mean=False,
    standardize=False,
    one_component_percentile=0,
    n_sigma=0,
    sklearn_kwargs=None,
    filename=None,
    plot_kwargs=None,
):
    """Threshold a connectivity matrix using the GMM method.

    Wrapper for connectivity.fit_gmm() and connectivity.threshold().

    Parameters
    ----------
    conn_map : np.ndarray
        Connectivity matrix. Shape must be (n_components, n_modes, n_channels,
        n_channels) or (n_modes, n_channels, n_channels) or (n_channels, n_channesl).
    subtract_mean : bool
        Should we subtract the mean over modes before fitting a GMM?
    standardize : bool
        Should we standardize the input to the GMM?
    one_component_percentile : float
        Percentile threshold if only one component is found.
        Should be a between 0 and 100. E.g. for the 95th percentile,
        one_component_percentile=95.
    n_sigma : float
        Number of standard deviations of the 'off' component the mean
        of the 'on' component must be for the fit to be considered to
        have two components.
    sklearn_kwargs : dict
        Dictionary of keyword arguments to pass to
        sklearn.mixture.BayesianGaussianMixture().
    filename : str
        Filename to save fit to.
    plot_kwargs : dict
        Dictionary of keyword arguments to pass to
        osl_dynamics.utils.plotting.plot_gmm().

    Returns
    -------
    conn_map : np.ndarray
        Thresholded connectivity matrix.
    """
    percentile = fit_gmm(
        conn_map,
        subtract_mean,
        standardize,
        one_component_percentile,
        n_sigma,
        sklearn_kwargs,
        filename,
        plot_kwargs,
    )
    conn_map = threshold(conn_map, percentile, subtract_mean, return_edges=False)
    return conn_map


def fit_gmm(
    conn_map,
    subtract_mean=False,
    standardize=False,
    one_component_percentile=0,
    n_sigma=0,
    sklearn_kwargs=None,
    filename=None,
    plot_kwargs=None,
):
    """Fit a two component Gaussian mixture model to connections to identify a
    threshold.

    Parameters
    ----------
    conn_map : np.ndarray
        Connectivity map.
    subtract_mean : bool
        Should we subtract the mean over modes before fitting a GMM?
    standardize : bool
        Should we standardize the input to the GMM?
    one_component_percentile : float
        Percentile threshold if only one component is found.
        Should be a between 0 and 100. E.g. for the 95th percentile,
        one_component_percentile=95.
    n_sigma : float
        Number of standard deviations of the 'off' component the mean
        of the 'on' component must be for the fit to be considered to
        have two components.
    sklearn_kwargs : dict
        Dictionary of keyword arguments to pass to
        sklearn.mixture.BayesianGaussianMixture().
    filename : str
        Filename to save fit to.
    plot_kwargs : dict
        Dictionary of keyword arguments to pass to
        osl_dynamics.utils.plotting.plot_gmm().

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
        sklearn_kwargs = {"max_iter": 5000, "n_init": 10}

    # Number of components, modes and channels
    n_components = conn_map.shape[0]
    n_modes = conn_map.shape[1]
    n_channels = conn_map.shape[2]

    # Mean over modes
    mean_conn_map = np.mean(conn_map, axis=1)

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
            percentiles[i, j] = gmm.fit_gaussian_mixture(
                c,
                bayesian=True,
                standardize=standardize,
                sklearn_kwargs=sklearn_kwargs,
                one_component_percentile=one_component_percentile,
                n_sigma=n_sigma,
                plot_filename=plot_filename,
                plot_kwargs=plot_kwargs,
                print_message=False,
            )

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
        Connectivity matrix to threshold.
        Can be (n_components, n_modes, n_channels, n_channels),
        (n_modes, n_channels, n_channels) or (n_channels, n_channels).
    percentile : float or np.ndarray
        Percentile to threshold with. Should be between 0 and 100.
        Can be a numpy array of shape (n_modes,) or (n_components, n_modes).
    subtract_mean : bool
        Should we subtract the mean over modes before thresholding?
        The thresholding is only done to identify edges, the values returned in
        conn_map are not mean subtracted.
    mean_weights : np.ndarray
        Weights when calculating the mean over modes.
    absolute_value : bool
        Should we take the absolute value before thresholding?
        The thresholding is only done to identify edges, the values returned in
        conn_map are not absolute values. If subtract_mean=True, the mean is
        subtracted before the absolute value.
    return_edges : bool
        Should we return a boolean array for whether edges are above the
        threshold?

    Returns
    -------
    conn_map : np.ndarray
        Connectivity matrix with connections below the threshold set to zero.
        Or a boolean array if return_edges=True.
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

    if isinstance(percentile, float) or isinstance(percentile, int):
        percentile = percentile * np.ones([n_components, n_modes])

    if percentile.ndim == 1:
        # A (n_modes,) array has been passed, add the n_components dimension
        percentile = percentile[np.newaxis, ...]

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

    # Which edges are greater than the threshold?
    edges = np.empty([n_components, n_modes, n_channels, n_channels], dtype=bool)
    for i in range(n_components):
        for j in range(n_modes):
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
        Connectivity map.

    Returns
    -------
    pos_conn_map : np.ndarray
        Connectivity map with positive edges.
    neg_conn_map : np.ndarray
        Connectivity map with negative edges.
    """
    pos_conn_map = conn_map.copy()
    neg_conn_map = conn_map.copy()
    pos_conn_map[pos_conn_map < 0] = 0
    neg_conn_map[neg_conn_map > 0] = 0
    return pos_conn_map, neg_conn_map


def save(
    connectivity_map,
    filename,
    parcellation_file,
    component=None,
    threshold=0,
    glassbrain=False,
    plot_kwargs=None,
):
    """Save connectivity maps.

    If glassbrain=True, this function is a wrapper for
    nilearn.plotting.view_connectome, otherwise this function is a wrapper for
    nilearn.plotting.plot_connectome.

    Parameters
    ----------
    connectivity_map : np.ndarray
        Matrices containing connectivity strengths to plot.
        Shape must be (n_modes, n_channels, n_channels) or (n_channels, n_channels).
    filename : str
        Output filename.
    parcellation_file : str
        Name of parcellation file used.
    component : int
        Spectral component to save.
    threshold : float or np.ndarray
        Threshold to determine which connectivity to show. Should be between 0 and 1.
        If a float is passed the same threshold is used for all modes. Otherwise,
        threshold should be a numpy array of shape (n_modes,).
    glassbrain : bool
        Sholud we create a 3D glass brain plot (as an interactive HTML file)
        or a 2D image plot (as a png, pdf, svg, etc. file).
    plot_kwargs : dict
        Keyword arguments to pass to the nilearn plotting function.
    """
    # Validation
    if glassbrain and Path(filename).suffix != ".html":
        raise ValueError(
            "If glassbrain=True then filename must have a .html extension."
        )

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
    conn_map = np.copy(connectivity_map[component])

    # Default plotting settings
    default_plot_kwargs = {"node_size": 10, "node_color": "black"}

    # Loop through each connectivity map
    n_modes = conn_map.shape[0]
    for i in trange(n_modes, desc="Saving images", ncols=98):

        # Overwrite keyword arguments if passed
        kwargs = override_dict_defaults(default_plot_kwargs, plot_kwargs)

        # Output filename
        output_file = "{fn.parent}/{fn.stem}{i:0{w}d}{fn.suffix}".format(
            fn=Path(filename), i=i, w=len(str(n_modes))
        )

        if glassbrain:
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
            connectome.save_as_html(output_file)

        else:
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
                **kwargs,
            )
