"""Functions to calculate and plot network connectivity.

"""

from pathlib import Path

import numpy as np
from scipy import stats
from nilearn import plotting
from tqdm import trange

from osl_dynamics import array_ops
from osl_dynamics.analysis import gmm, static
from osl_dynamics.analysis.spectral import get_frequency_args_range
from osl_dynamics.utils.parcellation import Parcellation
from osl_dynamics.utils.misc import override_dict_defaults


def sliding_window_connectivity(
    data,
    window_length,
    step_size=None,
    conn_type="corr",
    concatenate=False,
):
    """Calculate sliding window connectivity.

    Parameters
    ----------
    data : list or np.ndarray
        Time series data. Shape must be (n_subjects, n_samples, n_channels)
        or (n_samples, n_channels).
    window_length : int
        Window length in samples.
    step_size : int
        Number of samples to slide the window along the time series.
        If None is passed, then a 50% overlap is used.
    conn_type : str
        Metric to use to calculate pairwise connectivity in the network.
        Should "corr" for Pearson correlation or "cov" for covariance.
    concatenate : bool
        Should we concatenate the sliding window connectivities from each subject
        into one big time series?

    Returns
    -------
    sliding_window_conn : list or np.ndarray
        Time series of connectivity matrices. Shape is (n_subjects, n_windows,
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

    # Calculate sliding window connectivity for each subject
    sliding_window_conn = []
    for i in trange(len(data), desc="Calculating connectivity", ncols=98):
        n_samples = data[i].shape[0]
        n_channels = data[i].shape[1]

        # Define indices of time points that start windows
        time_idx = range(0, n_samples, step_size)
        n_windows = n_samples // step_size

        # Trim the data to only include complete window
        data[i] = data[i][: n_windows * window_length]

        # Preallocate an array to hold moving average values
        swc = np.empty([n_windows, n_channels, n_channels], dtype=np.float32)

        # Compute connectivity matrix for each window
        for k in range(n_windows):
            j = time_idx[k]
            window = data[i][j : j + window_length]
            swc[k] = metric(window, rowvar=False)

        # Add to list to return
        sliding_window_conn.append(swc)

    if concatenate or len(sliding_window_conn) == 1:
        sliding_window_conn = sliding_window_conn[0]

    return sliding_window_conn


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
        n_channels, n_freq).
    components : np.ndarray
        Spectral components. Shape is (n_components, n_freq).
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
        "A (n_channels, n_channels, n_freq), "
        + "(n_modes, n_channels, n_channels, n_freq) or "
        + "(n_subjects, n_modes, n_channels, n_channels, n_freq) "
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
    n_subjects, n_modes, n_channels, n_channels, n_freq = power_spectra.shape
    if components is None:
        n_components = 1
    else:
        n_components = components.shape[0]

    # Calculate connectivity maps for each subject
    covar = []
    for i in range(n_subjects):
        # Cross spectral densities
        csd = power_spectra[i].reshape(-1, n_freq)
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
        n_channels, n_freq).
    components : np.ndarray
        Spectral components. Shape is (n_components, n_freq).
    frequency_range : list
        Frequency range to integrate the PSD over (Hz).

    Returns
    -------
    coh : np.ndarray
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
    n_subjects, n_modes, n_channels, n_channels, n_freq = coherence.shape
    if components is None:
        n_components = 1
    else:
        n_components = components.shape[0]

    # Calculate connectivity for each subject
    c = []
    for i in range(n_subjects):

        # Concatenate over modes
        coh = coherence[i].reshape(-1, n_freq)

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


def eigenvectors(conn_map, n_eigenvectors=1, absolute_value=False, as_network=False):
    """Calculate eigenvectors of a connectivity matrix.

    Parameters
    ----------
    conn_map : np.ndarray
        Connectivity matrix. Shape must be (..., n_channels, n_channels).
    n_eigenvectors : int
        Number of eigenvectors to include.
    absolute_value : bool
        Should we take the absolute value of the connectivity matrix before
        calculating the eigen decomposition?
    as_network : bool
        Should we return a matrix?

    Returns
    -------
    eigenvectors : np.ndarray.
        Eigenvectors. Shape is (n_eigenvectors, ..., n_channels, n_channels)
        if as_network=True, otherwise it is (n_eigenvectors, ..., n_channels).
        If n_eigenvectors=1, the first dimension is removed.
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
    sklearn_kwargs={},
    show=False,
    filename=None,
    plot_kwargs={},
):
    """Threshold a connectivity matrix using the GMM method.

    Wrapper for connectivity.fit_gmm() and connectivity.threshold().

    Parameters
    ----------
    conn_map : np.ndarray
        Connectivity matrix. Shape must be (n_components, n_modes, n_channels,
        n_channels) or (n_modes, n_channels, n_channels) or (n_channels, n_channels).
    subtract_mean : bool
        Should we subtract the mean over modes before fitting a GMM?
    mean_weights: np.ndarray
        Numpy array with weightings for each mode/state to use to calculate the mean.
        Default is equal weighting.
    standardize : bool
        Should we standardize the input to the GMM?
    p_value : float
        Used to determine a threshold. We ensure the data points assigned
        to the 'on' component have a probability of less than p_value of
        belonging to the 'off' component.
    keep_positive_only : bool
        Should we only keep positive values to fit a GMM to?
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
    show : bool
        Should we show the GMM fit to the distribution of conn_map.
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
    sklearn_kwargs={},
    show=False,
    filename=None,
    plot_kwargs={},
):
    """Fit a two component Gaussian mixture model to connections to identify a
    threshold.

    Parameters
    ----------
    conn_map : np.ndarray
        Connectivity map. Shape must be (n_components, n_modes, n_channels, n_channels)
        or (n_modes, n_channels, n_channels) or (n_channels, n_channels).
    subtract_mean : bool
        Should we subtract the mean over modes before fitting a GMM?
    mean_weights: np.ndarray
        Numpy array with weightings for each mode/state to use to calculate the mean.
        Default is equal weighting.
    standardize : bool
        Should we standardize the input to the GMM?
    p_value : float
        Used to determine a threshold. We ensure the data points assigned
        to the 'on' component have a probability of less than p_value of
        belonging to the 'off' component.
    keep_positive_only : bool
        Should we only keep positive values to fit a GMM to?
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
        sklearn.mixture.GaussianMixture(). Default is
        {"max_iter": 5000, "n_init": 10}.
    show : bool
        Should we show the GMM fit to the distribution of conn_map.
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
                print_message=False,
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
        Connectivity matrix to threshold.
        Can be (n_components, n_modes, n_channels, n_channels),
        (n_modes, n_channels, n_channels) or (n_channels, n_channels).
    percentile : float or np.ndarray
        Percentile to threshold with. Should be between 0 and 100.
        Can be a numpy array of shape (n_components, n_modes), (n_modes,)
        or a float.
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
    parcellation_file,
    filename=None,
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
    parcellation_file : str
        Name of parcellation file used.
    filename : str
        Output filename.
        Optional, if None is passed then the image is shown on screen.
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
    if filename is not None:
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
        if filename is None:
            output_file = None
        else:
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
            if filename is not None:
                connectome.save_as_html(output_file)
            else:
                return connectome

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


def spectral_reordering(corr_mat):
    """Spectral re-ordering for correlation matrices.

    Parameters
    ----------
    corr_mat : np.ndarray
        Correlation matrix.

    Returns
    -------
    reorder_corr_mat : np.ndarray
        Re-ordered correlation matrix.
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
    perm = np.argsort(v)

    # Reorder
    return corr_mat[perm, :][:, perm]
