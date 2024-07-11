"""Functions to manipulate and calculate statistics for inferred mode/state
time courses.

"""

from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange
from scipy import cluster, spatial, optimize
from sklearn.cluster import AgglomerativeClustering

from osl_dynamics import analysis, array_ops
from osl_dynamics.inference import metrics
from osl_dynamics.utils import plotting
from osl_dynamics.utils.misc import override_dict_defaults


def argmax_time_courses(alpha, concatenate=False, n_modes=None):
    """Hard classifies a time course using an argmax operation.

    Parameters
    ----------
    alpha : list or np.ndarray
        Mode mixing factors or state probabilities. Shape must be
        (n_sessions, n_samples, n_modes) or (n_samples, n_modes).
    concatenate : bool, optional
        If :code:`alpha` is a :code:`list`, should we concatenate the
        time courses?
    n_modes : int, optional
        Number of modes/states there should be. Useful if there are
        modes/states which never activate.

    Returns
    -------
    argmax_tcs : list or np.ndarray
        Argmax time courses. Shape is (n_sessions, n_samples, n_modes)
        or (n_samples, n_modes).
    """
    if isinstance(alpha, list):
        if n_modes is None:
            n_modes = alpha[0].shape[1]
        tcs = [a.argmax(axis=1) for a in alpha]
        tcs = [array_ops.get_one_hot(tc, n_states=n_modes) for tc in tcs]
        if len(tcs) == 1:
            tcs = tcs[0]
        elif concatenate:
            tcs = np.concatenate(tcs)
    elif alpha.ndim == 3:
        if n_modes is None:
            n_modes = alpha.shape[-1]
        tcs = alpha.argmax(axis=2)
        tcs = np.array(
            [array_ops.get_one_hot(tc, n_states=n_modes) for tc in tcs],
        )
        if len(tcs) == 1:
            tcs = tcs[0]
        elif concatenate:
            tcs = np.concatenate(tcs)
    else:
        if n_modes is None:
            n_modes = alpha.shape[1]
        tcs = alpha.argmax(axis=1)
        tcs = array_ops.get_one_hot(tcs, n_states=n_modes)
    return tcs


def gmm_time_courses(
    alpha,
    logit_transform=True,
    standardize=True,
    p_value=None,
    filename=None,
    sklearn_kwargs=None,
    plot_kwargs=None,
):
    """Fit a two-component GMM to time courses to get a binary time course.

    Parameters
    ----------
    alpha : list of np.ndarray or np.ndarray
        Mode time courses. Shape must be (n_sessions, n_samples, n_modes) or
        (n_samples, n_modes).
    logit_transform : bool, optional
        Should we logit transform the mode time course?
    standardize : bool, optional
        Should we standardize the mode time course?
    p_value : float, optional
        Used to determine a threshold. We ensure the data points assigned
        to the 'on' component have a probability of less than :code:`p_value`
        of belonging to the 'off' component.
    filename : str, optional
        Path to directory to plot the GMM fit plots.
    sklearn_kwargs : dict, optional
        Keyword arguments to pass to `sklean.mixture.GaussianMixture \
        <https://scikit-learn.org/stable/modules/generated/\
        sklearn.mixture.GaussianMixture.html>`_.
    plot_kwargs : dict, optional
        Dictionary of keyword arguments to pass to
        `osl_dynamics.utils.plotting.plot_gmm \
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/\
        osl_dynamics/utils/plotting/index.html#osl_dynamics.utils.plotting\
        .plot_gmm>`_.

    Returns
    -------
    gmm_tcs : list of np.ndarray or np.ndarray
        GMM time courses with binary entries. Shape is
        (n_sessions, n_samples, n_modes) or (n_samples, n_modes).
    """
    if plot_kwargs is None:
        plot_kwargs = {}

    if not isinstance(alpha, list):
        alpha = [alpha]

    n_sessions = len(alpha)
    n_modes = alpha[0].shape[1]

    gmm_tcs = []
    gmm_metrics = []
    for sub in trange(n_sessions, desc="Fitting GMMs"):
        # Initialise an array to hold the gmm thresholded time course
        gmm_tc = np.empty(alpha[sub].shape, dtype=int)
        gmm_metric = []

        # Loop over modes
        for mode in range(n_modes):
            a = alpha[sub][:, mode]

            # Fit the GMM
            default_sklearn_kwargs = {"max_iter": 5000, "n_init": 3}
            sklearn_kwargs = override_dict_defaults(
                default_sklearn_kwargs, sklearn_kwargs
            )
            threshold, metrics = analysis.gmm.fit_gaussian_mixture(
                a,
                logit_transform=logit_transform,
                standardize=standardize,
                p_value=p_value,
                sklearn_kwargs=sklearn_kwargs,
                return_statistics=True,
                log_message=False,
            )
            gmm_tc[:, mode] = a > threshold
            gmm_metric.append(metrics)

        # Add to list containing session-specific time courses and
        # component metrics
        gmm_tcs.append(gmm_tc)
        gmm_metrics.append(gmm_metric)

    # Visualise session-specific time courses in one plot per mode
    avg_threshold = [
        np.mean([gmm_metrics[s][m]["threshold"] for s in range(n_sessions)])
        for m in range(n_modes)
    ]
    if filename:
        for mode in range(n_modes):
            # GMM plot filename
            if filename is not None:
                plot_filename = "{fn.parent}/{fn.stem}{mode:0{w}d}{fn.suffix}".format(
                    fn=Path(filename),
                    mode=mode,
                    w=len(str(n_modes)),
                )
            else:
                plot_filename = None

            # session-specific GMM plots per mode
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
            for sub in range(n_sessions):
                metric = gmm_metrics[sub][mode]
                plotting.plot_gmm(
                    metric["data"],
                    metric["amplitudes"],
                    metric["means"],
                    metric["stddevs"],
                    legend_loc=None,
                    ax=ax,
                    **plot_kwargs,
                )
            ax.set_title(f"Averaged Threshold = {avg_threshold[mode]:.3}")
            handles, labels = plt.gca().get_legend_handles_labels()
            label_class = dict(zip(labels, handles))
            ax.legend(label_class.values(), label_class.keys(), loc=1)
            ax.axvline(avg_threshold[mode], color="black", linestyle="--")
            plotting.save(fig, plot_filename)
            plotting.close()

    return gmm_tcs


def correlate_modes(mode_time_course_1, mode_time_course_2):
    """Calculate the correlation matrix between modes in two mode time courses.

    Given two mode time courses, calculate the correlation between each pair of
    modes in the mode time courses. The output for each value in the matrix is
    the value :code:`numpy.corrcoef(mode_time_course_1, \
    mode_time_course_2)[0, 1]`.

    Parameters
    ----------
    mode_time_course_1 : np.ndarray
        Mode time course. Shape must be (n_samples, n_modes).
    mode_time_course_2 : np.ndarray
        Mode time course. Shape must be (n_samples, n_modes).

    Returns
    -------
    correlation_matrix : np.ndarray
        Correlation matrix. Shape is (n_modes, n_modes).
    """
    correlation = np.zeros(
        (mode_time_course_1.shape[1], mode_time_course_2.shape[1]),
    )
    for i, mode1 in enumerate(mode_time_course_1.T):
        for j, mode2 in enumerate(mode_time_course_2.T):
            correlation[i, j] = np.corrcoef(mode1, mode2)[0, 1]
    return correlation


def match_covariances(
    *covariances,
    comparison="rv_coefficient",
    return_order=False,
):
    """Matches covariances.

    Parameters
    ----------
    covariances : tuple of np.ndarray
        Covariance matrices to match.
        Each covariance must be (n_modes, n_channel, n_channels).
    comparison : str, optional
        Either :code:`'rv_coefficient'`, :code:`'correlation'` or
        :code:`'frobenius'`. Default is :code:`'rv_coefficient'`.
    return_order : bool, optional
        Should we return the order instead of the covariances?

    Returns
    -------
    matched_covariances : tuple or list of np.ndarray
        Matched covariances of shape (n_channels, n_channels) or order if
        :code:`return_order=True`.

    Examples
    --------
    Reorder the matrices directly:

    >>> covs1, covs2 = match_covariances(covs1, covs2, comparison="correlation")

    Just get the reordering:

    >>> orders = match_covariances(covs1, covs2, comparison="correlation", return_order=True)
    >>> print(orders[0])  # order for covs1 (always unchanged)
    >>> print(orders[1])  # order for covs2
    """
    # Validation
    for matrix in covariances[1:]:
        if matrix.shape != covariances[0].shape:
            raise ValueError("Matrices must have the same shape.")

    if comparison not in ["frobenius", "correlation", "rv_coefficient"]:
        raise ValueError(
            "Comparison must be 'rv_coefficient', 'correlation' or 'frobenius'."
        )

    # Number of arguments and number of matrices in each argument passed
    n_args = len(covariances)
    n_matrices = covariances[0].shape[0]

    # Calculate the similarity between matrices
    F = np.empty([n_matrices, n_matrices])
    matched_covariances = [covariances[0]]
    orders = [np.arange(covariances[0].shape[0])]
    for i in range(1, n_args):
        for j in range(n_matrices):
            # Find the matrix that is most similar to matrix j
            for k in range(n_matrices):
                if comparison == "frobenius":
                    A = abs(
                        np.diagonal(covariances[i][k]) - np.diagonal(covariances[0][j])
                    )
                    F[j, k] = np.linalg.norm(A)
                elif comparison == "correlation":
                    F[j, k] = -np.corrcoef(
                        covariances[i][k].flatten(), covariances[0][j].flatten()
                    )[0, 1]
                else:
                    F[j, k] = -metrics.pairwise_rv_coefficient(
                        np.array([covariances[i][k], covariances[0][j]])
                    )[0, 1]
        order = optimize.linear_sum_assignment(F)[1]

        # Add the ordered matrix to the list
        matched_covariances.append(covariances[i][order])
        orders.append(order)

    if return_order:
        return orders
    else:
        return tuple(matched_covariances)


def match_vectors(*vectors, comparison="correlation", return_order=False):
    """Matches vectors.

    Parameters
    ----------
    vectors : tuple of np.ndarray
        Sets of vectors to match.
        Each variable must be shape (n_vectors, n_channels).
    comparison : str, optional
        Must be :code:`'correlation'` or :code:`'cosine_similarity'`.
    return_order : bool, optional
        Should we return the order instead of the matched vectors?

    Returns
    -------
    matched_vectors : tuple of np.ndarray
        Set of matched vectors of shape (n_vectors, n_channels)
        or order if :code:`return_order=True`.

    Examples
    --------
    Reorder the vectors directly:

    >>> v1, v2 = match_vectors(v1, v2, comparison="correlation")

    Just get the reordering:

    >>> orders = match_vectors(v1, v2, comparison="correlation", return_order=True)
    >>> print(orders[0])  # order for v1 (always unchanged)
    >>> print(orders[1])  # order for v2
    """
    # Validation
    for vector in vectors[1:]:
        if vector.shape != vectors[0].shape:
            raise ValueError("Vectors must have the same shape.")

    if comparison not in ["correlation", "cosine_similarity"]:
        raise ValueError("Comparison must be 'correlation' or 'cosine_similarity'.")

    # Number of arguments and number of vectors in each argument passed
    n_args = len(vectors)
    n_vectors = vectors[0].shape[0]

    # Calculate the similarity between vectors
    F = np.empty([n_vectors, n_vectors])
    matched_vectors = [vectors[0]]
    orders = [np.arange(vectors[0].shape[0])]
    for i in range(1, n_args):
        for j in range(n_vectors):
            # Find the vector that is most similar to vector j
            for k in range(n_vectors):
                if comparison == "correlation":
                    F[j, k] = -np.corrcoef(vectors[i][k], vectors[0][j])[0, 1]
                elif comparison == "cosine_similarity":
                    F[j, k] = -(
                        1 - spatial.distance.cosine(vectors[i][k], vectors[0][j])
                    )
        order = optimize.linear_sum_assignment(F)[1]

        # Add the ordered vector to the list
        matched_vectors.append(vectors[i][order])
        orders.append(order)

    if return_order:
        return orders
    else:
        return tuple(matched_vectors)


def match_modes(*mode_time_courses, return_order=False):
    """Find correlated modes between mode time courses.

    Given N mode time courses and using the first given mode time course as a
    basis, find the best matches for modes between all of the mode time courses.
    Once found, the mode time courses are returned with the modes reordered so
    that the modes match.

    Given two arrays with columns ABCD and CBAD, both will be returned with
    modes in the order ABCD.

    Parameters
    ----------
    mode_time_courses : list of np.ndarray
        Mode time courses. Each time course must be (n_samples, n_modes).
    return_order : bool, optional
        Should we return the order instead of the mode time courses.

    Returns
    -------
    matched_mode_time_courses : tuple or list of np.ndarray
        Matched mode time courses of shape (n_samples, n_modes) or order
        if :code:`return_order=True`.

    Examples
    --------
    Reorder the modes directly:

    >>> alp1, alp2 = match_modes(alp1, alp2)

    Just get the reordering:

    >>> orders = match_modes(alp1, alp2, return_order=True)
    >>> print(orders[0])  # order for alp1 (always unchanged)
    >>> print(orders[1])  # order for alp2
    """
    # If the mode time courses have different length we only use the
    # first n_samples
    n_samples = min([stc.shape[0] for stc in mode_time_courses])

    # Match time courses based on correlation
    matched_mode_time_courses = [mode_time_courses[0][:n_samples]]
    orders = [np.arange(mode_time_courses[0].shape[1])]
    for mode_time_course in mode_time_courses[1:]:
        correlation = correlate_modes(
            mode_time_courses[0][:n_samples], mode_time_course[:n_samples]
        )
        correlation = np.nan_to_num(
            np.nan_to_num(correlation, nan=np.nanmin(correlation) - 1)
        )
        matches = optimize.linear_sum_assignment(-correlation)
        matched_mode_time_courses.append(mode_time_course[:n_samples, matches[1]])
        orders.append(matches[1])

    if return_order:
        return orders
    else:
        return matched_mode_time_courses


def reduce_state_time_course(state_time_course):
    """Remove states that don't activate from a state time course.

    Parameters
    ----------
    state_time_course: np.ndarray
        State time course. Shape must be (n_samples, n_states).

    Returns
    -------
    reduced_state_time_course: np.ndarray
        Reduced state time course. Shape is (n_samples, n_reduced_states).
    """
    return state_time_course[:, ~np.all(state_time_course == 0, axis=0)]


def fractional_occupancies(state_time_course):
    """Wrapper for `analysis.modes.fractional_occupancies \
    <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
    /analysis/modes/index.html#osl_dynamics.analysis.modes\
    .fractional_occupancies>`_."""
    return analysis.modes.fractional_occupancies(state_time_course)


def mean_lifetimes(state_time_course, sampling_frequency=None):
    """Wrapper for `analysis.modes.mean_lifetimes \
    <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
    /analysis/modes/index.html#osl_dynamics.analysis.modes.mean_lifetimes>`_."""
    return analysis.modes.mean_lifetimes(state_time_course, sampling_frequency)


def mean_intervals(state_time_course, sampling_frequency=None):
    """Wrapper for `analysis.modes.mean_intervals \
    <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
    /analysis/modes/index.html#osl_dynamics.analysis.modes.mean_intervals>`_."""
    return analysis.modes.mean_intervals(state_time_course, sampling_frequency)


def switching_rates(state_time_course, sampling_frequency=None):
    """Wrapper for `analysis.modes.switching_rates \
    <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
    /analysis/modes/index.html#osl_dynamics.analysis.modes\
    .switching_rates>`_."""
    return analysis.modes.switching_rates(state_time_course, sampling_frequency)


def mean_amplitudes(state_time_course, data):
    """Wrapper for `analysis.modes.mean_amplitudes \
    <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
    /analysis/modes/index.html#osl_dynamics.analysis.modes\
    .mean_amplitudes>`_."""
    return analysis.modes.mean_amplitudes(state_time_course, data)


def lifetime_statistics(state_time_course, sampling_frequency=None):
    """Wrapper for `analysis.modes.lifetime_statistics \
    <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
    /analysis/modes/index.html#osl_dynamics.analysis.modes\
    .lifetime_statistics>`_."""
    return analysis.modes.lifetime_statistics(state_time_course, sampling_frequency)


def fano_factor(state_time_course, window_length, sampling_frequency=1.0):
    """Wrapper for `analysis.modes.fano_factor \
    <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
    /analysis/modes/index.html#osl_dynamics.analysis.modes.fano_factor>`_."""
    return analysis.modes.fano_factor(
        state_time_course, window_length, sampling_frequency
    )


def convert_to_mne_raw(
    alpha,
    raw,
    ch_names=None,
    n_embeddings=None,
    n_window=None,
    extra_chans="stim",
    verbose=False,
):
    """Convert a time series to an `MNE Raw \
    <https://mne.tools/stable/generated/mne.io.Raw.html>`_ object.

    Parameters
    ----------
    alpha : np.ndarray
        Time series containing raw data. Shape must be (n_samples, n_modes).
    raw : mne.io.Raw or str
        Raw object to extract info from. If a :code:`str` is passed, it must
        be the path to a fif file containing the Raw object.
    ch_names : list, optional
        Name for each channel. Defaults to :code:`alpha_0, ...,
        alpha_{n_modes-1}`.
    n_embeddings : int, optional
        Number of embeddings that was used to prepare time-delay embedded
        training data.
    n_window : int, optional
        Number of samples used to smooth amplitude envelope data.
    extra_chans : str or list of str, optional
        Extra channel types to add to the Raw object.
    verbose : bool, optional
        Should we print a verbose?

    Returns
    -------
    alpha_raw : mne.io.Raw
        `MNE Raw <https://mne.tools/stable/generated/mne.io.Raw.html>`_ object
        for :code:`alpha`.
    """
    # Validation
    if extra_chans is None:
        extra_chans = []
    if isinstance(extra_chans, str):
        extra_chans = [extra_chans]

    # How many time points from the start of parcellated data should we remove?
    if n_embeddings is not None and n_window is not None:
        raise ValueError("Cannot pass both n_embeddings and n_window.")
    n_trim = n_embeddings or n_window or 1
    n_trim = n_trim // 2

    # Load the Raw object
    if isinstance(raw, str):
        raw = mne.io.read_raw_fif(raw, verbose=verbose)

    # Get time indices excluding bad segments from raw
    _, times = raw.get_data(
        reject_by_annotation="omit", return_times=True, verbose=verbose
    )
    indices = raw.time_as_index(times, use_rounding=True)

    # Remove time points lost due to time delay embedding
    indices = indices[n_trim:]

    # Create an array for the full time series including bad segments
    n_samples = raw.times.shape[0]
    n_channels = alpha.shape[1]
    alpha_ = np.zeros([n_samples, n_channels], dtype=np.float32)

    # Trim the indices we lost when we separate the time series into sequences
    indices = indices[: alpha.shape[0]]

    # Fill in the value for the time series for non-bad segments
    alpha_[indices] = alpha

    # Create info object
    if ch_names is None:
        ch_names = [f"alpha_{ch}" for ch in range(n_channels)]
    alpha_info = mne.create_info(
        ch_names=ch_names,
        ch_types="misc",
        sfreq=raw.info["sfreq"],
        verbose=verbose,
    )

    # Create Raw object
    alpha_raw = mne.io.RawArray(alpha_.T, alpha_info, verbose=verbose)

    # Copy timing info
    alpha_raw.set_meas_date(raw.info["meas_date"])
    alpha_raw.__dict__["_first_samps"] = raw.__dict__["_first_samps"]
    alpha_raw.__dict__["_last_samps"] = raw.__dict__["_last_samps"]
    alpha_raw.__dict__["_cropped_samp"] = raw.__dict__["_cropped_samp"]

    # Copy annotations from raw
    alpha_raw.set_annotations(raw._annotations)

    # Add extra channels
    for extra_chan in extra_chans:
        if extra_chan in raw:
            chan_raw = raw.copy().pick(extra_chan)
            chan_data = chan_raw.get_data()
            chan_info = mne.create_info(
                chan_raw.ch_names,
                raw.info["sfreq"],
                [extra_chan] * chan_data.shape[0],
            )
            chan_raw = mne.io.RawArray(chan_data, chan_info, verbose=verbose)
            alpha_raw.add_channels([chan_raw], force_update_info=True)

    return alpha_raw


def reweight_alphas(alpha, covs):
    """Re-weight mixing coefficients to account for the magnitude of the mode covariances.

    Parameters
    ----------
    alpha : list of np.ndarray or np.ndarray
        Raw mixing coefficients. Shape must be (n_sessions, n_samples, n_modes)
        or (n_samples, n_modes).
    covs : np.ndarray
        Mode covariances. Shape must be (n_modes, n_channels, n_channels).

    Returns
    -------
    reweighted_alpha : list of np.ndarray or np.ndarray
        Re-weighted mixing coefficients. Shape is the same as :code:`alpha`.
    """
    return reweight_mtc(alpha, covs, "covariance")


def reweight_mtc(mtc, params, params_type):
    """Re-weight mixing coefficients to account for the magnitude of
    observation model parameters.

    Parameters
    ----------
    mtc : list of np.ndarray or np.ndarray
        Raw mixing coefficients. Shape must be (n_sessions, n_samples, n_modes)
        or (n_samples, n_modes).
    params : np.ndarray
        Observation model parameters.
        Shape must be (n_modes, n_channels, n_channels).
    params_type : str
        Observation model parameters type. Either 'covariance' or 'correlation'.

    Returns
    -------
    reweighted_mtc : list of np.ndarray
        Re-weighted mixing coefficients. Shape is the same as :code:`mtc`.
    """
    if isinstance(mtc, np.ndarray):
        mtc = [mtc]

    if params_type == "covariance":
        weights = np.trace(params, axis1=1, axis2=2)
    elif params_type == "correlation":
        m, n = np.tril_indices(params.shape[-1], -1)
        weights = np.sum(np.abs(params[:, m, n]), axis=-1)
    else:
        raise ValueError("params_type must be 'covariance' or 'correlation'.")

    reweighted_mtc = [x * weights[np.newaxis, :] for x in mtc]
    reweighted_mtc = [x / np.sum(x, axis=1, keepdims=True) for x in reweighted_mtc]

    if len(reweighted_mtc) == 1:
        reweighted_mtc = reweighted_mtc[0]

    return reweighted_mtc


def average_runs(alpha, n_clusters=None, return_cluster_info=False):
    """Average the state probabilities from different runs using hierarchical clustering.

    Parameters
    ----------
    alpha : list of list of np.ndarray or list of np.ndarray
        State probabilities. Shape must be (n_runs, n_sessions, n_samples,
        n_states) or (n_runs, n_samples, n_states).
    n_clusters : int, optional
        Number of clusters to fit. Defaults to the largest number of states
        in alpha.
    return_cluster_info : bool, optional
        Should we return information describing the clustering?

    Returns
    -------
    average_alpha : list of np.ndarray or np.ndarray
        State probabilities averaged over runs. Shape is (n_sessions, n_states).
    cluster_info : dict
        Clustering info. Only returned if :code:`return_cluster_info=True`.
        This is a dictionary with keys :code:`'correlation'`,
        :code:`'dissimiarity'`, :code:`'ids'` and :code:`'linkage'`.

    See Also
    --------
    S. Alonso and D. Vidaurre, "Towards stability of dynamic FC estimates in
    neuroimaging and electrophysiology: solutions and limits" `bioRxiv (2023): \
    2023-01 <https://www.biorxiv.org/content/10.1101/2023.01.18.524539v2>`_.
    """
    if not isinstance(alpha, list):
        raise ValueError(
            "alpha must be a list of lists (of numpy arrays) or list of numpy arrays."
        )
    if isinstance(alpha[0], np.ndarray):
        alpha = [[a] for a in alpha]

    # Number of runs and length of each session's data
    n_runs = len(alpha)
    n_session_samples = [a.shape[0] for a in alpha[0]]

    # Use the largest number of states as the number of clusters to find
    if n_clusters is None:
        n_clusters = max([a.shape[-1] for a in alpha[0]])

    # Concatenate over arrays, gives (n_runs, n_samples, n_states) array
    alpha = [np.concatenate(a, axis=0) for a in alpha]

    # Turn into a (n_runs * n_states, n_samples) array
    alpha_ = []
    for i in range(n_runs):
        for j in range(alpha[i].shape[-1]):
            alpha_.append(alpha[i][:, j])
    alpha = np.array(alpha_, dtype=np.float32).T

    # Calculate correlation between all pairwise state probability time courses
    corr = np.corrcoef(alpha, rowvar=False)

    # Convert correlation to a dis-similarity measure
    dissimilarity = 1 - corr

    # Hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters, linkage="ward")
    cluster_ids = clustering.fit_predict(dissimilarity)

    # Average alphas in each cluster
    average_alpha = []
    for i in range(n_clusters):
        a = np.mean(alpha[:, cluster_ids == i], axis=-1)
        average_alpha.append(a)
    average_alpha = np.array(average_alpha, dtype=np.float32).T

    # Split average alphas back into session-specific time courses
    average_alpha = np.split(average_alpha, np.cumsum(n_session_samples[:-1]))

    if return_cluster_info:
        # Create a dictionary containing the clustering info
        linkage = cluster.hierarchy.linkage(dissimilarity, method="ward")
        cluster_info = {
            "correlation": corr,
            "dissimilarity": dissimilarity,
            "ids": cluster_ids,
            "linkage": linkage,
        }
        return average_alpha, cluster_info

    else:
        return average_alpha
