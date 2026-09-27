"""
Functions to manipulate and calculate statistics for inferred mode/state
time courses.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange
from scipy import cluster, spatial, optimize
from sklearn.cluster import AgglomerativeClustering

from osl_dynamics.analysis import post_hoc
from osl_dynamics.inference import metrics
from osl_dynamics.utils import array_ops, sklearn_wrappers, plotting
from osl_dynamics.utils.misc import override_dict_defaults

_logger = logging.getLogger("osl-dynamics")


def argmax_time_courses(
    alpha: Union[List[np.ndarray], np.ndarray],
    concatenate: bool = False,
    n_modes: Optional[int] = None,
) -> Union[List[np.ndarray], np.ndarray]:
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
    alpha: Union[List[np.ndarray], np.ndarray],
    logit_transform: bool = True,
    standardize: bool = True,
    p_value: Optional[float] = None,
    filename: Optional[str] = None,
    sklearn_kwargs: Optional[Dict] = None,
    plot_kwargs: Optional[Dict] = None,
) -> List[np.ndarray]:
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
        :func:`osl_dynamics.utils.plotting.plot_gmm`.

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
            threshold, metrics = sklearn_wrappers.fit_gaussian_mixture(
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


def correlate_modes(
    mode_time_course_1: np.ndarray, mode_time_course_2: np.ndarray
) -> np.ndarray:
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
    n_modes = mode_time_course_1.shape[1]
    correlation = np.corrcoef(mode_time_course_1, mode_time_course_2, rowvar=False)
    return correlation[:n_modes, n_modes:]


def _match_to_first(items: Tuple, similarity: Callable) -> List[np.ndarray]:
    """Order of each item's modes that best matches the first item's.

    Parameters
    ----------
    items : tuple
        Items to match, e.g. sets of vectors or covariances.
    similarity : callable
        :code:`similarity(first, item)` must return a (n_modes, n_modes)
        array, larger for more similar, with the first item's modes along
        the rows.

    Returns
    -------
    orders : list of np.ndarray
        Order for each item (always unchanged for the first item). Found with
        the Hungarian algorithm, maximising the total similarity.
    """
    orders = [np.arange(len(items[0]))]
    for item in items[1:]:
        orders.append(optimize.linear_sum_assignment(-similarity(items[0], item))[1])
    return orders


def match_covariances(
    *covariances: np.ndarray,
    comparison: str = "rv_coefficient",
    return_order: bool = False,
) -> Union[Tuple[np.ndarray, ...], List[np.ndarray]]:
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

    def similarity(first, covs):
        n = len(first)
        if comparison == "frobenius":
            return -np.linalg.norm(first[:, None] - covs[None], axis=(2, 3))
        if comparison == "correlation":
            flat = np.concatenate([first, covs]).reshape(2 * n, -1)
            return np.corrcoef(flat)[:n, n:]
        rv = metrics.pairwise_rv_coefficient(np.concatenate([first, covs]))
        return rv[:n, n:]

    orders = _match_to_first(covariances, similarity)
    if return_order:
        return orders
    return tuple(c[order] for c, order in zip(covariances, orders))


def match_vectors(
    *vectors: np.ndarray,
    comparison: str = "correlation",
    return_order: bool = False,
) -> Union[Tuple[np.ndarray, ...], List[np.ndarray]]:
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

    def similarity(first, v):
        if comparison == "correlation":
            return np.corrcoef(first, v)[: len(first), len(first) :]
        return 1 - spatial.distance.cdist(first, v, metric="cosine")

    orders = _match_to_first(vectors, similarity)
    if return_order:
        return orders
    return tuple(v[order] for v, order in zip(vectors, orders))


def match_modes(
    *mode_time_courses: np.ndarray,
    return_order: bool = False,
) -> Union[List[np.ndarray], List[np.ndarray]]:
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

    # Match time courses based on correlation, with modes along the rows. A
    # mode that never changes has no correlation: treat it as the least similar
    def similarity(first, mtc):
        correlation = correlate_modes(first.T, mtc.T)
        # (and if no mode changes, all modes as equally similar)
        return np.nan_to_num(np.nan_to_num(correlation, nan=np.nanmin(correlation) - 1))

    mode_time_courses = [mtc[:n_samples] for mtc in mode_time_courses]
    orders = _match_to_first([mtc.T for mtc in mode_time_courses], similarity)
    if return_order:
        return orders
    return [mtc[:, order] for mtc, order in zip(mode_time_courses, orders)]


def reduce_state_time_course(state_time_course: np.ndarray) -> np.ndarray:
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


def fractional_occupancies(state_time_course: np.ndarray):
    """Wrapper for :func:`osl_dynamics.analysis.post_hoc.fractional_occupancies`."""
    return post_hoc.fractional_occupancies(state_time_course)


def mean_lifetimes(
    state_time_course: np.ndarray, sampling_frequency: Optional[float] = None
):
    """Wrapper for :func:`osl_dynamics.analysis.post_hoc.mean_lifetimes`."""
    return post_hoc.mean_lifetimes(state_time_course, sampling_frequency)


def mean_intervals(
    state_time_course: np.ndarray, sampling_frequency: Optional[float] = None
):
    """Wrapper for :func:`osl_dynamics.analysis.post_hoc.mean_intervals`."""
    return post_hoc.mean_intervals(state_time_course, sampling_frequency)


def switching_rates(
    state_time_course: np.ndarray, sampling_frequency: Optional[float] = None
):
    """Wrapper for :func:`osl_dynamics.analysis.post_hoc.switching_rates`."""
    return post_hoc.switching_rates(state_time_course, sampling_frequency)


def mean_amplitudes(state_time_course: np.ndarray, data: np.ndarray):
    """Wrapper for :func:`osl_dynamics.analysis.post_hoc.mean_amplitudes`."""
    return post_hoc.mean_amplitudes(state_time_course, data)


def lifetime_statistics(
    state_time_course: np.ndarray, sampling_frequency: Optional[float] = None
):
    """Wrapper for :func:`osl_dynamics.analysis.post_hoc.lifetime_statistics`."""
    return post_hoc.lifetime_statistics(state_time_course, sampling_frequency)


def fano_factor(
    state_time_course: np.ndarray, window_length: int, sampling_frequency: float = 1.0
):
    """Wrapper for :func:`osl_dynamics.analysis.post_hoc.fano_factor`."""
    return post_hoc.fano_factor(state_time_course, window_length, sampling_frequency)


def convert_to_mne_raw(
    alpha: np.ndarray,
    raw: Union[mne.io.Raw, str],
    ch_names: Optional[List[str]] = None,
    n_embeddings: Optional[int] = None,
    n_window: Optional[int] = None,
    extra_chans: Union[str, List[str], None] = "stim",
    verbose: bool = False,
) -> mne.io.Raw:
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
    from osl_dynamics.meeg.parcellation import (
        convert_to_mne_raw as _convert_to_mne_raw,
    )

    # Load the Raw object
    if isinstance(raw, str):
        raw = mne.io.read_raw_fif(raw, verbose=verbose)

    # How many time points from the start of parcellated data should we remove?
    n_trim = 0
    if n_embeddings is not None:
        n_trim += n_embeddings // 2
    if n_window is not None:
        n_trim += n_window // 2

    # Get time indices excluding bad segments from raw
    _, times = raw.get_data(
        reject_by_annotation="omit", return_times=True, verbose=verbose
    )
    indices = raw.time_as_index(times, use_rounding=True)

    # Remove time points lost due to time delay embedding
    indices = indices[n_trim:]

    # Trim the indices we lost when we separate the time series into sequences
    indices = indices[: alpha.shape[0]]

    # Create full-length array with bad segments as zeros
    n_channels = alpha.shape[1]
    data = np.zeros([n_channels, len(raw.times)], dtype=np.float32)
    data[:, indices] = alpha.T

    # Default channel names
    if ch_names is None:
        ch_names = [f"alpha_{ch}" for ch in range(n_channels)]

    return _convert_to_mne_raw(data, raw, ch_names=ch_names, extra_chans=extra_chans)


def reweight_alphas(
    alpha: Union[List[np.ndarray], np.ndarray],
    covs: np.ndarray,
) -> Union[List[np.ndarray], np.ndarray]:
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


def reweight_mtc(
    mtc: Union[List[np.ndarray], np.ndarray],
    params: np.ndarray,
    params_type: str,
) -> Union[List[np.ndarray], np.ndarray]:
    """Reweight mode time courses.

    Re-weight mixing coefficients to account for the magnitude of
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


def average_runs(
    alpha: Union[List[List[np.ndarray]], List[np.ndarray]],
    n_clusters: Optional[int] = None,
    return_cluster_info: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], Dict]]:
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
        raise TypeError(
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


def _mode_correlations(
    features: Union[np.ndarray, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Correlation between every pair of modes, pooled over runs.

    Parameters
    ----------
    features : np.ndarray or list of np.ndarray
        Features of each mode of each run. Shape must be (n_runs, n_modes,
        n_features) or a list of (n_modes, n_features) arrays.

    Returns
    -------
    corr : np.ndarray
        Correlation between the modes, after removing the mean over modes
        from each run's features. Shape is (n_total_modes, n_total_modes),
        where n_total_modes is the number of modes summed over runs.
    run : np.ndarray
        Run each mode belongs to. Shape is (n_total_modes,).
    modes : list of np.ndarray
        Indices into corr of each run's modes.
    """
    features = [np.asarray(f, dtype=np.float64) for f in features]
    if len(features) < 2:
        raise ValueError("features must contain at least two runs.")
    if any(f.ndim != 2 or f.shape[1] != features[0].shape[1] for f in features):
        raise ValueError(
            "each run's features must be (n_modes, n_features), with the "
            "same n_features for every run."
        )

    # Describe each mode by what distinguishes it from the other modes of
    # its run
    features = [f - f.mean(axis=0) for f in features]

    run = np.concatenate([np.full(len(f), i) for i, f in enumerate(features)])
    modes = [np.flatnonzero(run == i) for i in range(len(features))]
    return np.corrcoef(np.concatenate(features)), run, modes


def _mean_off_diagonal(matrix: np.ndarray) -> float:
    """Mean of the off-diagonal elements of a square matrix (NaN if 1x1)."""
    n = len(matrix)
    if n == 1:
        return np.nan
    return (matrix.sum() - np.trace(matrix)) / (n * (n - 1))


def _otsu_threshold(values: np.ndarray) -> float:
    """Threshold that best separates values into two groups (Otsu's method).

    Parameters
    ----------
    values : np.ndarray
        1D array of values.

    Returns
    -------
    threshold : float
        Midpoint between the two sorted values where the between-group
        variance is largest.
    """
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    n_low = np.arange(1, n)
    mean_low = cumsum[:-1] / n_low
    mean_high = (cumsum[-1] - cumsum[:-1]) / (n - n_low)
    between = n_low * (n - n_low) * (mean_low - mean_high) ** 2
    i = np.argmax(between)
    return float((values[i] + values[i + 1]) / 2)


def match_runs(
    features: Union[np.ndarray, List[np.ndarray]],
    threshold: Optional[float] = None,
    return_threshold: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Group the modes of different runs into networks.

    Training a model (e.g. DyNeMo or an HMM) multiple times on the same data
    gives modes (states) in a different order and, where the data do not
    constrain the solution well, a different set of modes. This function
    labels each mode of each run with the network it represents, so that
    runs can be compared network by network.

    Each mode is described by features, e.g. its power map. The mean over
    modes is removed from each run's features, so a mode is described by
    what distinguishes it from the other modes of its run. The modes of all
    runs are then clustered by average linkage on their correlation, with
    the constraint that two modes of the same run are never in the same
    network. Clustering stops when no two networks correlate more than
    :code:`threshold`.

    Parameters
    ----------
    features : np.ndarray or list of np.ndarray
        Features of each mode of each run. Shape must be (n_runs, n_modes,
        n_features) or a list of (n_modes, n_features) arrays. E.g. power
        maps computed from the mode covariances with
        :code:`osl_dynamics.analysis.post_hoc.raw_covariances`.
    threshold : float, optional
        Correlation above which two groups of modes are the same network.
        Defaults to the threshold that best separates (by Otsu's method) the
        correlation of each mode with its best match in every other run: a
        match is either the same network found again or a different network.
    return_threshold : bool, optional
        Should we return the threshold?

    Returns
    -------
    networks : np.ndarray or list of np.ndarray
        Network of each mode of each run. Shape is (n_runs, n_modes) or a
        list of (n_modes,) arrays, like :code:`features`. Networks are
        numbered by the number of runs they are found in, most first.
    threshold : float
        Correlation threshold. Only returned if :code:`return_threshold=True`.

    Examples
    --------
    Power maps from the mode covariances of each run of a TDE-DyNeMo model:

    >>> maps = []
    >>> for covs in covariances:
    ...     raw = post_hoc.raw_covariances(
    ...         covs, n_embeddings, pca_components, zero_lag=True
    ...     )
    ...     maps.append(np.diagonal(raw, axis1=-2, axis2=-1))
    >>> networks = match_runs(maps)
    """
    corr, run, modes = _mode_correlations(features)
    labels, threshold = _networks(corr, run, modes, threshold)
    networks = [labels[m] for m in modes]
    if isinstance(features, np.ndarray):
        networks = np.array(networks)

    if return_threshold:
        return networks, threshold
    return networks


def _networks(
    corr: np.ndarray,
    run: np.ndarray,
    modes: List[np.ndarray],
    threshold: Optional[float],
) -> Tuple[np.ndarray, float]:
    """Network of each mode, from the correlations of :func:`_mode_correlations`.

    See :func:`match_runs`. Returns the network of each mode (as indexed in
    corr) and the threshold.
    """
    if threshold is None:
        best_matches = [
            corr[i, m].max()
            for i in range(len(corr))
            for r, m in enumerate(modes)
            if r != run[i]
        ]
        threshold = _otsu_threshold(np.array(best_matches))

    # Average-linkage agglomerative clustering. Two modes of the same run
    # must not be in the same network: their similarity is -inf, which stays
    # -inf in any average, so clusters containing them are never merged.
    similarity = np.where(run[:, None] == run[None, :], -np.inf, corr)
    clusters = [[i] for i in range(len(corr))]
    while len(clusters) > 1:
        a, b = sorted(np.unravel_index(np.argmax(similarity), similarity.shape))
        if similarity[a, b] < threshold:
            break
        n_a, n_b = len(clusters[a]), len(clusters[b])
        merged = (n_a * similarity[a] + n_b * similarity[b]) / (n_a + n_b)
        similarity[a], similarity[:, a] = merged, merged
        similarity[a, a] = -np.inf
        clusters[a] += clusters.pop(b)
        similarity = np.delete(np.delete(similarity, b, axis=0), b, axis=1)

    # Number networks by the number of runs they are found in, then by how
    # well their modes agree
    clusters.sort(
        key=lambda c: (
            -len(c),
            -np.nan_to_num(_mean_off_diagonal(corr[np.ix_(c, c)]), nan=-np.inf),
        )
    )
    labels = np.empty(len(corr), dtype=int)
    for network, members in enumerate(clusters):
        labels[members] = network
    return labels, threshold


def run_families(
    networks: Union[np.ndarray, List[np.ndarray]],
) -> List[np.ndarray]:
    """Group runs that found the same networks.

    Parameters
    ----------
    networks : np.ndarray or list of np.ndarray
        Network of each mode of each run, as returned by
        :func:`match_runs`. Shape must be (n_runs, n_modes) or a list of
        (n_modes,) arrays.

    Returns
    -------
    families : list of np.ndarray
        Indices of the runs in each family: runs that found the same set of
        networks. Largest family first; families of the same size are in
        the order of their first run.
    """
    keys = [tuple(sorted(n)) for n in networks]
    families = {}
    for i, key in enumerate(keys):
        families.setdefault(key, []).append(i)
    return sorted(
        (np.array(f) for f in families.values()), key=lambda f: (-len(f), f[0])
    )


def select_run(
    features: Union[np.ndarray, List[np.ndarray]],
    free_energy: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    return_info: bool = False,
) -> Union[int, Tuple[int, Dict]]:
    """Select the run to analyse: the most typical run of the most common
    solution.

    When a model is trained multiple times, a common choice is the run with
    the lowest variational free energy. When the free energies of the runs
    are within the noise of one another, an alternative is to choose by the
    networks the runs found:

    1. Group the modes of all runs into networks (:func:`match_runs`).
    2. Group runs that found the same set of networks into families
       (:func:`run_families`).
    3. Take the largest family - the solution found most often. If two or
       more are as large, take the one with the lowest median free energy
       (or, if free energies are not given, the one whose runs agree best).
    4. Within it, take the medoid: the run whose modes correlate best, on
       average, with the same networks in the other runs of the family.

    The choice is only as reliable as the family sizes: with few runs,
    families are small and often equally large, and the tie-break decides.
    Train enough runs (e.g. 20) for the most common solution to stand out,
    and check the sizes of the families in :code:`info`.

    Parameters
    ----------
    features : np.ndarray or list of np.ndarray
        Features of each mode of each run. Shape must be (n_runs, n_modes,
        n_features) or a list of (n_modes, n_features) arrays. See
        :func:`match_runs`.
    free_energy : np.ndarray, optional
        Variational free energy of each run. Shape must be (n_runs,). Used
        to choose between families that are equally large. A NaN counts as
        the highest free energy.
    threshold : float, optional
        Correlation above which two groups of modes are the same network.
        See :func:`match_runs`.
    return_info : bool, optional
        Should we return information describing the selection?

    Returns
    -------
    run : int
        Index of the selected run.
    info : dict
        Only returned if :code:`return_info=True`. A dictionary with keys:

        - :code:`'networks'`: network of each mode of each run.
        - :code:`'threshold'`: correlation threshold used for the networks.
        - :code:`'network_runs'`: number of runs each network is found in.
        - :code:`'network_cohesion'`: mean correlation between the modes of
          each network (NaN for a network found in one run).
        - :code:`'families'`: run indices of each family, largest first.
        - :code:`'family_cohesion'`: mean similarity between the runs of
          each family (NaN for a family of one run).
        - :code:`'family_free_energy'`: median free energy of the runs of
          each family (NaN if :code:`free_energy` is not given).
        - :code:`'family'`: index of the selected family in
          :code:`'families'`.
        - :code:`'typicality'`: mean similarity of each run in the selected
          family to the others in it, in the order of its runs.

        The similarity between two runs of a family is the mean correlation
        of their modes over the networks they share.
    """
    corr, run, modes = _mode_correlations(features)
    if free_energy is not None:
        free_energy = np.asarray(free_energy, dtype=np.float64)
        if free_energy.shape != (len(modes),):
            raise ValueError("free_energy must have one value per run.")
        nan = np.isnan(free_energy)
        if nan.any():
            _logger.warning(
                f"The free energy is NaN for the runs with indices "
                f"{np.flatnonzero(nan).tolist()}: treating it as the highest."
            )
            free_energy = np.where(nan, np.inf, free_energy)
    labels, threshold = _networks(corr, run, modes, threshold)
    networks = [labels[m] for m in modes]
    families = run_families(networks)

    # Similarity between each pair of runs in a family: the mean over
    # networks of the correlation between their modes of that network. The
    # runs of a family share their networks, so sorting each run's modes by
    # network lines them up.
    def _similarity(family):
        by_network = np.array([modes[r][np.argsort(networks[r])] for r in family])
        s = np.mean([corr[np.ix_(m, m)] for m in by_network.T], axis=0)
        return (s + s.T) / 2  # exactly symmetric, so ties go to the first run

    similarities = [_similarity(f) for f in families]
    family_cohesion = np.array([_mean_off_diagonal(s) for s in similarities])
    family_free_energy = np.array(
        [np.nan if free_energy is None else np.median(free_energy[f]) for f in families]
    )

    # The largest family; if several are as large, the one with the lowest
    # median free energy or, without free energies, the most cohesive
    largest = [i for i, f in enumerate(families) if len(f) == len(families[0])]
    if free_energy is not None:
        family = min(largest, key=lambda i: family_free_energy[i])
    else:
        family = max(
            largest, key=lambda i: np.nan_to_num(family_cohesion[i], nan=-np.inf)
        )
    s = similarities[family]
    n = len(s)
    typicality = (s.sum(axis=1) - 1) / (n - 1) if n > 1 else np.array([np.nan])
    selected = int(families[family][np.nanargmax(typicality) if n > 1 else 0])

    if not return_info:
        return selected

    info = {
        "networks": (
            np.array(networks) if isinstance(features, np.ndarray) else networks
        ),
        "threshold": threshold,
        "network_runs": np.bincount(labels),
        "network_cohesion": np.array(
            [
                _mean_off_diagonal(corr[np.ix_(m, m)])
                for m in (np.flatnonzero(labels == k) for k in range(labels.max() + 1))
            ]
        ),
        "families": families,
        "family_cohesion": family_cohesion,
        "family_free_energy": family_free_energy,
        "family": family,
        "typicality": typicality,
    }
    return selected, info
