"""Functions to manipulate and calculate statistics for inferred mode/state time courses.

"""

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from osl_dynamics import array_ops, analysis
from osl_dynamics.inference import metrics
from osl_dynamics.utils.misc import override_dict_defaults


def argmax_time_courses(
    alpha,
    concatenate=False,
    n_modes=None,
):
    """Calculates mode time courses.

    Hard classifies the modes so that only one mode is active.

    Parameters
    ----------
    alpha : list, np.ndarray
        Mode mixing factors with shape (n_subjects, n_samples, n_modes)
        or (n_samples, n_modes).
    concatenate : bool
        If alpha is a list, should we concatenate the mode time course?
    n_modes : int
        Number of modes there should be.

    Returns
    -------
    stcs : list, np.ndarray
        Mode time courses.
    """
    if isinstance(alpha, list):
        if n_modes is None:
            n_modes = alpha[0].shape[1]
        stcs = [a.argmax(axis=1) for a in alpha]
        stcs = [array_ops.get_one_hot(stc, n_states=n_modes) for stc in stcs]
        if len(stcs) == 1:
            stcs = stcs[0]
        elif concatenate:
            stcs = np.concatenate(stcs)
    elif alpha.ndim == 3:
        if n_modes is None:
            n_modes = alpha.shape[-1]
        stcs = alpha.argmax(axis=2)
        stcs = np.array([array_ops.get_one_hot(stc, n_states=n_modes) for stc in stcs])
        if len(stcs) == 1:
            stcs = stcs[0]
        elif concatenate:
            stcs = np.concatenate(stcs)
    else:
        if n_modes is None:
            n_modes = alpha.shape[1]
        stcs = alpha.argmax(axis=1)
        stcs = array_ops.get_one_hot(stcs, n_states=n_modes)
    return stcs


def gmm_time_courses(
    time_course,
    logit_transform=True,
    standardize=True,
    p_value=None,
    filename=None,
    sklearn_kwargs=None,
):
    """Fit a two component GMM on the mode time courses to get a binary
    time course.

    Parameters
    ----------
    time_course : list of np.ndarray or np.ndarray
        Mode time courses.
    logit_transform : bool
        Should we logit transform the mode time course?
    standardize : bool
        Should we standardize the mode time course?
    p_value : float
        Used to determine a threshold. We ensure the data points assigned
        to the 'on' component have a probability of less than p_value of
        belonging to the 'off' component.
    filename : str
        Path to directory to plot the GMM fit plots.
    sklearn_kwargs : dict
        keyword arguments for sklearn's GaussianMixture.

    Returns
    -------
    gmm_time_course : list of np.ndarray or np.ndarray
        GMM fitted mode time courses with binary entries.
    """

    if isinstance(time_course, list):
        # Extract positions of discontinuities
        discontinuities = [tc.shape[0] for tc in time_course]
        time_course = np.concatenate(time_course)
    else:
        discontinuities = None

    n_modes = time_course.shape[1]

    # Initialise an array to hold the gmm thresholded time course
    gmm_time_course = np.empty(time_course.shape, dtype=int)

    # Loop over modes
    for mode in range(n_modes):
        a = time_course[:, mode]

        # GMM plot filename
        if filename is not None:
            plot_filename = "{fn.parent}/{fn.stem}{mode:0{w}d}{fn.suffix}".format(
                fn=Path(filename),
                mode=mode,
                w=len(str(n_modes)),
            )
        else:
            plot_filename = None

        # Fit the GMM
        default_sklearn_kwargs = {"max_iter": 5000, "n_init": 3}
        sklearn_kwargs = override_dict_defaults(default_sklearn_kwargs, sklearn_kwargs)
        threshold = analysis.gmm.fit_gaussian_mixture(
            a,
            logit_transform=logit_transform,
            standardize=standardize,
            p_value=p_value,
            sklearn_kwargs=sklearn_kwargs,
            plot_filename=plot_filename,
            print_message=False,
        )
        print(f"GMM theshold {mode}: {threshold}")
        gmm_time_course[:, mode] = a > threshold

    if discontinuities is None:
        return gmm_time_course

    return np.split(gmm_time_course, np.cumsum(discontinuities))


def correlate_modes(mode_time_course_1, mode_time_course_2):
    """Calculate the correlation matrix between modes in two mode time courses.

    Given two mode time courses, calculate the correlation between each pair of modes
    in the mode time courses. The output for each value in the matrix is the value
    numpy.corrcoef(mode_time_course_1, mode_time_course_2)[0, 1].

    Parameters
    ----------
    mode_time_course_1 : numpy.ndarray
    mode_time_course_2 : numpy.ndarray

    Returns
    -------
    correlation_matrix : numpy.ndarray
    """
    correlation = np.zeros((mode_time_course_1.shape[1], mode_time_course_2.shape[1]))
    for i, mode1 in enumerate(mode_time_course_1.T):
        for j, mode2 in enumerate(mode_time_course_2.T):
            correlation[i, j] = np.corrcoef(mode1, mode2)[0, 1]
    return correlation


def match_covariances(*covariances, comparison="rv_coefficient", return_order=False):
    """Matches covariances.

    Can match covariances using the Frobenius norm, correlation or RV coefficient.
    Each matrix must be 3D: (n_modes, n_channels, n_channels).

    Parameters
    ----------
    covarainces: list of numpy.ndarray
        Covariance matrices to match.
        Each covariance must be (n_modes, n_channel, n_channels).
    comparison : str
        Either 'rv_coefficient', 'correlation' or 'frobenius'.
        Default is 'rv_coefficient'.
    return_order : bool
        Should we return the order instead of the covariances?

    Returns
    -------
    covs : tuple
        Matched covariances or order.
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
                    F[j, k] = -metrics.rv_coefficient(
                        [covariances[i][k], covariances[0][j]]
                    )[0, 1]
        order = linear_sum_assignment(F)[1]

        # Add the ordered matrix to the list
        matched_covariances.append(covariances[i][order])

    if return_order:
        return order
    else:
        return tuple(matched_covariances)


def match_modes(*mode_time_courses, return_order=False):
    """Find correlated modes between mode time courses.

    Given N mode time courses and using the first given mode time course as a basis,
    find the best matches for modes between all of the mode time courses. Once found,
    the mode time courses are returned with the modes reordered so that the modes
    match.

    Given two arrays with columns ABCD and CBAD, both will be returned with modes in
    the order ABCD.

    Parameters
    ----------
    mode_time_courses : list of numpy.ndarray

    Returns
    -------
    matched_mode_time_courses : list of numpy.ndarray
        Matched mode time courses or order.
    """
    # If the mode time courses have different length we only use the first n_samples
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
        matches = linear_sum_assignment(-correlation)
        matched_mode_time_courses.append(mode_time_course[:n_samples, matches[1]])
        orders.append(matches[1])

    if return_order:
        return orders
    else:
        return matched_mode_time_courses


def reduce_mode_time_course(mode_time_course):
    """Remove empty modes from a mode time course.

    If a mode has no activation in the mode time course, remove the column
    corresponding to that mode.

    Parameters
    ----------
    mode_time_course: numpy.ndarray

    Returns
    -------
    reduced_mode_time_course: numpy.ndarray
        A mode time course with no modes with no activation.
    """
    return mode_time_course[:, ~np.all(mode_time_course == 0, axis=0)]


def fractional_occupancies(state_time_course):
    """Wrapper for osl_dynamics.analysis.modes.fractional_occupancies."""
    return analysis.modes.fractional_occupancies(state_time_course)


def mean_lifetimes(state_time_course, sampling_frequency=None):
    """Wrapper for osl_dynamics.analysis.modes.mean_lifetimes."""
    return analysis.modes.mean_lifetimes(state_time_course, sampling_frequency)


def mean_intervals(state_time_course, sampling_frequency=None):
    """Wrapper for osl_dynamics.analysis.modes.mean_intervals."""
    return analysis.modes.mean_intervals(state_time_course, sampling_frequency)


def switching_rates(state_time_course, sampling_frequency=None):
    """Wrapper for osl_dynamics.analysis.modes.switching_rates."""
    return analysis.modes.switching_rates(state_time_course, sampling_frequency)
