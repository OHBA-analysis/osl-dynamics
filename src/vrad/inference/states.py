"""Functions to manipulate and calculate statistics for inferred state time courses.

"""

import logging
from typing import List, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from vrad import array_ops
from vrad.inference import metrics
from vrad.utils.decorators import transpose

_logger = logging.getLogger("VRAD")
_rng = np.random.default_rng()


def time_courses(
    alpha: Union[list, np.ndarray],
    concatenate: bool = False,
    n_states: int = None,
) -> Union[list, np.ndarray]:
    """Calculates state time courses.

    Hard classifies the states so that only one state is active.

    Parameters
    ----------
    alpha : list, np.ndarray
        State mixing factors with shape (n_subjects, n_samples, n_states)
        or (n_samples, n_states).
    concatenate : bool
        If alpha is a list, should we concatenate the state time course?
        Optional, default is True.
    n_states : int
        Number of states there should be. Optional.

    Returns
    -------
    stcs : list, np.ndarray
        State time courses.
    """
    if isinstance(alpha, list):
        if n_states is None:
            n_states = alpha[0].shape[1]
        stcs = [a.argmax(axis=1) for a in alpha]
        stcs = [array_ops.get_one_hot(stc, n_states=n_states) for stc in stcs]
        if len(stcs) == 1:
            stcs = stcs[0]
        elif concatenate:
            stcs = np.concatenate(stcs)
    elif alpha.ndim == 3:
        if n_states is None:
            n_states = alpha.shape[-1]
        stcs = alpha.argmax(axis=2)
        stcs = np.array([array_ops.get_one_hot(stc, n_states=n_states) for stc in stcs])
        if len(stcs) == 1:
            stcs = stcs[0]
        elif concatenate:
            stcs = np.concatenate(stcs)
    else:
        if n_states is None:
            n_states = alpha.shape[1]
        stcs = alpha.argmax(axis=1)
        stcs = array_ops.get_one_hot(stcs, n_states=n_states)
    return stcs


@transpose
def correlate_states(
    state_time_course_1: np.ndarray, state_time_course_2: np.ndarray
) -> np.ndarray:
    """Calculate the correlation matrix between states in two state-time-courses.

    Given two state time courses, calculate the correlation between each pair of states
    in the state time courses. The output for each value in the matrix is the value
    numpy.corrcoef(state_time_course_1, state_time_course_2)[0, 1].

    Parameters
    ----------
    state_time_course_1: numpy.ndarray
    state_time_course_2: numpy.ndarray

    Returns
    -------
    correlation_matrix: numpy.ndarray
    """
    correlation = np.zeros((state_time_course_1.shape[1], state_time_course_2.shape[1]))
    for i, state1 in enumerate(state_time_course_1.T):
        for j, state2 in enumerate(state_time_course_2.T):
            correlation[i, j] = np.corrcoef(state1, state2)[0, 1]
    return correlation


def match_covariances(
    *covariances: np.ndarray, comparison="rv_coefficient", return_order: bool = False
) -> Tuple[np.ndarray]:
    """Matches covariances.

    Can match covariances using the Frobenius norm, correlation or RV coefficient.
    Each matrix must be 3D: (n_states, n_channels, n_channels).

    Parameters
    ----------
    covarainces: list of numpy.ndarray
        Covariance matrices to match.
        Each covariance must be (n_states, n_channel, n_channels).
    comparison : str
        Either 'rv_coefficient', 'correlation' or 'frobenius'.
        Optional, default is 'rv_coefficient'.
    return_order : bool
        Should we return the order instead of the covariances?
        Optional, default is False.

    Returns
    -------
    tuple
        Matched covariances.
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


@transpose
def match_states(
    *state_time_courses: np.ndarray, return_order: bool = False
) -> List[np.ndarray]:
    """Find correlated states between state time courses.

    Given N state time courses and using the first given state time course as a basis,
    find the best matches for states between all of the state time courses. Once found,
    the state time courses are returned with the states reordered so that the states
    match.

    Given two arrays with columns ABCD and CBAD, both will be returned with states in
    the order ABCD.

    Parameters
    ----------
    state_time_courses: list of numpy.ndarray

    Returns
    -------
    matched_state_time_courses: list of numpy.ndarray
    """
    # If the state time courses have different length we only use the first n_samples
    n_samples = min([stc.shape[0] for stc in state_time_courses])

    # Match time courses based on correlation
    matched_state_time_courses = [state_time_courses[0][:n_samples]]
    orders = [np.arange(state_time_courses[0].shape[1])]
    for state_time_course in state_time_courses[1:]:
        correlation = correlate_states(
            state_time_courses[0][:n_samples], state_time_course[:n_samples]
        )
        correlation = np.nan_to_num(
            np.nan_to_num(correlation, nan=np.nanmin(correlation) - 1)
        )
        matches = linear_sum_assignment(-correlation)
        matched_state_time_courses.append(state_time_course[:n_samples, matches[1]])
        orders.append(matches[1])

    if return_order:
        return orders
    else:
        return matched_state_time_courses


@transpose(0, "state_time_course")
def state_activation(state_time_course: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate state activations for a state time course.

    Given a state time course (strictly binary), calculate the beginning and end of each
    activation of each state.

    Parameters
    ----------
    state_time_course : numpy.ndarray
        State time course (strictly binary).

    Returns
    -------
    ons : list of numpy.ndarray
        List containing state beginnings in the order they occur for each channel.
        This cannot necessarily be converted into an array as an equal number of
        elements in each array is not guaranteed.
    offs : list of numpy.ndarray
        List containing state ends in the order they occur for each channel.
        This cannot necessarily be converted into an array as an equal number of
        elements in each array is not guaranteed.
    """
    channel_on = []
    channel_off = []

    diffs = np.diff(state_time_course, axis=0)
    for i, diff in enumerate(diffs.T):
        on = (diff == 1).nonzero()[0]
        off = (diff == -1).nonzero()[0]
        try:
            if on[-1] > off[-1]:
                off = np.append(off, len(diff))

            if off[0] < on[0]:
                on = np.insert(on, 0, -1)

            channel_on.append(on)
            channel_off.append(off)
        except IndexError:
            _logger.info(f"No activation in state {i}.")
            channel_on.append(np.array([]))
            channel_off.append(np.array([]))

    channel_on = np.array(channel_on, dtype=object)
    channel_off = np.array(channel_off, dtype=object)

    return channel_on, channel_off


@transpose(0, "state_time_course")
def reduce_state_time_course(state_time_course: np.ndarray) -> np.ndarray:
    """Remove empty states from a state time course.

    If a state has no activation in the state time course, remove the column
    corresponding to that state.

    Parameters
    ----------
    state_time_course: numpy.ndarray

    Returns
    -------
    reduced_state_time_course: numpy.ndarray
        A state time course with no states with no activation.
    """
    return state_time_course[:, ~np.all(state_time_course == 0, axis=0)]


@transpose(0, "state_time_course")
def state_lifetimes(state_time_course: np.ndarray) -> List[np.ndarray]:
    """Calculate state lifetimes for a state time course.

    Given a state time course (one-hot encoded), calculate the lifetime of each
    activation of each state.

    Parameters
    ----------
    state_time_course : numpy.ndarray
        State time course (strictly binary).

    Returns
    -------
    channel_lifetimes : list of numpy.ndarray
        List containing an array of lifetimes in the order they occur for each channel.
        This cannot necessarily be converted into an array as an equal number of
        elements in each array is not guaranteed.
    """
    ons, offs = state_activation(state_time_course)
    channel_lifetimes = offs - ons
    return channel_lifetimes
