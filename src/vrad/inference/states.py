"""Functions to manipulate and calculate statistics for inferred state time courses.

"""

import logging
from typing import List, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from vrad import array_ops
from vrad.utils.decorators import transpose


_logger = logging.getLogger("VRAD")


def time_courses(alpha: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
    """Calculates state time courses.

    Hard classifies the state probabilities (alpha).
    """
    if isinstance(alpha, list):
        stcs = [a.argmax(axis=1) for a in alpha]
        stcs = [array_ops.get_one_hot(stc) for stc in stcs]
    else:
        stcs = alpha.argmax(axis=1)
        stcs = array_ops.get_one_hot(stcs)
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
        correlation = np.nan_to_num(correlation, nan=np.nanmin(correlation) - 1)
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

    channel_on = np.array(channel_on)
    channel_off = np.array(channel_off)

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
