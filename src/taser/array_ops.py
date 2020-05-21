"""Helper functions using NumPy

"""
import logging
from itertools import permutations
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from taser.helpers.decorators import transpose


@transpose
def match_states(
    state_time_course_1: np.ndarray, state_time_course_2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    correlation = np.zeros((state_time_course_1.shape[1], state_time_course_2.shape[1]))
    for i, state1 in enumerate(state_time_course_1.T):
        for j, state2 in enumerate(state_time_course_2.T):
            correlation[i, j] = np.corrcoef(state1, state2)[0, 1]
    matches = linear_sum_assignment(-correlation)
    return state_time_course_1, state_time_course_2[:, matches[1]]


def get_one_hot(values: np.ndarray, n_states: int = None):
    """Expand a categorical variable to a series of boolean columns (one-hot encoding).

    +----------------------+
    | Categorical Variable |
    +======================+
    |           A          |
    +----------------------+
    |           C          |
    +----------------------+
    |           D          |
    +----------------------+
    |           B          |
    +----------------------+

    becomes

    +---+---+---+---+
    | A | B | C | D |
    +===+===+===+===+
    | 1 | 0 | 0 | 0 |
    +---+---+---+---+
    | 0 | 0 | 1 | 0 |
    +---+---+---+---+
    | 0 | 0 | 0 | 1 |
    +---+---+---+---+
    | 0 | 1 | 0 | 0 |
    +---+---+---+---+

    Parameters
    ----------
    values : numpy.ndarray
        Categorical variable in a 1D array. Values should be integers (i.e. state 0, 1,
        2, 3, ... , `n_states`).
    n_states : int
        Total number of states in `values`. Must be at least the number of states
        present in `values`. Default is the number of unique values in `values`.

    Returns
    -------
    one_hot : numpy.ndarray
        A 2D array containing the one-hot encoded form of the input data.

    """
    if values.ndim == 2:
        logging.warning("argmax being taken on shorter axis.")
        values = values.argmax(axis=min(values.shape))
    if n_states is None:
        n_states = values.max() + 1
    res = np.eye(n_states)[np.array(values).reshape(-1)]
    return res.reshape(list(values.shape) + [n_states])


def dice_coefficient_1d(sequence_1: np.ndarray, sequence_2: np.ndarray) -> float:
    """Calculate the Dice coefficient of a discrete array

    Given two sequences containing a number of discrete elements (i.e. a
    categorical variable), calculate the Dice coefficient of those sequences.

    The Dice coefficient is 2 times the number of equal elements (equivalent to
    true-positives) divided by the sum of the total number of elements.

    Parameters
    ----------
    sequence_1 : numpy.ndarray
        A sequence containing discrete elements.
    sequence_2 : numpy.ndarray
        A sequence containing discrete elements.

    Returns
    -------
    dice_coefficient : float
        The Dice coefficient of the two sequences.

    Raises
    ------
    ValueError
        If either sequence is not one dimensional.

    """
    if len(sequence_1.shape) != 1:
        raise ValueError(
            f"sequence_1 must be a 1D array. Dimensions are {sequence_1.shape}."
        )
    if len(sequence_2.shape) != 1:
        raise ValueError(
            f"sequence_2 must be a 1D array. Dimensions are {sequence_2.shape}."
        )
    if sequence_1.dtype != np.int:
        raise TypeError("sequence_1 must by an array of integers.")
    if sequence_1.dtype != np.int:
        raise TypeError("sequence_2 must by an array of integers.")

    return 2 * ((sequence_1 == sequence_2).sum()) / (len(sequence_1) + len(sequence_2))


@transpose(0, 1, "sequence_1", "sequence_2")
def dice_coefficient(sequence_1: np.ndarray, sequence_2: np.ndarray) -> float:
    """Wrapper method for `dice_coefficient`.

    If passed a one-dimensional array, it will be sent straight to `dice_coefficient`.
    Given a two-dimensional array, it will perform an argmax calculation on each sample.
    The default axis for this is zero, i.e. each row represents a sample.

    Parameters
    ----------
    sequence_1 : numpy.ndarray
        A sequence containing either 1D discrete or 2D continuous data.
    sequence_2 : numpy.ndarray
        A sequence containing either 1D discrete or 2D continuous data.
    axis_1 : int
        For a 2D sequence_1, the axis on which to perform argmax. Default is 0.
    axis_2 : int
        For a 2D sequence_2, the axis on which to perform argmax. Default is 0.

    Returns
    -------
    dice_coefficient : float
        The Dice coefficient of the two sequences.

    See Also
    --------
    dice_coefficient_1D : Dice coefficient of 1D categorical sequences.

    """
    if (len(sequence_1.shape) not in [1, 2]) or (len(sequence_2.shape) not in [1, 2]):
        raise ValueError("Both sequences must be either 1D or 2D")
    if (len(sequence_1.shape) == 1) and (len(sequence_2.shape) == 1):
        return dice_coefficient_1d(sequence_1, sequence_2)
    if len(sequence_1.shape) == 2:
        sequence_1 = sequence_1.argmax(axis=1)
    if len(sequence_2.shape) == 2:
        sequence_2 = sequence_2.argmax(axis=1)
    return dice_coefficient_1d(sequence_1, sequence_2)


@transpose(0, "sequence_1", 1, "sequence_2")
def align_arrays(
    sequence_1: np.ndarray, sequence_2: np.ndarray, alignment: str = "left"
) -> Tuple[np.ndarray, np.ndarray]:
    min_length = min(len(sequence_1), len(sequence_2))

    if alignment == "left":
        sequence_1 = sequence_1[:min_length]
        sequence_2 = sequence_2[:min_length]

    elif alignment == "right":
        sequence_1 = sequence_1[-min_length:]
        sequence_2 = sequence_2[-min_length:]

    elif alignment == "center":
        half_length = int(min_length / 2)
        mid_1 = int(len(sequence_1) / 2)
        mid_2 = int(len(sequence_2) / 2)

        sequence_1 = sequence_1[mid_1 - half_length : mid_1 + half_length]
        sequence_2 = sequence_2[mid_2 - half_length : mid_2 + half_length]

    else:
        raise ValueError(f"Alignment must be left, right or center.")

    return sequence_1, sequence_2


@transpose(0, "sequence_1", 1, "sequence_2")
def dice_from_permutations(
    sequence_1: np.ndarray, sequence_2: np.ndarray, alignment: str = "all"
) -> List[List[Tuple[float, List[int], str]]]:
    if alignment == "all":
        dice_list = []
        for a in ["left", "right", "center"]:
            aligned_1, aligned_2 = align_arrays(sequence_1, sequence_2, alignment=a)
            for order in permutations(range(aligned_1.shape[1])):
                dice_list.append(
                    [dice_coefficient(aligned_1[:, order], aligned_2), order, a]
                )

    else:
        aligned_1, aligned_2 = align_arrays(sequence_1, sequence_2, alignment=alignment)

        dice_list = []
        perms = permutations(range(aligned_1.shape[1]))
        for order in perms:
            dice_list.append(
                [dice_coefficient(aligned_1[:, order], aligned_2), order, alignment]
            )

    dice_list = sorted(dice_list, reverse=True)

    return dice_list


@transpose(0, "sequence_1", 1, "sequence_2")
def get_maximum_dice_arrays(
    sequence_1: np.ndarray, sequence_2: np.ndarray, alignment: str = "all"
):
    ordered_dice = dice_from_permutations(sequence_1, sequence_2, alignment=alignment)
    dice, order, alignment = ordered_dice[0]
    print(
        f"maximum dice coefficent is {dice} from order {order} "
        f"with alignment {alignment}"
    )
    return sequence_1[:, order], sequence_2


@transpose(0, "state_time_course")
def state_activation(state_time_course: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate state activations for a state time course.

    Given a state time course (strictly binary), calculate the beginning and end of each
    activation of each state.

    Parameters
    ----------
    state_time_course : numpy.ndarray
        State time course (strictly binary).
    time_axis : int
        Specify the axis which denotes time. If 0, `state_time_course` should have
        dimensions [time points x channels].

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
            logging.info(f"No activation in state {i}.")

    channel_on = np.array(channel_on)
    channel_off = np.array(channel_off)

    return channel_on, channel_off


@transpose(0, "state_time_course")
def reduce_state_time_course(state_time_course: np.ndarray) -> np.ndarray:
    return state_time_course[:, ~np.all(state_time_course == 0, axis=0)]


@transpose(0, "state_time_course")
def state_lifetimes(state_time_course: np.ndarray) -> List[np.ndarray]:
    """Calculate state lifetimes for a state time course.

    Given a state time course (strictly binary), calculate the lifetime of each
    activation of each state.

    Parameters
    ----------
    state_time_course : numpy.ndarray
        State time course (strictly binary).
    time_axis : int
        Specify the axis which denotes time. If 0, `state_time_course` should have
        dimensions [time points x channels].

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


def from_cholesky(cholesky_matrix: np.ndarray):
    return cholesky_matrix @ cholesky_matrix.transpose((0, 2, 1))


@transpose(0, "state_time_course")
def calculate_trans_prob_matrix(
    state_time_course: np.ndarray,
    zero_diagonal: bool = False,
    n_states: int = None,
    normalize: bool = True,
) -> np.ndarray:
    if state_time_course.ndim == 2:
        state_time_course = state_time_course.argmax(axis=1)
    if state_time_course.ndim != 1:
        raise ValueError("state_time_course should either be 1D or 2D.")

    vals, counts = np.unique(
        state_time_course[
            np.arange(2)[None, :] + np.arange(len(state_time_course) - 1)[:, None]
        ],
        axis=0,
        return_counts=True,
    )

    if n_states is None:
        n_states = state_time_course.max() + 1

    trans_prob = np.zeros((n_states, n_states))
    trans_prob[vals[:, 0], vals[:, 1]] = counts

    if normalize:
        trans_prob = trans_prob / trans_prob.sum(axis=1)

    if zero_diagonal:
        np.fill_diagonal(trans_prob, 0)
    return trans_prob


def trace_normalize(matrix: np.ndarray):
    matrix = np.array(matrix)
    if matrix.ndim == 2:
        matrix = matrix[None, ...]
    if matrix.ndim != 3:
        raise ValueError("Matrix should be 2D or 3D.")

    return matrix / matrix.trace(axis1=1, axis2=2)[:, None, None]
