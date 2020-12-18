import numpy as np


def epoch(
    data: np.ndarray, time_index: np.ndarray, pre: int = 125, post: int = 1000
) -> np.ndarray:
    """Transform [time x channel] data to [time x channel x epoch] data.

    Given a series of triggers given by `time_index`, spit a continuous dataset into
    epochs. `time_index` should be a sequence of integers representing the triggers
    of the epochs. `pre` and `post` specify the window around each trigger event.

    Parameters
    ----------
    data: numpy.ndarray
        A [time x channels] dataset to be epoched.
    time_index: numpy.ndarray
        The integer indices of the start of each epoch.
    pre: int
        The integer number of samples to include before the trigger.
    post: int
        The integer number of samples to include after the trigger.

    Returns
    -------
    epoched: numpy.ndarray
        A [time x channels x epochs] dataset.
    """
    # If there are not enough time points before the first trigger, discard it.
    # TODO: Use padding with NaNs instead of discarding.
    #       Check interaction with plots and means.
    if time_index[0] - pre < 0:
        time_index = time_index[1:]

    starts, stops = time_index - pre, time_index + post
    epoched = np.array([data[start:stop] for start, stop in zip(starts, stops)])
    return epoched


def epoch_mean(
    data: np.ndarray, time_index: np.ndarray, pre: int = 125, post: int = 1000
) -> np.ndarray:
    """Get the mean over epochs of a [time x channels] dataset.

    Calls `epoch_mean`, and takes a mean over epochs, returning data with dimensions
    [time x channels] in which the time is the length of the epoch window.

    Parameters
    ----------
    data: numpy.ndarray
        A [time x channels] dataset to be epoched.
    time_index: numpy.ndarray
        The integer indices of the start of each epoch.
    pre: int
        The integer number of samples to include before the trigger.
    post: int
        The integer number of samples to include after the trigger.

    Returns
    -------
    epoch_mean: numpy.ndarray
        [time x channels] data meaned over epochs.
    """
    return epoch(data, time_index, pre, post).mean(axis=0)
