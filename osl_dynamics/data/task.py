"""Functions to handle task data.

"""

import numpy as np


def epoch(data, time_index, pre, post, pad=False):
    """Transform (time, channel) data to (time, channel, epoch) data.

    Given a series of triggers given by :code:`time_index`, spit a continuous
    dataset into epochs. :code:`time_index` should be a sequence of integers
    representing the triggers of the epochs. :code:`pre` and :code:`post`
    specify the window around each trigger event.

    Parameters
    ----------
    data : np.ndarray
        A (time, channels) dataset to be epoched.
    time_index : np.ndarray
        The integer indices of the start of each epoch.
    pre : int
        The integer number of samples to include before the trigger.
    post : int
        The integer number of samples to include after the trigger.
    pad : bool, optional
        Pad with NaNs so that initial epochs will always been included.

    Returns
    -------
    epoched : np.ndarray
        A (time, channels, epochs) dataset.
    """
    if pad:
        # Pad before and after the data with zeros
        data = np.pad(data, ((pre, post), (0, 0)), constant_values=np.nan)
        time_index = time_index + pre
    else:
        # Only keep epochs we have all time points for
        keep = np.logical_and(
            time_index - pre > 0,
            time_index + post < data.shape[0],
        )
        time_index = time_index[keep]

    starts, stops = time_index - pre, time_index + post
    epoched = np.array([data[start:stop] for start, stop in zip(starts, stops)])
    return epoched


def epoch_mean(data, time_index, pre, post, pad=False):
    """Get the mean over epochs of a (time, channels) dataset.

    Calls :code:`epoch_mean`, and takes a mean over epochs, returning data
    with dimensions (time, channels) in which the time is the length of the
    epoch window.

    Parameters
    ----------
    data : np.ndarray
        A (time, channels) dataset to be epoched.
    time_index : np.ndarray
        The integer indices of the start of each epoch.
    pre : int
        The integer number of samples to include before the trigger.
    post : int
        The integer number of samples to include after the trigger.
    pad : bool, optional
        Pad with NaNs so that initial epochs will always been included.

    Returns
    -------
    epoch_mean : np.ndarray
        (time, channels) data meaned over epochs.
    """
    return np.nanmean(epoch(data, time_index, pre, post, pad=pad), axis=0)
