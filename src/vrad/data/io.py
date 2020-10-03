import logging
from typing import Tuple, Union

import mat73
import numpy as np
import scipy.io
from vrad.utils.misc import check_iterable_type, listify, time_axis_first

_logger = logging.getLogger("VRAD")


def load_spm(file_name: str) -> Tuple[np.ndarray, float]:
    """Load an SPM MEEG object.

    Highly untested function for reading SPM MEEG objects from MATLAB.

    Parameters
    ----------
    file_name: str
        Filename of an SPM MEEG object.

    Returns
    -------
    data: numpy.ndarray
        The time series referenced in the SPM MEEG object.
    sampling_frequency: float
        The sampling frequency listed in the SPM MEEG object.

    """
    spm = scipy.io.loadmat(file_name)
    data_file = spm["D"][0][0][6][0][0][0][0]
    n_channels = spm["D"][0][0][6][0][0][1][0][0]
    n_time_points = spm["D"][0][0][6][0][0][1][0][1]
    sampling_frequency = spm["D"][0][0][2][0][0]
    try:
        data = np.fromfile(data_file, dtype=np.float64).reshape(
            n_time_points, n_channels
        )
    except ValueError:
        data = np.fromfile(data_file, dtype=np.float32).reshape(
            n_time_points, n_channels
        )
    return data, sampling_frequency


def load_matlab(
    file_name: str, sampling_frequency: float = 1, ignored_keys=None
) -> Tuple[np.ndarray, float]:
    try:
        mat = scipy.io.loadmat(file_name)
    except NotImplementedError:
        mat = mat73.loadmat(file_name)

    discontinuity_indices = None

    if "D" in mat:
        _logger.info("Assuming that key 'D' corresponds to an SPM MEEG object.")
        time_series, sampling_frequency = load_spm(file_name=file_name)
    else:
        try:
            time_series = mat["X"]
            if "T" in mat:
                discontinuity_indices = mat["T"][0].astype(int)
        except:
            raise KeyError("data in MATLAB file must be contained in a field called X.")

    return time_series, discontinuity_indices, sampling_frequency


def load_data(
    time_series: Union[str, np.ndarray],
    sampling_frequency: float = 1,
    mmap_location=None,
) -> Tuple[np.ndarray, float]:

    # Read time series from a file
    if isinstance(time_series, str):
        if time_series[-4:] == ".npy":
            time_series = np.load(time_series)
        elif time_series[-4:] == ".mat":
            time_series, discontinuity_indices, sampling_frequency = load_matlab(
                file_name=time_series, sampling_frequency=sampling_frequency,
            )

    # If a python list has been passed, convert to a numpy array
    if isinstance(time_series, list):
        time_series = np.array(time_series)

    # Check the time series has the appropriate shape
    if time_series.ndim != 2:
        raise ValueError(
            f"{time_series.shape} detected. Time series must be a 2D array."
        )

    # Check time is the first axis, channels are the second axis
    time_series = time_axis_first(time_series)

    # Indicate the entire time series is continuous if discontinuities have not
    # been passed
    if discontinuity_indices is None:
        discontinuity_indices = [time_series.shape[0]]

    if mmap_location is not None:
        np.save(mmap_location, time_series)
        time_series = np.load(mmap_location, mmap_mode="r+")

    return time_series, discontinuity_indices, sampling_frequency
