import logging
from typing import Tuple, Union
from os import path

import mat73
import numpy as np
import scipy.io
from vrad.utils.misc import time_axis_first

_logger = logging.getLogger("VRAD")


def load_spm(filename: str) -> Tuple[np.ndarray, float]:
    """Load an SPM MEEG object.

    Highly untested function for reading SPM MEEG objects from MATLAB.

    Parameters
    ----------
    filename: str
        Filename of an SPM MEEG object.

    Returns
    -------
    data: numpy.ndarray
        The time series referenced in the SPM MEEG object.
    sampling_frequency: float
        The sampling frequency listed in the SPM MEEG object.

    """
    spm = scipy.io.loadmat(filename)
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


def load_matlab(filename: str, ignored_keys=None) -> np.ndarray:
    """Loads a MATLAB or SPM file.

    Parameters
    ----------
    filename : str
        Filename of MATLAB file to read.
    ignored_keys :  list of str
        Keys in the MATLAB file to ignore.

    Returns
    -------
    time_series: np.ndarray
        Data in the MATLAB/SPM file.
    """
    try:
        mat = scipy.io.loadmat(filename)
    except NotImplementedError:
        mat = mat73.loadmat(filename)

    if "D" in mat:
        _logger.info("Assuming that key 'D' corresponds to an SPM MEEG object.")
        time_series, sampling_frequency = load_spm(filename=filename)
        print(f"Sampling frequency of the data is {sampling_frequency} Hz.")
    else:
        try:
            time_series = mat["X"]
        except KeyError:
            raise KeyError("data in MATLAB file must be contained in a field called X.")

    return time_series


def load_data(
    time_series: Union[str, list, np.ndarray], mmap_location: str = None
) -> np.ndarray:
    """Loads time series data.

    Parameters
    ----------
    time_series : numpy.ndarray or str
        An array or filename of a .npy or .mat file containing timeseries data.
    mmap_location : str
        Filename to save the data as a numpy memory map.

    Returns
    -------
    np.ndarray
        Time series data.
    """

    # Read time series from a file
    if isinstance(time_series, str):
        if not path.exists(time_series):
            raise FileNotFoundError(time_series)
        if time_series[-4:] == ".npy":
            time_series = np.load(time_series)
        elif time_series[-4:] == ".mat":
            time_series = load_matlab(filename=time_series)

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

    # Load from memmap
    if mmap_location is not None:
        np.save(mmap_location, time_series)
        time_series = np.load(mmap_location, mmap_mode="r+")

    # Make sure the time series is type float32
    time_series = time_series.astype(np.float32)

    return time_series
