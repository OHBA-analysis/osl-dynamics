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


def load_matlab(filename: str, field: str, ignored_keys=None) -> np.ndarray:
    """Loads a MATLAB or SPM file.

    Parameters
    ----------
    filename : str
        Filename of MATLAB file to read.
    field : str
        Field that corresponds to the data.
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
        _logger.warning("Assuming that key 'D' corresponds to an SPM MEEG object.")
        time_series, sampling_frequency = load_spm(filename=filename)
        print(f"Sampling frequency of the data is {sampling_frequency} Hz.")
    else:
        try:
            time_series = mat[field]
        except KeyError:
            raise KeyError(f"field '{field}' missing from MATLAB file.")

    return time_series


def load_data(
    time_series: Union[str, list, np.ndarray],
    matlab_field: str = "X",
    epoched: bool = False,
    mmap_location: str = None,
) -> np.ndarray:
    """Loads time series data.

    Parameters
    ----------
    time_series : numpy.ndarray or str or list
        An array or filename of a .npy or .mat file containing timeseries data.
    matlab_field : str
        If a MATLAB filename is passed, this is the field that corresponds to the data.
    epoched : bool
        Is the data epoched? Optional, default is False.
    mmap_location : str
        Filename to save the data as a numpy memory map.

    Returns
    -------
    np.ndarray
        Time series data.
    """

    # Read time series data from a file
    if isinstance(time_series, str):
        time_series = read_from_file(time_series, matlab_field)

    if isinstance(time_series, list):

        # If list of filenames, read data from file
        if isinstance(time_series[0], str):
            time_series = np.array(
                [read_from_file(fname, matlab_field) for fname in time_series]
            )

        # Otherwise we assume it's a list of numpy arrays
        else:
            time_series = np.array(time_series)

    # Check the time series has the appropriate dimensionality
    if epoched and time_series.ndim != 3:
        raise ValueError(
            f"Shape {time_series.shape} detected for time series. "
            + "epoched data must be 3D."
        )

    if (not epoched) and time_series.ndim != 2:
        if time_series.ndim == 3:
            raise ValueError(
                f"Shape {time_series.shape} detected for time series. "
                + "Try epoched=True."
            )
        else:
            raise ValueError(
                f"Shape {time_series.shape} detected. Time series must be 2D."
            )

    # Check time is the first axis and channels are the second axis
    if epoched:
        time_series = np.array([time_axis_first(ts) for ts in time_series])
    else:
        time_series = time_axis_first(time_series)

    # Make sure the time series is type float32
    time_series = time_series.astype(np.float32)

    # Load from memmap
    if mmap_location is not None:
        np.save(mmap_location, time_series)
        time_series = np.load(mmap_location, mmap_mode="r+")

    return time_series


def read_from_file(filename: str, matlab_field: str = None) -> np.ndarray:
    """Loads time series data.

    Parameters
    ----------
    filename : str
        Filename of a .npy or .mat file containing time series data.
    matlab_field : str
        If a MATLAB filename is passed, this is the field that corresponds to the data.

    Returns
    -------
    np.ndarray
        Time series data.
    """

    # Check if file exists
    if not path.exists(filename):
        raise FileNotFoundError(filename)

    # Read data from the file
    if filename[-4:] == ".npy":
        time_series = np.load(filename)

    elif filename[-4:] == ".mat":
        time_series = load_matlab(filename=filename, field=matlab_field)

    return time_series
