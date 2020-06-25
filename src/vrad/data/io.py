import logging
from typing import Tuple, Union

import mat73
import numpy as np
import scipy.io
from vrad.utils.misc import listify

_logger = logging.getLogger("VRAD")


def get_ignored_keys(new_keys):
    new_keys = listify(new_keys)
    ignored_matlab_keys = [
        "__globals__",
        "__header__",
        "__version__",
        "save_time",
        "pca_applied",
        "T",
    ] + new_keys
    return ignored_matlab_keys


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
    ignored_keys = get_ignored_keys(ignored_keys)
    try:
        mat = scipy.io.loadmat(file_name)
    except NotImplementedError:
        mat = mat73.loadmat(file_name)

    if "D" in mat:
        _logger.info("Assuming that key 'D' corresponds to an SPM MEEG object.")
        time_series, sampling_frequency = load_spm(file_name=file_name)
    else:
        for key in mat:
            if key not in ignored_keys:
                time_series = mat[key]
                break
        else:
            raise KeyError("No keys found which aren't excluded.")

    return time_series, sampling_frequency


def load_data(
    time_series: Union[str, np.ndarray],
    multi_sequence="all",
    sampling_frequency: float = 1,
    ignored_keys=None,
) -> Tuple[np.ndarray, float]:
    if isinstance(time_series, str):
        if time_series[-4:] == ".npy":
            time_series = np.load(time_series)
        elif time_series[-4:] == ".mat":
            time_series, sampling_frequency = load_matlab(
                file_name=time_series,
                sampling_frequency=sampling_frequency,
                ignored_keys=ignored_keys,
            )
    if isinstance(time_series, list):
        if multi_sequence == "all":
            _logger.warning(
                f"{len(time_series)} sequences detected. "
                f"Concatenating along first axis."
            )
            time_series = np.concatenate(time_series)
        if isinstance(multi_sequence, int):
            _logger.warning(
                f"{len(time_series)} sequences detected. "
                f"Using sequence with index {multi_sequence}."
            )
            time_series = time_series[multi_sequence]

    return time_series, sampling_frequency
