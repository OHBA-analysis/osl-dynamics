import logging
import pathlib
from os import path
from shutil import rmtree
from typing import Tuple, Union

import mat73
import numpy as np
import scipy.io
from tqdm import tqdm
from vrad.utils.misc import time_axis_first

_logger = logging.getLogger("VRAD")


class IO:
    """Class for reading/writing data.

    Parameters
    ----------
    inputs : list of str or str
        Filenames to be read.
    matlab_field : str
        If a MATLAB filename is passed, this is the field that corresponds to the data.
        Optional. By default we read the field 'X'.
    sampling_frequency : float
        Sampling frequency of the data in Hz. Optional.
    store_dir : str
        Directory to save results and intermediate steps to. Optional, default is /tmp.
    epoched : bool
        Flag indicating if the data has been epoched. Optional, default is False.
        If True, inputs must be a list of lists.
    """

    def __init__(
        self,
        inputs: Union[list, str, np.ndarray],
        matlab_field: str,
        sampling_frequency: float,
        store_dir: str,
        epoched: bool,
    ):
        # Validate inputs
        if isinstance(inputs, str):
            self.inputs = [inputs]

        elif isinstance(inputs, np.ndarray):
            if (inputs.ndim == 2 and not epoched) or (inputs.ndim == 3 and epoched):
                self.inputs = [inputs]
            else:
                self.inputs = inputs

        elif isinstance(inputs, list):
            if epoched and not isinstance(inputs[0], list):
                raise ValueError("If data is epoched, inputs must be a list of lists.")
            else:
                self.inputs = inputs

        else:
            raise ValueError("inputs must be str, np.ndarray or list.")

        self.store_dir = pathlib.Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Raw data memory maps
        raw_data_pattern = "raw_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(len(inputs))), identifier=self._identifier
        )
        self.raw_data_filenames = [
            str(self.store_dir / raw_data_pattern.format(i=i))
            for i in range(len(inputs))
        ]

        # Load the data
        self.epoched = epoched
        self.raw_data_memmaps = self.load_data(matlab_field)

        # Validate the data
        self.validate_data()

        # Attributes describing the raw data
        self.raw_data_mean = [
            np.mean(raw_data, axis=0) for raw_data in self.raw_data_memmaps
        ]
        self.raw_data_std = [
            np.std(raw_data, axis=0) for raw_data in self.raw_data_memmaps
        ]
        self.n_raw_data_channels = self.raw_data_memmaps[0].shape[-1]
        self.sampling_frequency = sampling_frequency

        # Use raw data for the subject data
        self.subjects = self.raw_data_memmaps

    def delete_dir(self):
        """Deletes the directory that stores the memory maps."""
        rmtree(self.store_dir, ignore_errors=True)

    def load_data(self, matlab_field: str) -> list:
        """Import data into a list of memory maps.

        Parameters
        ----------
        matlab_field : str
            If a MATLAB filename is passed, this is the field that corresponds
            to the data. By default we read the field 'X'.

        Returns
        -------
        list
            list of numpy memmaps.
        """
        memmaps = []
        for in_file, out_file in zip(
            tqdm(self.inputs, desc="Loading files", ncols=98), self.raw_data_filenames
        ):
            data = load_time_series(
                in_file, matlab_field, self.epoched, mmap_location=out_file
            )
            memmaps.append(data)
        return memmaps

    def validate_data(self):
        """Validate data files."""
        n_channels = [memmap.shape[-1] for memmap in self.raw_data_memmaps]
        if not np.equal(n_channels, n_channels[0]).all():
            raise ValueError("All inputs should have the same number of channels.")


def load_time_series(
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
        If a MATLAB filename is passed, this is the field that corresponds to
        the data.

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
