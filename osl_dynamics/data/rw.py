"""Functions for reading and writing data.

"""

import warnings
from os import listdir, path

import mat73
import numpy as np
import scipy.io

from osl_dynamics.data import spm


def validate_inputs(inputs):
    """Validates inputs.

    Parameters
    ----------
    inputs : list of str or str or np.ndarray
        Inputs files or data.

    Returns
    -------
    validated_inputs : list of str or str
        Validated inputs.
    """
    if isinstance(inputs, str):
        if path.isdir(inputs):
            validated_inputs = list_dir(inputs, keep_ext=[".npy", ".mat", ".txt"])
        else:
            validated_inputs = [inputs]

    elif isinstance(inputs, np.ndarray):
        if inputs.ndim == 1:
            validated_inputs = [inputs[:, np.newaxis]]
        elif inputs.ndim == 2:
            validated_inputs = [inputs]
        else:
            validated_inputs = inputs

    elif isinstance(inputs, list):
        if len(inputs) == 0:
            raise ValueError("Empty list passed.")
        elif isinstance(inputs[0], str):
            validated_inputs = []
            for inp in inputs:
                if path.isdir(inp):
                    validated_inputs += list_dir(inp, keep_ext=[".npy", ".mat", ".txt"])
                else:
                    validated_inputs.append(inp)
        else:
            validated_inputs = inputs

    else:
        raise ValueError("inputs must be str, np.ndarray or list.")

    return validated_inputs


def file_ext(filename):
    """Returns the extension of a file.

    Parameters
    ----------
    filename : str
        Path to file.
    """
    if not isinstance(filename, str):
        return None
    _, ext = path.splitext(filename)
    return ext


def list_dir(path, keep_ext=None):
    """Lists a directory.

    Parameters
    ----------
    path : str
        Directory to list.
    keep_ext : str or list
        Extensions of files to include in the returned list. Default
        is to include add files.

    Returns
    -------
    files : list
        Full path to files with the correct extension.
    """
    files = []
    if keep_ext is None:
        for file in sorted(listdir(path)):
            files.append(path + "/" + file)
    else:
        if isinstance(keep_ext, str):
            keep_ext = [keep_ext]
        for file in sorted(listdir(path)):
            if file_ext(file) in keep_ext:
                files.append(path + "/" + file)
    return files


def load_data(
    data,
    data_field="X",
    mmap_location=None,
    mmap_mode="r+",
):
    """Loads time series data.

    Checks the data shape is time by channel and that the data is float32.

    Parameters
    ----------
    data : numpy.ndarray or str or list
        An array or filename of a .npy, .txt, or .mat file containing the data.
    data_field : str
        If a MATLAB filename is passed, this is the field that corresponds to
        the data.
    mmap_location : str
        Filename to save the data as a numpy memory map.
    mmap_mode : str
        Mode to load memory maps in. Default is 'r+'.

    Returns
    -------
    data : np.memmap or np.ndarray
        Data.
    """
    if isinstance(data, np.ndarray):
        data = data.astype(np.float32)
        if mmap_location is None:
            return data
        else:
            # Save to a file so we can load data as a memory map
            np.save(mmap_location, data)
            data = mmap_location

    if isinstance(data, str):
        # Check if file/folder exists
        if not path.exists(data):
            raise FileNotFoundError(data)

        # Check extension
        ext = file_ext(data)
        if ext not in [".npy", ".mat", ".txt"]:
            raise ValueError("Data file must be .npy, .txt or .mat.")

        # Load a MATLAB file
        if ext == ".mat":
            data = load_matlab(data, data_field)
            data = data.astype(np.float32)
            if mmap_location is None:
                return data
            else:
                # Save to a file so we can load data as a memory map
                np.save(mmap_location, data)
                data = mmap_location

        # Load a numpy file
        elif ext == ".npy":
            if mmap_location is None:
                data = np.load(data)
                data = data.astype(np.float32)
                return data
            else:
                mmap_location = data

        # Load a text file
        elif ext == ".txt":
            data = np.loadtxt(data)
            data = data.astype(np.float32)
            if mmap_location is None:
                return data
            else:
                np.save(mmap_location, data)
                data = mmap_location

    # Load data as memmap
    data = np.load(mmap_location, mmap_mode=mmap_mode)
    data = data.astype(np.float32)

    return data


def load_matlab(filename, field, ignored_keys=None):
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
    data : np.ndarray
        Data in the MATLAB/SPM file.
    """
    # Load file
    mat = loadmat(filename, return_dict=True)

    # Get data
    if "D" in mat:
        warnings.warn(
            "Assuming that key 'D' corresponds to an SPM MEEG object.", RuntimeWarning
        )
        D = spm.SPM(filename)
        data = D.data
    else:
        try:
            data = mat[field]
        except KeyError:
            raise KeyError(f"field '{field}' missing from MATLAB file.")

    return data


def loadmat(filename, return_dict=False):
    """Wrapper for scipy.io.loadmat or mat73.loadmat.

    Parameters
    ----------
    filename : str
        Filename of MATLAB file to read.
    return_dict : bool
        If there's only one field should we return a dictionary.
        Default is to return a numpy array if there is only one field.
        If there are multiple fields, a dictionary is always returned.

    Returns
    -------
    mat : dict or np.ndarray
        Data in the MATLAB file.
    """
    try:
        mat = scipy.io.loadmat(filename, simplify_cells=True)
    except NotImplementedError:
        mat = mat73.loadmat(filename)

    if not return_dict:
        # Check if there's only one key in the MATLAB file
        fields = [field for field in mat if "__" not in field]
        if len(fields) == 1:
            mat = mat[fields[0]]

    return mat
