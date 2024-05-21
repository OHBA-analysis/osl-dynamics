"""Miscellaneous utility classes and functions.

"""

import inspect
import logging
import pickle
import random
import sys
from copy import copy
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import yaml
from yaml.constructor import ConstructorError

_logger = logging.getLogger("osl-dynamics")


def nextpow2(x):
    """Next power of 2.

    Parameters
    ----------
    x : int
        Any integer.

    Returns
    -------
    res : int
        The smallest power of two that is greater than or equal to the absolute
        value of x.
    """
    if x == 0:
        return 0
    res = np.ceil(np.log2(np.abs(x)))
    return res.astype("int")


def leading_zeros(number, largest_number):
    """Pad a number with leading zeros.

    This is useful for creating a consistent naming scheme for files.

    Parameters
    ----------
    number : int
        Number to be padded.
    largest_number : int
        Largest number in the set.

    Returns
    -------
    padded_number : str
        Number padded with leading zeros.
    """
    min_length = len(str(largest_number))
    padded_number = str(number).zfill(min_length)
    return padded_number


def override_dict_defaults(default_dict, override_dict=None):
    """Helper function to update default dictionary values with user values.

    Parameters
    ----------
    default_dict : dict
        Dictionary of default values.
    override_dict : dict, optional
        Dictionary of user values.

    Returns
    -------
    new_dict : dict
        default_dict with values replaced by user values.
    """
    if override_dict is None:
        override_dict = {}
    return {**default_dict, **override_dict}


def listify(obj):
    """Create a list from any input.

    If :code:`None` is passed, return an empty list.
    If a list is passed, return the list.
    If a tuple is passed, return it as a list.
    If any other object is passed, return it as a single item list.

    Parameters
    ----------
    obj : typing.Any
        Object to be transformed to a list.

    Returns
    -------
    Object as a list.
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    else:
        return [obj]


def replace_argument(func, name, item, args, kwargs, append=False):
    """Replace arguments in function calls.

    Parameters
    ----------
    func : callable
        The function being called.
    name : str
        Name of the variable to be modified.
    item
        The value to be added.
    args : dict
        Original arguments.
    kwargs : dict
        Original keyword arguments.
    append : bool, optional
        Whether the value should be appended or replace the existing argument.

    Returns
    -------
    args : list
        Arguments.
    kwargs : dict
        Keyword arguments.
    """
    args = copy(listify(args))
    kwargs = copy(kwargs)
    param_order = list(inspect.signature(func).parameters)
    param_position = param_order.index(name)
    if len(args) > param_position:
        if append:
            args[param_position] = listify(args[param_position]) + listify(item)
        else:
            args[param_position] = item
    elif name in kwargs:
        if append:
            kwargs[name] = listify(kwargs[name]) + listify(item)
        else:
            kwargs[name] = item
    else:
        kwargs[name] = item
    return args, kwargs


def get_argument(func, name, args, kwargs):
    """Get an argument passed to a function call whether it is a normal
    argument or keyword argument.

    Parameters
    ----------
    func : callable
        The function being called.
    name : str
        Name of the variable to be modified.
    args : dict
        Arguments.
    kwargs : dict
        Keyword arguments.

    Returns
    -------
    args : argument
        Argument.
    """
    args = copy(listify(args))
    kwargs = copy(kwargs)
    param_order = list(inspect.signature(func).parameters)
    param_position = param_order.index(name)
    if len(args) > param_position:
        arg = args[param_position]
    else:
        if name not in kwargs:
            return None
        arg = kwargs[name]
    return arg


def check_arguments(args, kwargs, index, name, value, comparison_op):
    """Checks the arguments passed to a function.

    Parameters
    ----------
    args : list
        Arguments.
    kwargs : dict
        Keyword arguments.
    index : int
        Index of argument.
    name : str
        Name of keyword argument.
    value
        Value to compare to given argument.
    comparison_op : func
        Comparison operation for checking the original.

    Returns
    -------
    valid : bool
        If the given value is valid as determined by the comparison operation.
    """

    # Check if the argument we want to check is a normal argument
    args = listify(args)
    if len(args) >= index:
        return comparison_op(args[index], value)

    # Check if it is a keyword argument
    elif name in kwargs:
        return comparison_op(kwargs[name], value)

    # Otherwise the argument we want to check isn't in args or kwargs
    else:
        return False


def array_to_memmap(filename, array):
    """Save an array and reopen it as a np.memmap.

    Parameters
    ----------
    filename : str
        The name of the file to save to.
    array : np.ndarray
        The array to save.

    Returns
    -------
    memmap : np.memmap
        Memory map.
    """
    path = Path(filename)
    if path.exists():
        # Delete npy file
        path.unlink()

    # Save array
    np.save(filename, array)

    # Load as a memmap
    return np.load(filename, mmap_mode="r+")


class MockFlags:
    """Flags for memmap header construction.

    Parameters
    ----------
    shape : list of int
        The shape of the array being mapped.
    c_contiguous : bool, optional
        Is the array C contiguous or F contiguous?
    """

    def __init__(self, shape, c_contiguous=True):
        self.c_contiguous = c_contiguous
        self.f_contiguous = (not c_contiguous) or (c_contiguous and len(shape) == 1)


class MockArray:
    """Create an empty array on disk without creating it in memory.

    Parameters
    ----------
    shape : list of int
        Dimensions or the array being created.
    dtype : type
        The data type of the array.
    c_contiguous : bool, optional
        Is the array C contiguous or F contiguous?
    """

    def __init__(self, shape, dtype=np.float64, c_contiguous=True):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.flags = MockFlags(shape, c_contiguous)

        self.filename = None

    def save(self, filename):
        if filename[-4:] != ".npy":
            filename = f"{filename}.npy"
        self.filename = filename
        if self.dtype.itemsize == 0:
            buffer_size = 0
        else:
            # Set buffer size to 16 MiB to hide the Python loop overhead.
            buffer_size = max(16 * 1024**2 // self.dtype.itemsize, 1)

        n_chunks, remainder = np.divmod(
            np.product(self.shape) * self.dtype.itemsize, buffer_size
        )

        with open(filename, "wb") as f:
            np.lib.format.write_array_header_2_0(
                f, np.lib.format.header_data_from_array_1_0(self)
            )

            for chunk in range(n_chunks):
                f.write(b"\x00" * buffer_size)
            f.write(b"\x00" * remainder)

    def memmap(self):
        if self.filename is None:
            raise ValueError("filename has not been provided.")
        return np.load(self.filename, mmap_mode="r+")

    @classmethod
    def to_disk(cls, filename, shape, dtype=np.float64, c_contiguous=True):
        mock_array = cls(shape, dtype, c_contiguous)
        mock_array.save(filename)

    @classmethod
    def get_memmap(cls, filename, shape, dtype=np.float64, c_contiguous=True):
        cls.to_disk(filename, shape, dtype, c_contiguous)
        return np.load(filename, mmap_mode="r+")


class NumpyLoader(yaml.UnsafeLoader):
    def find_python_name(self, name, mark, unsafe=False):
        if not name:
            raise ConstructorError(
                "while constructing a Python object",
                mark,
                "expected non-empty name appended to the tag",
                mark,
            )
        if "." in name:
            module_name, object_name = name.rsplit(".", 1)
        else:
            module_name = "builtins"
            object_name = name
        if "numpy" in module_name:
            try:
                __import__(module_name)
            except ImportError as exc:
                raise ConstructorError(
                    "while constructing a Python object",
                    mark,
                    "cannot find module %r (%s)" % (module_name, exc),
                    mark,
                )
        if module_name not in sys.modules:
            raise ConstructorError(
                "while constructing a Python object",
                mark,
                "module %r is not imported" % module_name,
                mark,
            )
        module = sys.modules[module_name]
        if not hasattr(module, object_name):
            raise ConstructorError(
                "while constructing a Python object",
                mark,
                "cannot find %r in the module %r" % (object_name, module.__name__),
                mark,
            )
        return getattr(module, object_name)


def save(filename, array):
    """Save a file.

    Parameters
    ----------
    filename : str
        Path to file to save to. Must be '.npy' or '.pkl'.
    array : np.ndarray or list
        Array to save.
    """
    # Validation
    ext = Path(filename).suffix
    if ext not in [".npy", ".pkl"]:
        raise ValueError("filename extension must be .npy or .pkl.")

    # Save
    _logger.info(f"Saving {filename}")
    if ext == ".pkl":
        pickle.dump(array, open(filename, "wb"))
    else:
        np.save(filename, array)


def load(filename, **kwargs):
    """Load a file.

    Parameters
    ----------
    filename : str
        Path to file to load. Must be '.npy' or '.pkl'.

    Returns
    -------
    array : np.ndarray or list
        Array loaded from the file.
    """
    # Validation
    ext = Path(filename).suffix
    if ext not in [".npy", ".pkl"]:
        raise ValueError("filename extension must be .npy or .pkl.")

    # Load
    _logger.info(f"Loading {filename}")
    if ext == ".pkl":
        array = pickle.load(open(filename, "rb"))
    else:
        array = np.load(filename, **kwargs)

    return array


def set_random_seed(seed):
    """Set all random seeds.

    This includes Python's random module, NumPy and TensorFlow.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    import tensorflow as tf  # avoids slow imports

    _logger.info(f"Setting random seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@contextmanager
def set_logging_level(logger, level):
    current_level = logger.getEffectiveLevel()
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(current_level)
