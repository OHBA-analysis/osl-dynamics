import inspect
import logging
from copy import copy
from typing import Any

import numpy as np

_logger = logging.getLogger("VRAD")


def override_dict_defaults(default_dict: dict, override_dict: dict = None) -> dict:
    """Helper function to update default dictionary values with user values.

    Parameters
    ----------
    default_dict : dict
        Dictionary of default values.
    override_dict : dict
        Dictionary of user values.

    Returns
    -------
    new_dict : dict
        default_dict with values replaced by user values.

    """
    if override_dict is None:
        override_dict = {}

    return {**default_dict, **override_dict}


def listify(obj: object):
    """Create a list from any input.

    If None is passed, return an empty list.
    If a list is passed, return the list.
    If a tuple is passed, return it as a list.
    If any other object is passed, return it as a single item list.

    Parameters
    ----------
    obj : Any
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


def replace_argument(
    func: callable, name: str, item: Any, args: list, kwargs: dict, append: bool = False
):
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
    append : bool
        Whether the value should be appended or replace the existing argument.

    Returns
    -------
    args : list
    kwargs : dict

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
    _______
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


def check_iterable_type(iterable, object_type: type):
    """Check iterable is non-empty and contains only objects of specific type.

    Parameters
    ----------
    iterable : iterable
        Iterable to check the type of.
    object_type : type
        Type to check for.

    Returns
    _______
    type_correct : bool
        Whether the iterable only contains the specified type.
    """
    if not hasattr(iterable, "__iter__") and not isinstance(iterable, str):
        return False
    if isinstance(iterable, np.ndarray):
        return iterable.dtype == object_type
    return bool(iterable) and all(isinstance(elem, object_type) for elem in iterable)


def time_axis_first(input_array: np.ndarray) -> np.ndarray:
    """Make arrays have their longest dimension first.

    Parameters
    ----------
    input_array : numpy.ndarray
        The array to be transposed or returned.

    Returns
    -------
    transposed_array : numpy.ndarray

    """
    if input_array.ndim != 2:
        _logger.info("Non-2D array not transposed.")
        return input_array
    if input_array.shape[1] > input_array.shape[0]:
        input_array = np.transpose(input_array)
        _logger.warning(
            "More channels than time points detected. Time series has been transposed."
        )
    return input_array


class LoggingContext:
    """

    Parameters
    ----------
    logger
    suppress_level
    handler
    close
    """

    def __init__(self, logger, suppress_level="warning", handler=None, close=True):

        self.logger = logging.getLogger(logger)
        if isinstance(suppress_level, str):
            suppress_level = logging.getLevelName(suppress_level.upper())
        self.level = suppress_level + 10
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()


def array_to_memmap(filename, array):
    np.save(filename, array)
    return np.load(filename, mmap_mode="r+")


class MockFlags:
    def __init__(self, shape, c_contiguous=True):
        self.c_contiguous = c_contiguous
        self.f_contiguous = (not c_contiguous) or (c_contiguous and len(shape) == 1)


class MockArray:
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
            buffer_size = max(16 * 1024 ** 2 // self.dtype.itemsize, 1)

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
