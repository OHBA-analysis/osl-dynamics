"""Miscellaneous utility classes and functions.

"""

import inspect
import sys
from copy import copy
from pathlib import Path

import numpy as np
import yaml
from yaml.constructor import ConstructorError


def override_dict_defaults(default_dict, override_dict=None):
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


def listify(obj):
    """Create a list from any input.

    If None is passed, return an empty list.
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


def check_iterable_type(iterable, object_type):
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


def time_axis_first(input_array):
    """Make arrays have their longest dimension first.

    Parameters
    ----------
    input_array : np.ndarray
        The array to be transposed or returned.

    Returns
    -------
    transposed_array : np.ndarray
    """
    if input_array.ndim != 2:
        return input_array
    if input_array.shape[1] > input_array.shape[0]:
        input_array = np.transpose(input_array)
    return input_array


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
    """
    np.save(filename, array)
    return np.load(filename, mmap_mode="r+")


class MockFlags:
    """Flags for memmap header construction.

    Parameters
    ----------
    shape : list of int
        The shape of the array being mapped.
    c_contiguous : bool
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
    dtype
        The data type of the array.
    c_contiguous : bool
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


def _gen_dict_extract(key, dictionary, current_key="root"):
    """Search for a key in a nested dict and get the value and full path of a key.

    Parameters
    ----------
    key: str
        The key to search for.
    dictionary: dict
        The nested dictionary to search.
    current_key: str
        The current path (nesting level) in the dictionary.
    """
    if hasattr(dictionary, "items"):
        for k, v in dictionary.items():
            this_key = "".join([current_key, f'["{k}"]'])
            if k == key:
                yield {this_key: v}
            if isinstance(v, dict):
                for result in _gen_dict_extract(key, v, this_key):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in _gen_dict_extract(key, d, this_key):
                        yield result


def dict_extract(key, dictionary):
    """Wrapper for _gen_dict_extract

    Parameters
    ----------
    key: str
        The key to search for.
    dictionary: dict
        The nested dictionary to search.

    Returns
    -------
    dictionary : dict
        Extracted dictionary.
    """

    full_dictionary = {}
    for item in _gen_dict_extract(key, dictionary):
        full_dictionary.update(item)

    return full_dictionary


def class_from_yaml(cls, file, kwargs):
    file = Path(file)
    with file.open() as f:
        args = yaml.load(f, Loader=yaml.Loader)

    args.update(kwargs)

    signature = inspect.signature(cls)
    parameters = np.array(list(signature.parameters.items()))

    extra = [arg for arg in args if arg not in parameters]
    missing = np.array([parameter[0] not in args for parameter in parameters])
    has_default = np.array(
        [parameter[1].default is not parameter[1].empty for parameter in parameters]
    )
    allowed = ~missing | has_default
    using_default = missing & has_default

    actually_missing = parameters[~allowed]
    using_default = parameters[missing & has_default]

    if actually_missing.size > 0:
        raise ValueError(f"Missing arguments: {', '.join(actually_missing[:, 0])}")
    if extra:
        print(f"Extra arguments: {', '.join(extra)}")
    if using_default.size > 0:
        print(f"Using defaults for: {', '.join(using_default[:, 0])}")

    return cls(**{key: value for key, value in args.items() if key not in extra})


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
