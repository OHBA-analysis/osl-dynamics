"""Decorators.

"""

import inspect
import warnings
from functools import wraps
from time import time

import numpy as np
import yaml


def timing(f):
    """Decorator to print function execution time.

    Parameters
    ----------
    f : func
        Function to be decorated.

    Returns
    -------
    wrap : func
        Decorated function.
    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"{f.__name__!r} took: {te - ts:2.4f} sec")
        return result

    return wrap


def doublewrap(f):
    """
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator

    Parameters
    ----------
    f : func
        Function to be decorated.
    """

    @wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec


def get_params(me, args, kwargs):
    """Get a dictionary of parameters and their defaults for a class.

    Parameters
    ----------
    me : class
        Class to be analyzed.
    args : list
        Arguments to the class constructor.
    kwargs : dict
        Keyword arguments to the class constructor.

    Returns
    -------
    arg_dict : dict
        Dictionary of parameters and defaults.
    """
    arg_dict = {}

    params = inspect.signature(me.__class__).parameters

    for i, (key, val) in enumerate(params.items()):
        if i < len(args):
            arg_dict[key] = args[i]
        if i >= len(args):
            if key in kwargs:
                arg_dict[key] = kwargs[key]
            elif key in params:
                arg_dict[key] = params[key].default
    return arg_dict


def auto_repr(func):
    """Decorator to automatically generate a __repr__ method from input parameters.

    Parameters
    ----------
    func : func
        Function to be decorated (typically __init__).

    Returns
    -------
    wrapper_function : func
        Decorated function.
    """

    @wraps(func)
    def wrapper_function(me, *args, **kwargs):
        me.__arg_dict = get_params(me, args, kwargs)

        def __repr__(self):
            return_string = [self.__class__.__name__, "("]
            var_string = []
            for item in me.__arg_dict.items():
                if isinstance(item[1], str):
                    var_string.append(f"{item[0]}='{item[1]}'")
                else:
                    var_string.append(f"{item[0]}={item[1]}")
            return_string.append(", ".join(var_string))
            return_string.append(")")
            return "".join(return_string)

        setattr(me.__class__, "__repr__", __repr__)

        func(me, *args, **kwargs)

    return wrapper_function


def auto_str(func):
    """Decorator to automatically generate a __str__ method from input parameters.

    Parameters
    ----------
    func : func
        Function to be decorated (typically __init__).

    Returns
    -------
    wrapper_function : func
        Decorated function.
    """

    @wraps(func)
    def wrapper_function(me, *args, **kwargs):
        me.__arg_dict = get_params(me, args, kwargs)

        def __str__(self):
            str_output = [f"{self.__class__.__name__}:"]
            for item in self.__arg_dict.items():
                if isinstance(item[1], str):
                    str_output.append(f"{item[0]}: '{item[1]}'")
                else:
                    str_output.append(f"{item[0]}: {item[1]}")

            return "\n  ".join(str_output)

        setattr(me.__class__, "__str__", __str__)

        func(me, *args, **kwargs)

    return wrapper_function


def auto_yaml(func):
    """Decorator to automatically generate a yaml representation from input parameters.

    Parameters
    ----------
    func : func
        Function to be decorated (typically __init__).

    Returns
    -------
    wrapper_function : func
        Decorated function.
    """

    @wraps(func)
    def wrapper_function(me, *args, **kwargs):
        me.__arg_dict = get_params(me, args, kwargs)

        def __str__(self):
            yaml_safe_arg_dict = self.__arg_dict.copy()
            for key in yaml_safe_arg_dict:
                if isinstance(yaml_safe_arg_dict[key], np.ndarray):
                    yaml_safe_arg_dict[
                        key
                    ] = f"numpy array with shape {yaml_safe_arg_dict[key].shape}"
            return yaml.dump({self.__class__.__name__: yaml_safe_arg_dict})

        setattr(me.__class__, "__str__", __str__)

        func(me, *args, **kwargs)

    return wrapper_function


@doublewrap
def deprecated(f, *, replaced_by: str = None, reason: str = None):
    """Decorator to mark functions as deprecated.

    Parameters
    ----------
    f : func
        Function being deprecated.
    replaced_by : str
        Optional name of function to use instead.
    reason : str
        Message explaining the decision to deprecate.

    Returns
    -------
    wrapper_function: func
        Decorated function.
    """

    @wraps(f)
    def wrapper_function(*args, **kwargs):
        message = [f"The '{f.__name__}' function is deprecated."]
        if replaced_by is not None:
            message.append(f"Replace it with '{replaced_by}'.")
        if reason is not None:
            message.append(reason)
        message = " ".join(message)
        warnings.warn(message, DeprecationWarning)
        return f(*args, **kwargs)

    return wrapper_function
