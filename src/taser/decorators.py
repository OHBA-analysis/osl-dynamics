import inspect
import logging
from functools import wraps
from time import time
import numpy as np

import yaml

from taser.helpers.misc import time_axis_first


def timing(f):
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


@doublewrap
def transpose(f, *arguments):
    try:
        iter(arguments)
    except TypeError:
        arguments = [arguments]
    finally:
        positional_arguments = [a for a in arguments if isinstance(a, int)]
        keyword_arguments = [a for a in arguments if isinstance(a, str)]

    @wraps(f)
    def wrap(*args, **kwargs):
        args = [*args]
        for i in positional_arguments:
            try:
                args[i], transposed = time_axis_first(args[i])
                if transposed:
                    logging.warning(
                        f"Argument {i}: assuming longer axis to be time and transposing."
                    )
            except IndexError:
                logging.debug(f"Positional argument {i} not in args.")
        for k in keyword_arguments:
            try:
                kwargs[k], transposed = time_axis_first(kwargs[k])
                if transposed:
                    logging.warning(
                        f"Argument {k}: assuming longer axis to be time and transposing."
                    )
            except KeyError:
                logging.debug(f"Argument {k} not found in kwargs.")

        return f(*args, **kwargs)

    return wrap


def get_params(me, args, kwargs):
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
