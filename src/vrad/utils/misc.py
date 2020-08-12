import inspect
import logging
from copy import copy

import numpy as np

_logger = logging.getLogger("VRAD")


def override_dict_defaults(default_dict: dict, override_dict: dict = None) -> dict:
    if override_dict is None:
        override_dict = {}

    return {**default_dict, **override_dict}


def listify(obj: object):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    else:
        return [obj]


def replace_argument(func, name, item, args, kwargs, append=False):
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


def check_iterable_type(iterable, object_type: type):
    """Check iterable is non-empty and contains only objects of specific type."""
    if isinstance(iterable, np.ndarray):
        return iterable.dtype == object_type
    return bool(iterable) and all(isinstance(elem, object_type) for elem in iterable)


def time_axis_first(input_array: np.ndarray) -> np.ndarray:
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
