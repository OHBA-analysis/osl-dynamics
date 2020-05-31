import logging
from typing import Tuple

import numpy as np


def override_dict_defaults(default_dict: dict, override_dict: dict = None) -> dict:
    if override_dict is None:
        override_dict = {}

    return {**default_dict, **override_dict}


def listify(obj: object):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


def time_axis_first(input_array: np.ndarray) -> Tuple[np.ndarray, bool]:
    if input_array.ndim != 2:
        logging.info("Non-2D array not transposed.")
        return input_array, False
    transposed = False
    if input_array.shape[1] > input_array.shape[0]:
        input_array = input_array.T
        transposed = True
    return input_array, transposed
