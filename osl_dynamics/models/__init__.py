"""Implemented models.

"""
import yaml

from osl_dynamics.models import (
    dynemo,
    mage,
    sage,
    mdynemo,
    sedynemo,
    state_dynemo,
    hmm,
    sehmm,
)
from osl_dynamics.utils.misc import NumpyLoader

models = {
    "DyNeMo": dynemo.Model,
    "MAGE": mage.Model,
    "SAGE": sage.Model,
    "M-DyNeMo": mdynemo.Model,
    "SE-DyNeMo": sedynemo.Model,
    "State-DyNeMo": state_dynemo.Model,
    "HMM": hmm.Model,
    "SE-HMM": sehmm.Model,
}


def load(dirname):
    """Load model from dirname.

    Parameters
    ----------
    dirname : str
        Path to directory where the config.yml and weights are stored.

    Returns
    -------
    model : DyNeMo model
        Model object.
    """
    with open(f"{dirname}/config.yml", "r") as file:
        config_dict = yaml.load(file, NumpyLoader)

    if "model_name" not in config_dict:
        raise ValueError(
            "Either use a specific `Model.load` method or "
            + "provide a `model_name` field in config"
        )

    try:
        model_type = models[config_dict["model_name"]]
    except KeyError:
        raise NotImplementedError(
            f"{config_dict['model_name']} was not found. "
            + f"Options are {', '.join(models.keys())}"
        )

    return model_type.load(dirname)
