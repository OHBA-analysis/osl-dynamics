"""Implemented models.

"""
import yaml

from osl_dynamics.models import (
    dynemo,
    dynemo_obs,
    mage,
    sage,
    mdynemo,
    mdynemo_obs,
    sedynemo,
    sedynemo_obs,
    state_dynemo,
    hmm,
)
from osl_dynamics.utils.misc import NumpyLoader

models = {
    "DyNeMo": dynemo.Model,
    "DyNeMo-Obs": dynemo_obs.Model,
    "MAGE": mage.Model,
    "SAGE": sage.Model,
    "M-DyNeMo": mdynemo.Model,
    "M-DyNeMo-Obs": mdynemo_obs.Model,
    "SE-DyNeMo": sedynemo.Model,
    "SE-DyNeMo-Obs": sedynemo_obs.Model,
    "State-DyNeMo": state_dynemo.Model,
    "HMM": hmm.Model,
}


def load(filepath):
    """Load model from filepath.

    Parameters
    ----------
    filepath : str
        Path to directory where the config.yml and weights are stored.

    Returns
    -------
    model : DyNeMo model
        Model object.
    """
    with open(f"{filepath}/config.yml", "r") as f:
        config_dict = yaml.load(f, NumpyLoader)

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

    return model_type.load(filepath)
