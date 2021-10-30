"""Main DyNeMo models.

"""

from dynemo.models.go import GO
from dynemo.models.maro import MARO
from dynemo.models.cnno import CNNO
from dynemo.models.ricnno import RICNNO
from dynemo.models.ridgo import RIDGO
from dynemo.models.rigo import RIGO
from dynemo.models.rimaro import RIMARO


def Model(config):
    """Main DyNeMo model.

    Selects either an observation model (GO, MARO or CNNO) or joint inference
    and observation model (RIGO, RIDGO, RIMARO, RICNNO) based on the passed
    config.

    Parameters
    ----------
    config : dynemo.models.Config

    Returns
    -------
    A DyNeMo model class.
    """

    if config.inference_rnn is None:

        if config.observation_model == "multivariate_normal":
            return GO(config)

        elif config.observation_model == "multivariate_autoregressive":
            return MARO(config)

        elif config.observation_model == "wavenet":
            return CNNO(config)

    if config.observation_model == "multivariate_normal":

        if config.alpha_pdf == "normal":
            return RIGO(config)

        elif config.alpha_pdf == "dirichlet":
            return RIDGO(config)

    elif config.observation_model == "multivariate_autoregressive":

        if config.alpha_pdf == "normal":
            return RIMARO(config)

        elif config.alpha_pdf == "dirichlet":
            raise NotImplementedError("Requested config not available.")

    elif config.observation_model == "wavenet":

        if config.alpha_pdf == "normal":
            return RICNNO(config)

        elif config.alpha_pdf == "dirichlet":
            raise NotImplementedError("Requested config not available.")
