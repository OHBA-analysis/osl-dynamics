"""Main DyNeMo models.

"""

from dynemo.models.go import GO
from dynemo.models.maro import MARO
from dynemo.models.wno import WNO
from dynemo.models.riwno import RIWNO
from dynemo.models.rigo import RIGO
from dynemo.models.rimaro import RIMARO
from dynemo.models.mrigo import MRIGO


def Model(config):
    """Main DyNeMo model.

    Selects either an observation model (GO, MARO or WNO) or joint inference
    and observation model (RIGO, RIMARO, RIWNO) based on the passed
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
            return WNO(config)

    if config.observation_model == "multivariate_normal":

        if config.multiple_scales:
            return MRIGO(config)

        else:
            return RIGO(config)

    elif config.observation_model == "multivariate_autoregressive":
        return RIMARO(config)

    elif config.observation_model == "wavenet":
        return RIWNO(config)
