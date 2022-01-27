"""Main DyNeMo models.

"""

from dynemo.models.go import GO
from dynemo.models.rigo import RIGO
from dynemo.models.mrigo import MRIGO


def Model(config):
    """Main DyNeMo model.

    Selects either an observation model (GO) or joint inference
    and observation model (RIGO, MRIGO) based on the passed config.

    Parameters
    ----------
    config : dynemo.models.Config

    Returns
    -------
    A DyNeMo model class.
    """

    if config.inference_rnn is None:
        return GO(config)


    if config.multiple_scales:
        return MRIGO(config)

    else:
        return RIGO(config)
