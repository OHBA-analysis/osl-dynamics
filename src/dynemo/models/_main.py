"""Main DyNeMo models.

"""

from dynemo.models.go import GO
from dynemo.models.maro import MARO
from dynemo.models.cnno import CNNO
from dynemo.models.ridgo import RIDGO
from dynemo.models.rigo import RIGO
from dynemo.models.rimaro import RIMARO
from dynemo.models.rivqgo import RIVQGO


def Model(config):
    """Main DyNeMo model.

    Selects either an observation model (GO or MARO) or joint inference
    and observation model (RIGO, RIDGO, RIMARO) based on the passed config.

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

        elif config.observation_model == "conv_net":
            return CNNO(config)

    if config.observation_model == "multivariate_normal":

        if config.alpha_pdf == "normal":
            if config.n_quantized_vectors is not None:
                return RIVQGO(config)
            else:
                return RIGO(config)

        elif config.alpha_pdf == "dirichlet":
            return RIDGO(config)

    elif config.observation_model == "multivariate_autoregressive":

        if config.alpha_pdf == "normal":
            if config.n_quantized_vectors is not None:
                raise NotImplementedError("Requested config not available.")
            else:
                return RIMARO(config)

        elif config.alpha_pdf == "dirichlet":
            raise NotImplementedError("Requested config not available.")
