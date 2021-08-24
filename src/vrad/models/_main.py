"""Main VRAD models.

"""

from vrad.models.go import GO
from vrad.models.maro import MARO
from vrad.models.ridgo import RIDGO
from vrad.models.rigo import RIGO
from vrad.models.rimaro import RIMARO
from vrad.models.rivqgo import RIVQGO


def Model(config):
    """Main VRAD model.

    Selects either an observation model (GO or MARO) or joint inference
    and observation model (RIGO, RIDGO, RIMARO) based on the passed config.

    Parameters
    ----------
    config : vrad.models.Config

    Returns
    -------
    A VRAD model class.
    """

    if config.inference_rnn is None:

        if config.observation_model == "multivariate_normal":
            return GO(config)

        elif config.observation_model == "multivariate_autoregressive":
            return MARO(config)

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
