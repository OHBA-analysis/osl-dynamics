"""Main VRAD models.

"""

from vrad.models.go import GO
from vrad.models.maro import MARO
from vrad.models.rigo import RIGO
from vrad.models.ridgo import RIDGO
from vrad.models.rimaro import RIMARO


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
            print("Using GO")
            return GO(config)

        elif config.observation_model == "multivariate_autoregressive":
            print("Using MARO")
            return MARO(config)

    if config.observation_model == "multivariate_normal":

        if config.alpha_pdf == "normal":
            print("Using RIGO")
            return RIGO(config)

        elif config.alpha_pdf == "dirichlet":
            print("Using RIDGO")
            return RIDGO(config)

    elif config.observation_model == "multivariate_autoregressive":

        if config.alpha_pdf == "normal":
            print("Using RIMARO")
            return RIMARO(config)

        elif config.alpha_pdf == "dirichlet":
            raise NotImplementedError("Requested config not available.")
