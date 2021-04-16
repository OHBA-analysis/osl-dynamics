"""Main VRAD models.

"""

from vrad.models.go import GO
from vrad.models.maro import MARO
from vrad.models.rigo import RIGO
from vrad.models.ridgo import RIDGO
from vrad.models.rimaro import RIMARO


def Model(config):
    """A joint inference and observation model.

    Parameters
    ----------
    config : vrad.models.Config

    Returns
    -------
    A VRAD model class.
    """

    if config.inference_rnn is None:
        raise ValueError(
            "Inference network parameters not passed. Use ObservationModel."
        )

    if config.observation_model == "multivariate_normal":

        if config.alpha_pdf == "normal":
            print("Using RIGO")
            return RIGO(config)

        elif config.alpha_pdf == "dirichlet":
            print("Using RIDGO")
            return RIDGO(config)

        else:
            raise NotImplementedError("Requested config not implemented.")

    elif config.observation_model == "multivariate_autoregressive":

        if config.alpha_pdf == "normal":
            print("Using RIMARO")
            return RIMARO(config)

        else:
            raise NotImplementedError("Requested config not implemented.")

    else:
        raise NotImplementedError("Requested config not implemented.")


def ObservationModel(config):
    """An observation model.

    Parameters
    ----------
    config : vrad.models.Config

    Returns
    -------
    A VRAD model class.
    """

    if config.observation_model == "multivariate_normal":
        print("Using GO")
        return GO(config)

    elif config.observation_model == "multivariate_autoregressive":
        print("Using MARO")
        return MARO(config)

    else:
        raise NotImplementedError("Requested config not implemented.")
