"""Simulations for MEG data

This module allows the user to conveniently simulate MEG data. Instantiating the
`Simulation` class automatically takes the user's input parameters and produces data
which can be analysed.

isort:skip_file
"""

from osl_dynamics.simulation._base import *  # noqa
from osl_dynamics.simulation._sin import *  # noqa
from osl_dynamics.simulation._mar import *  # noqa
from osl_dynamics.simulation._mvn import *  # noqa
from osl_dynamics.simulation._hmm import *  # noqa
from osl_dynamics.simulation._hsmm import *  # noqa
from osl_dynamics.simulation._sm import *  # noqa
