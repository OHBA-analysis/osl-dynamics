"""Simulations for MEG data

This module allows the user to conveniently simulate MEG data. Instantiating the
`Simulation` class automatically takes the user's input parameters and produces data
which can be analysed.

isort:skip_file
"""

from ohba_models.simulation._base import *  # noqa
from ohba_models.simulation._sin import *  # noqa
from ohba_models.simulation._mar import *  # noqa
from ohba_models.simulation._mvn import *  # noqa
from ohba_models.simulation._hmm import *  # noqa
from ohba_models.simulation._hsmm import *  # noqa
from ohba_models.simulation._sm import *  # noqa
