"""Simulations for MEG data

This module allows the user to conveniently simulate MEG data. Instantiating the
`Simulation` class automatically takes the user's input parameters and produces data
which can be analysed.

isort:skip_file
"""

from dynemo.simulation._base import *  # noqa
from dynemo.simulation._sin import *  # noqa
from dynemo.simulation._mar import *  # noqa
from dynemo.simulation._mvn import *  # noqa
from dynemo.simulation._hmm import *  # noqa
from dynemo.simulation._hsmm import *  # noqa
from dynemo.simulation._sm import *  # noqa
