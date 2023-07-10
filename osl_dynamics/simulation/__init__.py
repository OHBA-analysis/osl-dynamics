"""Simulations for time series data.

"""

from osl_dynamics.simulation.base import Simulation
from osl_dynamics.simulation.mar import MAR
from osl_dynamics.simulation.mvn import MVN, MDyn_MVN, MSubj_MVN
from osl_dynamics.simulation.hmm import (
    HMM,
    HMM_MAR,
    HMM_MVN,
    MDyn_HMM_MVN,
    HierarchicalHMM_MVN,
    MSubj_HMM_MVN,
)
from osl_dynamics.simulation.hsmm import HSMM, HSMM_MVN, MixedHSMM_MVN
from osl_dynamics.simulation.sm import MixedSine, MixedSine_MVN, MSubj_MixedSine_MVN
