"""Simulations for time series data.

This subpackage provides tools for generating synthetic time series data
with known ground truth dynamics. Simulations are built by combining a
**state model** (how states evolve over time) with an **observation model**
(how data is generated given the active state).

State models
------------
- :py:class:`HMM` — Hidden Markov Model. Generates discrete state sequences
  with Markovian transitions.
- :py:class:`HSMM` — Hidden Semi-Markov Model. Like HMM but state lifetimes
  follow a Gamma distribution rather than a geometric distribution.
- :py:class:`MixedSine` — Sinusoidal soft mixtures. Generates smooth,
  overlapping mode time courses using softmax-normalised sinusoids.

Observation models
------------------
- :py:class:`MVN` — Multivariate normal. Each state has a mean vector and
  covariance matrix.
- :py:class:`MAR` — Multivariate autoregressive. Each state has a set of
  autoregressive coefficients and an error covariance.
- :py:class:`OscillatoryBursts` — Oscillatory bursts. Each state has a
  frequency and a set of active channels; generates sinusoidal signals
  during active periods.
- :py:class:`TDECovs` — TDE covariance. Each state has a TDE covariance
  matrix; generates autoregressive data via conditional sampling.
- :py:class:`Poisson` — Poisson. Each state has a rate vector.

Combined simulation classes
---------------------------
These combine a state model with an observation model:

- :py:class:`HMM_MVN` — HMM + MVN.
- :py:class:`HMM_MAR` — HMM + MAR.
- :py:class:`HMM_OscillatoryBursts` — HMM + oscillatory bursts.
- :py:class:`HMM_TDECovs` — HMM + TDE covariances.
- :py:class:`HMM_Poi` — HMM + Poisson.
- :py:class:`HSMM_MVN` — HSMM + MVN.
- :py:class:`MixedHSMM_MVN` — HSMM with overlapping states + MVN.
- :py:class:`MixedSine_MVN` — Sinusoidal mixing + MVN.

Variants
--------
- ``MDyn_`` prefix — Multi-time-scale dynamics (separate HMMs for power and
  connectivity): :py:class:`MDyn_HMM_MVN`.
- ``MSess_`` prefix — Multi-session data with session-specific parameters:
  :py:class:`MSess_HMM_MVN`, :py:class:`MSess_MixedSine_MVN`.
- :py:class:`HierarchicalHMM_MVN` — Two-level hierarchical HMM where a
  top-level HMM selects which bottom-level HMM is active.

Python example scripts
----------------------
- `Simulation <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/simulation>`_
"""

from osl_dynamics.simulation.base import Simulation
from osl_dynamics.simulation.obs_mod import (
    MAR,
    MVN,
    MDyn_MVN,
    MSess_MVN,
    OscillatoryBursts,
    Poisson,
    TDECovs,
)
from osl_dynamics.simulation.hmm import (
    HMM,
    HMM_MAR,
    HMM_MVN,
    HMM_OscillatoryBursts,
    HMM_TDECovs,
    MDyn_HMM_MVN,
    HierarchicalHMM_MVN,
    MSess_HMM_MVN,
    HMM_Poi,
)
from osl_dynamics.simulation.hsmm import HSMM, HSMM_MVN, MixedHSMM_MVN
from osl_dynamics.simulation.sm import MixedSine, MixedSine_MVN, MSess_MixedSine_MVN
