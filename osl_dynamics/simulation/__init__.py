"""Simulations for time series data.

This subpackage generates synthetic time series data with known ground truth
dynamics. A simulation combines a **state model**, which generates a time
course, with an **observation model**, which generates data given the time
course:

.. code-block:: python

    from osl_dynamics import simulation

    sim = simulation.Simulation(
        state_model=simulation.HMM(trans_prob="sequence", stay_prob=0.9, n_states=4),
        obs_model=simulation.MVN(means="zero", covariances="random", n_channels=10),
        n_samples=25600,
    )
    sim.standardize()
    sim.time_series, sim.state_time_course, sim.covariances

A state model generates either mutually exclusive **states**, giving a
simulation with ``n_states`` and ``state_time_course``, or a mixture of
**modes**, giving a simulation with ``n_modes`` and ``mode_time_course``. The
observation model takes the number of states or modes from the state model.

State models
------------
States:

- :py:class:`HMM` — Hidden Markov Model.
- :py:class:`HSMM` — Hidden Semi-Markov Model: state lifetimes follow a Gamma
  distribution.
- :py:class:`HierarchicalHMM` — a top-level HMM selects which bottom-level HMM
  generates the states.
- :py:class:`MDynHMM` — two independent HMMs for multi-time-scale dynamics.

Modes:

- :py:class:`MixedSine` — softmax-normalized sinusoids.
- :py:class:`MixedHSMM` — HSMM where some states are mixtures of modes.

Observation models
------------------
- :py:class:`MVN` — multivariate normal: a mean vector and covariance matrix
  for each state or mode.
- :py:class:`OscillatoryMVN` — multivariate normal with a spectral peak at a
  different frequency for each state or mode (AR(2) processes), and optional
  aperiodic (1/f) background activity.
- :py:class:`MDyn_MVN` — multivariate normal with separate time courses for
  the means/standard deviations and the correlations.
- :py:class:`MSess_MVN` — multivariate normal with session-specific means and
  covariances.
- :py:class:`MAR` — multivariate autoregressive (states only).
- :py:class:`TDECovs` — data generated from time-delay embedded covariances.
- :py:class:`OscillatoryBursts` — sinusoidal bursts (states only).
- :py:class:`Poisson` — Poisson counts (states only).
- :py:class:`Dirichlet` — Dirichlet compositions (states only).

Simulations
-----------
:py:class:`Simulation` combines any state model with any observation model.
The following classes combine common pairs:

- :py:class:`HMM_MVN`, :py:class:`HMM_OscillatoryMVN`, :py:class:`HMM_MAR`,
  :py:class:`HMM_TDECovs`, :py:class:`HMM_OscillatoryBursts`,
  :py:class:`HMM_Poi`, :py:class:`HMM_Dirichlet`.
- :py:class:`HSMM_MVN`, :py:class:`MixedHSMM_MVN`,
  :py:class:`HierarchicalHMM_MVN`, :py:class:`MDyn_HMM_MVN`.
- :py:class:`MixedSine_MVN`.
- ``MSess_`` prefix — session-specific observation model parameters:
  :py:class:`MSess_HMM_MVN`, :py:class:`MSess_MixedSine_MVN`.

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
    OscillatoryMVN,
    Poisson,
    TDECovs,
    Dirichlet,
)
from osl_dynamics.simulation.hmm import (
    HMM,
    HierarchicalHMM,
    MDynHMM,
    HMM_MAR,
    HMM_MVN,
    HMM_OscillatoryBursts,
    HMM_OscillatoryMVN,
    HMM_TDECovs,
    MDyn_HMM_MVN,
    HierarchicalHMM_MVN,
    MSess_HMM_MVN,
    HMM_Poi,
    HMM_Dirichlet,
)
from osl_dynamics.simulation.hsmm import HSMM, MixedHSMM, HSMM_MVN, MixedHSMM_MVN
from osl_dynamics.simulation.sm import MixedSine, MixedSine_MVN, MSess_MixedSine_MVN
