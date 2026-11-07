Models
======

osl-dynamics implements several generative models for inferring dynamic functional brain networks from time series data. All of the models share the same interface: a ``Config`` object sets the hyperparameters, a ``Model`` object is trained on a :class:`~osl_dynamics.data.Data` object, and the inferred state or mode time courses are passed to the post-hoc analysis tools in :mod:`osl_dynamics.analysis`.

The pages in this section describe each model: the generative model, how it is trained and what post-hoc analysis is recommended.

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20

   * - Model
     - State type
     - Temporal model
     - Best for
   * - :doc:`HMM <hmm>`
     - Discrete (mutually exclusive)
     - Markovian (transition probability matrix)
     - Resting-state analysis; interpretable summary stats
   * - :doc:`DyNeMo <dynemo>`
     - Continuous (linear mixture of modes)
     - Non-Markovian (RNN)
     - Task data; overlapping network activity
   * - :doc:`DyNeSte <dyneste>`
     - Discrete (mutually exclusive)
     - Non-Markovian (RNN)
     - Discrete states with long-range dynamics
   * - :doc:`M-DyNeMo <mdynemo>`
     - Continuous (linear mixture; separate for power and FC)
     - Non-Markovian (RNN)
     - Separate power and connectivity dynamics
   * - :doc:`HIVE <hive>`
     - Discrete (mutually exclusive)
     - Markovian (transition probability matrix)
     - Modeling inter-session variability (e.g. subjects, scanners, sites)

The :doc:`FAQ <../faq>` has guidance on choosing a model and its hyperparameters. The :doc:`HMM Training <../tutorials_build/3-2_hmm_training>` and :doc:`DyNeMo Training <../tutorials_build/3-3_dynemo_training>` tutorials show the models applied to real data.

Further models are documented in the API reference only: :mod:`DIVE <osl_dynamics.models.dive>` (DyNeMo with embeddings for inter-session variability), :mod:`SC-DyNeMo <osl_dynamics.models.sc_dynemo>` (single-channel DyNeMo), :mod:`HMM-Poisson <osl_dynamics.models.hmm_poi>` (an HMM for spike count data) and :mod:`HMM-Dirichlet <osl_dynamics.models.hmm_dirichlet>`.

.. toctree::
   :maxdepth: 1

   hmm
   dynemo
   dyneste
   mdynemo
   hive
