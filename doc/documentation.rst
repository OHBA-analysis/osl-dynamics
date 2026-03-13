Documentation
=============

Welcome to the osl-dynamics documentation!

New users can start with the :doc:`Getting Started <getting_started>` guide for a quick introduction and then work through the tutorials below. The :doc:`FAQ <faq>` covers common questions about data preparation, model training, and post-hoc analysis.

API Reference
-------------

The :doc:`API reference <autoapi/index>` provides documentation for all classes, methods, and functions in osl-dynamics.

Models
------

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20

   * - Model
     - State type
     - Temporal model
     - Best for
   * - :doc:`HMM <models/hmm>`
     - Discrete (mutually exclusive)
     - Markovian (transition probability matrix)
     - Resting-state analysis; interpretable summary stats
   * - :doc:`DyNeMo <models/dynemo>`
     - Continuous (linear mixture of modes)
     - Non-Markovian (RNN)
     - Task data; overlapping network activity
   * - :doc:`DyNeSte <models/dyneste>`
     - Discrete (mutually exclusive)
     - Non-Markovian (RNN)
     - Discrete states with long-range dynamics
   * - :doc:`M-DyNeMo <models/mdynemo>`
     - Continuous (linear mixture; separate for power and FC)
     - Non-Markovian (RNN)
     - Separate power and connectivity dynamics
   * - :doc:`HIVE <models/hive>`
     - Discrete (mutually exclusive)
     - Markovian (transition probability matrix)
     - Modelling inter-session variability (e.g. subjects, scanners, sites)

Also see the :doc:`FAQ <faq>` for guidance on choosing a model and hyperparameters.

Parcellations
-------------

For information regarding the parcellations available in osl-dynamics, see :doc:`here <parcellations/index>`.

Tutorials
---------

The following tutorials illustrate basic usage and analysis that can be done with osl-dynamics.

**M/EEG processing tutorial**:

- :doc:`tutorials_build/0-1_meg_preprocessing`.

**Data tutorials**:

- :doc:`tutorials_build/1-1_data_loading`.
- :doc:`tutorials_build/1-2_data_prepare_meg`.
- :doc:`tutorials_build/1-3_data_prepare_fmri`.
- :doc:`tutorials_build/1-4_data_time_delay_embedding`.

**Static (time-averaged) modelling tutorials for MEG**:

- :doc:`tutorials_build/2-1_static_spectra_analysis`.
- :doc:`tutorials_build/2-2_static_power_analysis`.
- :doc:`tutorials_build/2-3_static_aec_analysis`.

**Dynamic modelling tutorials**:

- :doc:`tutorials_build/3-1_sliding_window_analysis`.
- :doc:`tutorials_build/3-2_hmm_training`.
- :doc:`tutorials_build/3-3_dynemo_training`.
- :doc:`tutorials_build/3-4_hmm_dynemo_get_inf_params`.

**HMM post-hoc analysis tutorials**:

- :doc:`tutorials_build/4-1_hmm_multitaper_spectra`.
- :doc:`tutorials_build/4-2_hmm_plotting_meg_networks`.
- :doc:`tutorials_build/4-3_hmm_summary_stats`.
- :doc:`tutorials_build/4-4_hmm_fmri_dual_estimation`.
- :doc:`tutorials_build/4-5_hmm_plotting_fmri_networks`.

**DyNeMo post-hoc analysis tutorials**:

- :doc:`tutorials_build/5-1_dynemo_regression_spectra`.
- :doc:`tutorials_build/5-2_dynemo_plotting_networks`.
- :doc:`tutorials_build/5-3_dynemo_mixing_coefs`.

**Task analysis tutorials**:

- :doc:`tutorials_build/6-1_epoching_alpha`.
- `Dynamic Network Analysis of Electrophysiological Task Data <https://github.com/OHBA-analysis/Gohil2024_NetworkAnalysisOfTaskData>`_.

**Group-level analysis tutorials**:

- :doc:`tutorials_build/7-1_group_contrast`.
- :doc:`tutorials_build/7-2_group_network_response`.

Examples Directory
------------------

More examples scripts can be found in the `examples directory <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples>`_ of the repository.

Workshops
---------

The `OHBA Methods Group <https://www.psych.ox.ac.uk/research/ohba-analysis-group>`_ organises teaching workshops for analysing M/EEG data using `osl-ephys <https://osl-ephys.readthedocs.io/en/latest/>`_ and osl-dynamics.

Links to past workshops:

- `2023 OSL workshop <https://osf.io/zxb6c/>`_.
- `2025 OSL workshop <https://github.com/OHBA-analysis/osl-workshop-2025-dynamics>`_.
