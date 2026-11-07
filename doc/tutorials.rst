Tutorials
=========

The tutorials walk through each stage of an osl-dynamics analysis with real data: processing M/EEG data, loading and preparing data, static (time-averaged) analysis, training the dynamic network models, post-hoc analysis of the inferred networks, task analysis and statistical significance testing. Each tutorial can be downloaded as a Python script or Jupyter notebook from the links at the bottom of its page.

If you are new to osl-dynamics, start with the :doc:`Getting Started <getting_started>` guide, then follow the tutorials in order. Already have parcellated data? Start with the :doc:`Data tutorials <tutorials_build/1-1_data_loading>`.

.. toctree::
   :maxdepth: 1
   :caption: M/EEG processing

   tutorials_build/0-1_meg_preprocessing
   tutorials_build/0-2_sign_flipping
   tutorials_build/0-3_meg_batch_processing
   tutorials_build/0-4_eeg_batch_processing
   tutorials_build/0-5_mne_sample_data

Also see `Canonical-HMM-Networks <https://github.com/OHBA-analysis/Canonical-HMM-Networks>`_ for start-to-end tutorials with Elekta, CTF, OPM and EEG data.

.. toctree::
   :maxdepth: 1
   :caption: Data

   tutorials_build/1-1_data_loading
   tutorials_build/1-2_data_prepare_meg
   tutorials_build/1-3_data_prepare_fmri
   tutorials_build/1-4_data_time_delay_embedding

.. toctree::
   :maxdepth: 1
   :caption: Static (time-averaged) analysis

   tutorials_build/2-1_static_spectra_analysis
   tutorials_build/2-2_static_power_analysis
   tutorials_build/2-3_static_aec_analysis

.. toctree::
   :maxdepth: 1
   :caption: Dynamic modelling

   tutorials_build/3-1_sliding_window_analysis
   tutorials_build/3-2_hmm_training
   tutorials_build/3-3_dynemo_training
   tutorials_build/3-4_dynemo_selecting_a_run
   tutorials_build/3-5_hmm_dynemo_get_inf_params

.. toctree::
   :maxdepth: 1
   :caption: HMM post-hoc analysis

   tutorials_build/4-1_hmm_multitaper_spectra
   tutorials_build/4-2_hmm_plotting_meg_networks
   tutorials_build/4-3_hmm_summary_stats
   tutorials_build/4-4_hmm_fmri_dual_estimation
   tutorials_build/4-5_hmm_plotting_fmri_networks

.. toctree::
   :maxdepth: 1
   :caption: DyNeMo post-hoc analysis

   tutorials_build/5-1_dynemo_regression_spectra
   tutorials_build/5-2_dynemo_plotting_networks
   tutorials_build/5-3_dynemo_mixing_coefs

.. toctree::
   :maxdepth: 1
   :caption: Task analysis

   tutorials_build/6-1_epoching_alpha

Also see `Dynamic Network Analysis of Electrophysiological Task Data <https://github.com/OHBA-analysis/Gohil2024_NetworkAnalysisOfTaskData>`_ for a full task analysis.

.. toctree::
   :maxdepth: 1
   :caption: Statistical significance testing

   tutorials_build/7-1_statistical_inference
   tutorials_build/7-2_group_contrast
   tutorials_build/7-3_network_response
   tutorials_build/7-4_custom_designs
