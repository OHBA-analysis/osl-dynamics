Tutorials
=========

The tutorials walk through each stage of an osl-dynamics analysis with real data: processing M/EEG data, loading and preparing data, static (time-averaged) analysis, training the dynamic network models, post-hoc analysis of the inferred networks, task analysis and statistical significance testing. Each tutorial can be downloaded as a Python script or Jupyter notebook from the links at the bottom of its page.

If you are new to osl-dynamics, start with the :doc:`Getting Started <getting_started>` guide, then follow the tutorials in order. Already have parcellated data? Start with the :doc:`Data tutorials <tutorial_sections/data>`.

.. toctree::
   :hidden:

   tutorial_sections/meeg_processing
   tutorial_sections/data
   tutorial_sections/static_analysis
   tutorial_sections/dynamic_modelling
   tutorial_sections/hmm_post_hoc
   tutorial_sections/dynemo_post_hoc
   tutorial_sections/task_analysis
   tutorial_sections/statistics

M/EEG Processing
----------------

From raw sensor recordings to parcellated source-space time courses: preprocessing, surface extraction, coregistration, source reconstruction, parcellation and sign flipping. See the :doc:`M/EEG Processing <meeg>` page for an overview of the steps.

.. minigallery::

   tutorials/0-1_meg_preprocessing.py
   tutorials/0-2_sign_flipping.py
   tutorials/0-3_meg_batch_processing.py
   tutorials/0-4_eeg_batch_processing.py
   tutorials/0-5_mne_sample_data.py

Data
----

Loading data into osl-dynamics and preparing it for modelling: standardisation, time-delay embedding and PCA.

.. minigallery::

   tutorials/1-1_data_loading.py
   tutorials/1-2_data_prepare_meg.py
   tutorials/1-3_data_prepare_fmri.py
   tutorials/1-4_data_time_delay_embedding.py

Static Analysis
---------------

Time-averaged analysis of MEG data: spectra, power maps and amplitude envelope correlation networks.

.. minigallery::

   tutorials/2-1_static_spectra_analysis.py
   tutorials/2-2_static_power_analysis.py
   tutorials/2-3_static_aec_analysis.py

Dynamic Modelling
-----------------

Inferring dynamic networks: sliding windows, training an HMM or DyNeMo, selecting a run and getting the inferred parameters.

.. minigallery::

   tutorials/3-1_sliding_window_analysis.py
   tutorials/3-2_hmm_training.py
   tutorials/3-3_dynemo_training.py
   tutorials/3-4_dynemo_selecting_a_run.py
   tutorials/3-5_hmm_dynemo_get_inf_params.py

HMM Post-hoc Analysis
---------------------

Analysing the states of a trained HMM: spectra, networks, summary statistics and fMRI dual estimation.

.. minigallery::

   tutorials/4-1_hmm_multitaper_spectra.py
   tutorials/4-2_hmm_plotting_meg_networks.py
   tutorials/4-3_hmm_summary_stats.py
   tutorials/4-4_hmm_fmri_dual_estimation.py
   tutorials/4-5_hmm_plotting_fmri_networks.py

DyNeMo Post-hoc Analysis
------------------------

Analysing the modes of a trained DyNeMo: regression spectra, networks and mixing coefficients.

.. minigallery::

   tutorials/5-1_dynemo_regression_spectra.py
   tutorials/5-2_dynemo_plotting_networks.py
   tutorials/5-3_dynemo_mixing_coefs.py

Task Analysis
-------------

Epoching state and mode time courses around events to get network responses.

.. minigallery::

   tutorials/6-1_epoching_alpha.py

Statistical Significance Testing
--------------------------------

GLM permutation tests for group contrasts, network responses, conditions and custom designs, with the background on the methods. See the :doc:`statistics page <stats>` for which test and correction to use.

.. minigallery::

   tutorials/7-1_statistical_inference.py
   tutorials/7-2_group_contrast.py
   tutorials/7-3_network_response.py
   tutorials/7-4_custom_designs.py

Examples Directory
------------------

Full analysis scripts, including the scripts used to produce the results in the `osl-dynamics paper <https://elifesciences.org/articles/91949>`_, can be found in the `examples directory <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples>`_ of the repository.

Workshops
---------

The `OHBA Methods Group <https://www.psych.ox.ac.uk/research/ohba-analysis-group>`_ organises teaching workshops for analysing M/EEG data using `osl-ephys <https://osl-ephys.readthedocs.io/en/latest/>`_ and osl-dynamics.

Links to past workshops:

- `2023 OSL workshop <https://osf.io/zxb6c/>`_.
- `2025 OSL workshop <https://github.com/OHBA-analysis/osl-workshop-2025-dynamics>`_.
