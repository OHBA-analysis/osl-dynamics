"""M/EEG processing pipeline.

This subpackage provides a complete pipeline for processing M/EEG data from
raw sensor recordings to parcellated source-space time courses.

Modules
-------
- :py:mod:`~osl_dynamics.meeg.preproc` — Sensor-level preprocessing
  (filtering, bad segment/channel detection, QC plots).
- :py:mod:`~osl_dynamics.meeg.rhino` — Surface extraction, coregistration
  (RHINO), and forward modelling.
- :py:mod:`~osl_dynamics.meeg.source_recon` — Source reconstruction (LCMV
  beamformer).
- :py:mod:`~osl_dynamics.meeg.parcellation` — Parcellation of voxel data
  into parcel time courses, QC plots.
- :py:mod:`~osl_dynamics.meeg.parallel` — Utilities for running pipeline
  steps on multiple sessions in parallel.
- :py:mod:`~osl_dynamics.meeg.report` — HTML QC report generation.

Tutorials
---------
- :doc:`MEG Preprocessing </tutorials_build/0-1_meg_preprocessing>`

Python example scripts
----------------------
- `Batch MEG preprocessing <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/meg_preproc>`_
"""
