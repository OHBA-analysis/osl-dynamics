"""M/EEG processing pipeline.

This subpackage provides a complete pipeline for processing M/EEG data from
raw sensor recordings to parcellated source-space time courses.

Tutorials
---------
- :doc:`MEG Preprocessing </tutorials_build/0-1_meg_preprocessing>`

Python example scripts
----------------------
- `Batch MEG preprocessing <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/meg_preproc>`_

Modules
-------
- ``preproc.py`` — Sensor-level preprocessing (filtering, bad segment/channel
  detection, QC plots).
- ``rhino.py`` — Surface extraction, coregistration (RHINO), and forward
  modelling.
- ``source_recon.py`` — Source reconstruction (LCMV beamformer).
- ``parcellation.py`` — Parcellation of voxel data into parcel time courses,
  QC plots.
- ``parallel.py`` — Utilities for running pipeline steps on multiple sessions
  in parallel.
- ``report.py`` — HTML QC report generation.
"""
