Toolbox Paper Scripts
---------------------

This directory contains scripts to reproduce the results shown in the toolbox paper:

- `ctf_rest` contains scripts for training a TDE-HMM a TDE-DyNeMo model on resting-state MEG data collected using a CTF scanner (Nottingham MEGUK).
- `elekta_task` contains scripts for training an AE-HMM and TDE-HMM on a visual task MEG data collected using an Elekta scanner (Wakeman-Henson).

Both datasets are publicly available.

Note: the original toolbox paper used the OHBA Software Library (`osl-ephys`) package to preprocess and source reconstruct the MEG data. That functionality has since been included in `osl-dynamics` (under `osl_dynamics.meeg`), and the scripts in `preproc/` now use it directly. `osl-ephys` is no longer required.
