"""
OPM: Preprocessing, Source Reconstruction, Parcellation (Without an MRI)
========================================================================

This tutorial walks through the full OPM-MEG processing pipeline when no
individual structural MRI is available. A template MRI is scaled to match
the participant's 3D head scan (e.g., EinScan) to generate a "pseudo-MRI"
that can be used for source reconstruction.

1. Preprocessing — Downsample, filter, detect bad segments/channels, ICA artefact rejection, decimate headshape points.
2. Pseudo-MRI Generation — Scale a template MRI to match the EinScan headshape.
3. Surface Extraction — Extract skull/scalp surfaces from the pseudo-MRI.
4. Coregistration — Align OPM sensor space to pseudo-MRI space.
5. Forward Model — Compute the lead field matrix.
6. Source Reconstruction — LCMV beamformer to project sensor data to source space.
7. Parcellation — Reduce voxel data to parcel time courses.

If you have an individual MRI, you can skip Step 2 and follow the standard
:doc:`MEG preprocessing tutorial </tutorials_build/0-1_meg_preprocessing>`
instead.

Prerequisites
^^^^^^^^^^^^^

- `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki>`_ (needed for surface extraction and pseudo-MRI generation).
- `osl-dynamics <https://github.com/OHBA-analysis/osl-dynamics>`_ (this installs `MNE-Python <https://mne.tools/stable/index.html>`_ as a dependency). Note, TensorFlow is not required for processing M/EEG. (osl-dynamics can be installed without TensorFlow using the `envs/osld.yml <https://github.com/OHBA-analysis/osl-dynamics/blob/main/envs/osld.yml>`_ environment.)
- (Optional) A custom template MRI. If not provided, the MNI152 standard brain from FSL is used. For paediatric data, an age-appropriate template (e.g. from the `Neurodevelopmental MRI Database <https://doi.org/10.1016/j.neuroimage.2015.04.055>`_) is recommended.

Input Data
^^^^^^^^^^

OPM data with EinScan headshape points stored in the FIF file's ``info['dig']``::

    BIDS/
    ├── sub-01/
    │   ├── meg/
    │   │   └── sub-01_task-verb_run-01_meg.fif
    ├── ...

Output is written to ``BIDS/derivatives/``.
"""

#%%
# Download the dataset
# ^^^^^^^^^^^^^^^^^^^^
# We will download example OPM data hosted on `OSF <https://osf.io/by2tc/>`_.

import os

def get_data(name):
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.system(f"unzip -o {name}.zip")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset
get_data("opm_1_subject")

#%%
# Setup and Configuration
# ^^^^^^^^^^^^^^^^^^^^^^^

from pathlib import Path

import mne
import numpy as np

from osl_dynamics.meeg import preproc, rhino, source_recon, parcellation
from osl_dynamics.utils.filenames import OSLFilenames

#%%
# Edit the cell below to match your data.

# Session info
subject = "01"
task = "verb"
run = "01"
id = f"sub-{subject}_task-{task}_run-{run}"

# Paths
bids_dir = Path("BIDS")
raw_file = bids_dir / f"sub-{subject}/meg/{id}_meg.fif"
output_dir = bids_dir / "derivatives"

# Template MRI (optional)
# If None, the MNI152 standard brain from FSL is used.
# For children or older adults, an age-matched template is recommended,
# e.g. from https://doi.org/10.1016/j.neuroimage.2015.04.055
template_mri = None

# Preprocessing parameters
resample_freq = 250  # Hz
bandpass = (1, 45)  # Hz
notch_freqs = [50, 100]  # Hz (mains frequency and harmonics)

# Source reconstruction parameters
gridstep = 8  # dipole grid resolution in mm
chantypes = "mag"  # OPMs are magnetometers
rank = {"mag": 100}  # adjust based on your system

# Parcellation
parcellation_file = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"

print(f"Session ID: {id}")
print(f"Raw file: {raw_file}")

#%%
# Step 1: Preprocessing
# ^^^^^^^^^^^^^^^^^^^^^
#
# We clean the sensor-level OPM data. This is similar to standard MEG
# preprocessing, but with two OPM-specific considerations:
#
# - OPMs use ``picks="mag"`` (they are magnetometers).
# - The EinScan headshape is very dense and should be decimated before
#   coregistration.
#
# Load raw data
# *************

raw = mne.io.read_raw_fif(raw_file, preload=True)

#%%
# Resample and filter
# *******************

raw = raw.resample(sfreq=resample_freq)
raw = raw.filter(
    l_freq=bandpass[0],
    h_freq=bandpass[1],
    method="iir",
    iir_params={"order": 5, "ftype": "butter"},
)
raw = raw.notch_filter(notch_freqs)

#%%
# Bad segment and channel detection
# **********************************

raw = preproc.detect_bad_segments(raw, picks="mag", significance_level=0.1)
raw = preproc.detect_bad_segments(raw, picks="mag", mode="diff", significance_level=0.1)
raw = preproc.detect_bad_segments(raw, picks="mag", metric="kurtosis", significance_level=0.1)
raw = preproc.detect_bad_channels(raw, picks="mag", significance_level=0.1)

#%%
# ICA artefact rejection
# **********************
#
# We use MEGNet automatic labelling (via ``mne-icalabel``) to identify and
# remove artefact components (e.g., cardiac, eye blinks). MEGNet was trained
# on magnetometer topographies, making it well suited for OPM data.

raw, ica, ic_labels = preproc.ica_label(raw, picks="mag", method="megnet")

#%%
# Decimate headshape points
# *************************
#
# The EinScan structured-light scanner produces a very dense mesh of the
# head, face, and neck. Many of these points are redundant and can cause
# the ICP algorithm used in coregistration to get stuck in local minima.
# We decimate the headshape to a manageable number of points.

raw = preproc.decimate_headshape_points(raw)

#%%
# Save preprocessed data
# **********************

preproc_file = output_dir / "preprocessed" / f"{id}_preproc-raw.fif"
preproc_file.parent.mkdir(parents=True, exist_ok=True)
raw.save(preproc_file, overwrite=True)
print(f"Saved: {preproc_file}")

#%%
# Step 2: Pseudo-MRI Generation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This step replaces the need for an individual structural MRI. We scale a
# template MRI to match the participant's head shape, as captured by the
# EinScan headshape points stored in the FIF file.
#
# The method:
#
# 1. Segments the template MRI to extract a scalp surface.
# 2. Aligns the EinScan headshape to the template scalp using fiducials + ICP.
# 3. Computes an affine warp (with anisotropic scaling) from surface
#    correspondences.
# 4. Applies the warp to the template MRI to produce the pseudo-MRI.
#
# The ``padding`` parameter (default 50 voxels) adds space around the template
# to accommodate heads larger than the template without clipping.

pseudo_mri_dir = str(output_dir / "pseudo_mri" / f"sub-{subject}")

pseudo_mri = rhino.generate_pseudo_mri(
    preproc_file=str(preproc_file),
    outdir=pseudo_mri_dir,
    template_mri_file=template_mri,
)

#%%
# .. note::
#
#     If you have an individual structural MRI, skip this step and pass the
#     MRI path directly to ``rhino.extract_surfaces`` in the next step.

#%%
# Step 3: Surface Extraction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We extract skull and scalp surfaces from the pseudo-MRI, just as we would
# from an individual structural MRI.

surfaces_dir = str(output_dir / "anat_surfaces" / f"sub-{subject}")

rhino.extract_surfaces(
    mri_file=pseudo_mri,
    outdir=surfaces_dir,
    include_nose=False,
    show=True,
)

#%%
# Check that the extracted surfaces (yellow lines) match the corresponding
# anatomical boundaries in the pseudo-MRI.

#%%
# Step 4: Coregistration
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We coregister the OPM sensor space to the pseudo-MRI using the EinScan
# headshape points and fiducials.

fns = OSLFilenames(
    outdir=str(output_dir / "osl"),
    id=id,
    preproc_file=str(preproc_file),
    surfaces_dir=surfaces_dir,
)

rhino.extract_fiducials_and_headshape_from_fif(fns)
rhino.coregister_head_and_mri(fns, use_nose=False, show=True)

#%%
# Check that the headshape points (red) sit on the scalp surface and the
# sensors (blue) surround the head correctly.

#%%
# Step 5: Forward Model
# ^^^^^^^^^^^^^^^^^^^^^
#
# Compute the forward model using a Single Layer (Single Shell) head model.

rhino.forward_model(fns, model="Single Layer", gridstep=gridstep)

#%%
# Step 6: Source Reconstruction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We compute and apply an LCMV beamformer. For OPMs, we use
# ``chantypes="mag"`` since OPMs are magnetometers.

source_recon.lcmv_beamformer(fns, raw, chantypes=chantypes, rank=rank)
voxel_data, voxel_coords = source_recon.apply_lcmv_beamformer(fns, raw)
print(f"Voxel data shape: {voxel_data.shape} (voxels x time)")

#%%
# Step 7: Parcellation
# ^^^^^^^^^^^^^^^^^^^^
#
# Reduce voxel data to parcel time courses using a brain atlas with symmetric
# orthogonalisation to reduce spatial leakage.

parcel_data = parcellation.parcellate(
    fns,
    voxel_data,
    voxel_coords,
    method="spatial_basis",
    orthogonalisation="symmetric",
    parcellation_file=parcellation_file,
)
print(f"Parcel data shape: {parcel_data.shape} (parcels x time)")

#%%
# Save parcellated data
# *********************

parc_fif = str(output_dir / "osl" / id / "lcmv-parc-raw.fif")
parcellation.save_as_fif(
    parcel_data,
    raw,
    extra_chans="stim",
    filename=parc_fif,
)
print(f"Saved: {parc_fif}")

#%%
# QC: plot PSDs. We expect posterior alpha (~10 Hz) if the source
# reconstruction is working correctly.

parcellation.save_qc_plots(parc_fif, parcellation_file, show=True)

#%%
# Summary and Next Steps
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We have completed the full OPM pipeline without an individual MRI:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 5 25 40
#
#    * - Step
#      - Name
#      - Output
#    * - 1
#      - Preprocessing
#      - ``BIDS/derivatives/preprocessed/{id}_preproc-raw.fif``
#    * - 2
#      - Pseudo-MRI
#      - ``BIDS/derivatives/pseudo_mri/sub-{subject}/pseudo_mri.nii.gz``
#    * - 3
#      - Surface extraction
#      - ``BIDS/derivatives/anat_surfaces/sub-{subject}/``
#    * - 4
#      - Coregistration
#      - ``BIDS/derivatives/osl/{id}/coreg/``
#    * - 5
#      - Forward model
#      - ``BIDS/derivatives/osl/{id}/coreg/model-fwd.fif``
#    * - 6
#      - Source reconstruction
#      - ``BIDS/derivatives/osl/{id}/src/filters-lcmv.h5``
#    * - 7
#      - Parcellation
#      - ``BIDS/derivatives/osl/{id}/lcmv-parc-raw.fif``
#
# Loading the parcellated data
# ****************************
#
# The final output can be loaded for downstream analysis::
#
#     from osl_dynamics.data import Data
#
#     data = Data(parc_fif, picks="misc", reject_by_annotation="omit")
#
# Where to go from here
# *********************
#
# - **Canonical HMM analysis** — See `here <https://github.com/OHBA-analysis/Canonical-HMM-Networks>`_ for notebooks applying a canonical HMM to the parcellated data.
# - **Analysis** — See the :doc:`documentation <../documentation>` for further tutorials for static and dynamic network analysis.
