"""
MEG Preprocessing
=================

This tutorial walks through the full MEG preprocessing pipeline step by step:

1. Preprocessing — Downsample, filter, detect bad segments/channels.
2. Surface Extraction — Extract skull/scalp surfaces from a structural MRI.
3. Coregistration — Align MEG sensor space to MRI space.
4. Forward Model — Compute the lead field matrix.
5. Source Reconstruction — LCMV beamformer to project sensor data to source space.
6. Parcellation — Reduce voxel data to parcel time courses.

Prerequisites
^^^^^^^^^^^^^

- `osl-dynamics <https://github.com/OHBA-analysis/osl-dynamics>`_ (this installs `MNE-Python <https://mne.tools/stable/index.html>`_ as a dependency).
- `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki>`_ (needed for surface extraction).

Input Data
^^^^^^^^^^

In this tutorial, we will use data from the `Wakeman-Henson dataset <https://www.nature.com/articles/sdata20151>`_. This is a visual perception task recorded on an Elekta MEG system with a co-registered structural MRI.

We expect data in BIDS format::

    BIDS/
    ├── sub-01/
    │   ├── meg/
    │   │   └── sub-01_run-01_task-visual_raw_meg.fif
    │   └── anat/
    │       └── sub-01_T1w.nii.gz
    ├── ...

Output is written to ``BIDS/derivatives/``.

Batch Processing
^^^^^^^^^^^^^^^^

This tutorial processes a single session interactively. For processing many sessions in parallel, see the batch scripts in the `examples/meg_preproc <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/meg_preproc>`_ directory.
"""

#%%
# Download the dataset
# ^^^^^^^^^^^^^^^^^^^^
# We will download example data hosted on `OSF <https://osf.io/by2tc/>`_.

import os
import matplotlib
matplotlib.use("Agg")

def get_data(name):
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    os.system(f"unzip -o {name}.zip")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset
get_data("wakeman_henson_raw_1_subject")

#%%
# Setup and Configuration
# ^^^^^^^^^^^^^^^^^^^^^^^

from pathlib import Path

import mne

from osl_dynamics import files
from osl_dynamics.meeg import preproc, rhino, source_recon, parcellation
from osl_dynamics.utils.filenames import OSLFilenames

#%%
# Edit the cell below to match your data. This is the only cell you need to change.

# Session info
subject = "01"
run = "01"
task = "visual"
id = f"sub-{subject}_run-{run}_task-{task}"

# Paths
bids_dir = Path("BIDS")
raw_file = bids_dir / f"sub-{subject}/meg/sub-{subject}_run-{run}_task-{task}_raw_meg.fif"
mri_file = bids_dir / f"sub-{subject}/anat/sub-{subject}_T1w.nii.gz"
output_dir = bids_dir / "derivatives"
plots_dir = Path("plots")

# Preprocessing parameters
resample_freq = 250  # Hz
bandpass = (1, 45)  # Hz
notch_freqs = [50, 100]  # Hz (mains frequency and harmonics)

# Source reconstruction parameters
gridstep = 8  # dipole grid resolution in mm
chantypes = ["mag", "grad"]  # Elekta has both magnetometers and gradiometers
rank = {"meg": 60}  # effective rank after MaxFilter

# Parcellation
parcellation_file = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"

print(f"Session ID: {id}")
print(f"Raw file: {raw_file}")
print(f"MRI file: {mri_file}")

#%%
# Step 1: Preprocessing
# ^^^^^^^^^^^^^^^^^^^^^
#
# We clean the sensor-level MEG data by:
#
# - Resampling to reduce data size.
# - Bandpass filtering to remove slow drifts and high-frequency noise.
# - Notch filtering to remove mains frequency contamination.
# - Detecting and annotating bad segments (outlier time windows).
# - Detecting bad channels (outlier sensors).
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
# Bad segment detection
# *********************
#
# We use the Generalized Extreme Studentized Deviate (G-ESD) algorithm to identify windows with outlier standard deviation. We run this separately for magnetometers and gradiometers, in both standard and ``diff`` modes (the latter catches segments where the signal changes abruptly within a window).

raw = preproc.detect_bad_segments(raw, picks="mag")
raw = preproc.detect_bad_segments(raw, picks="mag", mode="diff")
raw = preproc.detect_bad_segments(raw, picks="grad")
raw = preproc.detect_bad_segments(raw, picks="grad", mode="diff")

#%%
# Bad channel detection
# *********************
#
# Channels with outlier standard deviations are marked as bad.

raw = preproc.detect_bad_channels(raw, picks="mag")
raw = preproc.detect_bad_channels(raw, picks="grad")

#%%
# **QC:** Save preprocessing QC plots (PSD, sum-of-squares time series, channel standard deviations). Check that bad segments and channels are being correctly identified.

preproc.save_qc_plots(raw, plots_dir / id)

#%%
# Save preprocessed data
# **********************

preproc_file = output_dir / "preprocessed" / f"{id}_preproc-raw.fif"
preproc_file.parent.mkdir(parents=True, exist_ok=True)
raw.save(preproc_file, overwrite=True)
print(f"Saved: {preproc_file}")

#%%
# Step 2: Surface Extraction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We extract three surfaces from the structural MRI using FSL BET:
#
# - **Inner skull (inskull)** — used for the forward model.
# - **Outer skull (outskull)** — boundary between skull and scalp.
# - **Outer skin (outskin)** — scalp surface, used for coregistration.
#
# These surfaces define the geometry needed for coregistration and source reconstruction.
#
# .. note::
#
#     **No structural MRI?** If you don't have a subject-specific MRI, you can skip this step and use the standard MNI152 brain bundled with osl-dynamics. Set ``surfaces_dir = files.mni152_surfaces.directory`` in Step 3 and pass ``allow_mri_scaling=True`` during coregistration.

surfaces_dir = str(output_dir / "anat_surfaces" / f"sub-{subject}")

rhino.extract_surfaces(
    mri_file=str(mri_file),
    outdir=surfaces_dir,
    include_nose=False,
    do_mri2mniaxes_xform=False,
)

#%%
# The plots above overlay each extracted surface (yellow line) on the structural MRI. Check that each surface matches the corresponding anatomical boundary. If they don't, consider using the standard MNI152 brain as a fallback.

#%%
# Step 3: Coregistration
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Coregistration aligns the MEG sensor coordinate system ("head" space) to the MRI coordinate system. We use the Polhemus fiducials (nasion, LPA, RPA) and headshape points recorded during the MEG session, and fit them to the MRI surfaces using the Iterative Closest Point (ICP) algorithm.
#
# First, let's create an ``OSLFilenames`` container to keep track of all the pipeline output files.

fns = OSLFilenames(
    outdir=str(output_dir / "osl"),
    id=id,
    preproc_file=str(preproc_file),
    surfaces_dir=surfaces_dir,
    # If using standard brain: surfaces_dir=files.mni152_surfaces.directory
)
print(fns)

#%%
# Extract fiducials and headshape
# *******************************
#
# The Polhemus digitisation points are stored inside the FIF file. We extract them to text files that RHINO uses for coregistration.

rhino.extract_fiducials_and_headshape_from_fif(fns)

#%%
# Run coregistration
# ******************
#
# Key parameters:
#
# - ``use_nose=False`` — Exclude nose headshape points (the nose is often poorly digitised and can pull the fit off).
# - ``allow_mri_scaling=False`` — Don't scale the MRI to fit the headshape. Set ``True`` if using the MNI152 standard brain, which needs to be scaled to match the subject's head size.

rhino.coregister_head_and_mri(
    fns,
    use_nose=False,
    allow_mri_scaling=False,  # set True if using MNI152 standard brain
    plot_type="html",
)

#%%
# **QC:** Save the coregistration plot. The 3D plot shows the MEG sensors (blue), headshape points (red dots), fiducials, and MRI surfaces. Check that the headshape points sit on the scalp surface and the sensors surround the head correctly.
#
# If the coregistration looks off, you can try:
#
# - Removing stray headshape points with ``rhino.remove_stray_headshape_points``.
# - Manually editing the headshape/fiducial text files in ``fns.coreg_dir``.
# - Setting ``use_nose=True`` if the nose digitisation is good.

import shutil

session_plots_dir = plots_dir / id
session_plots_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(
    Path(fns.coreg_dir) / "coreg.html",
    session_plots_dir / "3_coreg.html",
)

#%%
# Step 4: Forward Model
# ^^^^^^^^^^^^^^^^^^^^^
#
# The forward model (lead field matrix) describes how a dipole at each source location projects onto the MEG sensors. We use a Single Layer (Single Shell) head model based on the inner skull surface and a volumetric dipole grid.
#
# - ``model="Single Layer"`` — Single shell head model (standard for MEG).
# - ``gridstep=8`` — 8 mm dipole grid spacing. Smaller values give finer resolution but are slower.

rhino.forward_model(fns, model="Single Layer", gridstep=gridstep)

#%%
# Step 5: Source Reconstruction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We use an LCMV (Linearly Constrained Minimum Variance) beamformer to project the sensor data into source space. The beamformer computes a spatial filter for each voxel that passes activity from that location while suppressing interference from elsewhere.
#
# Key parameters:
#
# - ``chantypes=["mag", "grad"]`` — Use both magnetometers and gradiometers (Elekta systems). For CTF, use ``["mag"]``.
# - ``rank={"meg": 60}`` — Effective rank of the data after MaxFilter. Elekta data is typically rank 60-64. For non-MaxFiltered data or CTF, you can use ``"info"`` to let MNE estimate the rank.
#
# Compute beamformer weights
# **************************

source_recon.lcmv_beamformer(fns, raw, chantypes=chantypes, rank=rank)

#%%
# Apply beamformer
# ****************
#
# This applies the spatial filters to the sensor data to produce voxel time courses in MNI space. Bad segments are automatically excluded.

voxel_data, voxel_coords = source_recon.apply_lcmv_beamformer(fns, raw)
print(f"Voxel data shape: {voxel_data.shape} (voxels x time)")
print(f"Voxel coords shape: {voxel_coords.shape} (voxels x 3, in MNI mm)")

#%%
# Step 6: Parcellation
# ^^^^^^^^^^^^^^^^^^^^
#
# We reduce the high-dimensional voxel data to a smaller number of parcel time courses using a brain atlas. This makes the data more manageable for downstream analysis.
#
# - ``method="spatial_basis"`` — Weight voxels by their loading on each parcel (from the atlas), rather than simple averaging.
# - ``orthogonalisation="symmetric"`` — Apply symmetric orthogonalisation to reduce spatial leakage between parcels caused by the beamformer.

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
#
# We save the parcel time courses as a FIF file. The ``extra_chans="stim"`` option preserves any stimulus channels from the original recording.

parc_fif = str(output_dir / "osl" / id / "lcmv-parc-raw.fif")
parcellation.save_as_fif(
    parcel_data,
    raw,
    extra_chans="stim",
    filename=parc_fif,
)
print(f"Saved: {parc_fif}")

#%%
# QC: PSD and power maps
# **********************
#
# A good sanity check is to plot the PSD of each parcel and band-limited power maps. We expect to see an alpha peak (~10 Hz) that is strongest in posterior parcels. If this is absent or the spectra look unusual, something may have gone wrong in the pipeline.

parcellation.save_qc_plots(parc_fif, parcellation_file, plots_dir / id)

#%%
# Summary and Next Steps
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We have completed the full MEG preprocessing pipeline:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 5 25 40
#
#    * - Step
#      - What
#      - Output
#    * - 1
#      - Preprocessing
#      - ``derivatives/preprocessed/{id}_preproc-raw.fif``
#    * - 2
#      - Surface extraction
#      - ``derivatives/anat_surfaces/sub-{subject}/``
#    * - 3
#      - Coregistration
#      - ``derivatives/osl/{id}/coreg/``
#    * - 4
#      - Forward model
#      - ``derivatives/osl/{id}/coreg/model-fwd.fif``
#    * - 5
#      - Source reconstruction
#      - ``derivatives/osl/{id}/src/filters-lcmv.h5``
#    * - 6
#      - Parcellation
#      - ``derivatives/osl/{id}/lcmv-parc-raw.fif``
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
# - **Canonical HMM analysis** — See the `Canonical-HMM-Networks <https://github.com/OHBA-analysis/Canonical-HMM-Networks>`_ notebooks for applying a canonical HMM to the parcellated data.
# - **Batch processing** — See the `batch scripts <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/meg_preproc>`_ for processing many sessions in parallel.
# - **osl-dynamics documentation** — See the :doc:`documentation <../documentation>` for further analysis tutorials.
#
# Notes on CTF and OPM data
# *************************
#
# This tutorial uses Elekta MEG data. If you are working with different MEG systems:
#
# - **CTF:** Use ``mne.io.read_raw_ctf()`` to load data. Set ``chantypes=["mag"]`` and adjust ``rank`` (CTF data is not MaxFiltered, so the rank is typically higher). Fiducials may need to be extracted from a ``.pos`` file rather than the data file — pass ``pos_file`` to ``OSLFilenames``.
# - **OPM:** Loading depends on the OPM system. Adjust ``chantypes`` and ``rank`` accordingly.
