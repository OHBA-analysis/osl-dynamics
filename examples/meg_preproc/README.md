# Parallelised Pipeline for Elekta MEG

This directory contains Python scripts that run the M/EEG processing pipeline on multiple sessions in parallel using `multiprocessing`.

For a step-by-step interactive walkthrough, see the [MEG Preprocessing tutorial](https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/0-1_meg_preprocessing.html) in the docs.

## Input Data

The scripts expect data in BIDS format:
```
BIDS/
├── sub-01/
│   ├── anat/
│   │   └── sub-01_T1w.nii.gz
│   └── meg/
│       └── sub-01_task-rest.fif
├── sub-02/
│   ├── ...
```

Output is written to `BIDS/derivatives/`.

## Scripts

Run the scripts **in order**. Each script processes all sessions in parallel.

| Script | Step | Description |
|--------|------|-------------|
| `1_preproc.py` | Preprocessing | Downsample (250 Hz), bandpass filter (1-45 Hz), notch filter (50/100 Hz), bad segment and bad channel detection |
| `2_surfaces.py` | Surface Extraction | Extract inner skull, outer skull and scalp surfaces from structural MRI using FSL BET |
| `3_coreg.py` | Coregistration | Coregister MEG to MRI using Polhemus headshape points |
| `4_source_recon_and_parc.py` | Forward Model, Source Reconstruction and Parcellation | Compute forward model (8 mm dipole grid), LCMV beamformer, parcellate voxel data, apply symmetric orthogonalisation |

## Usage

These scripts can be copied and run from anywhere on your system. The pipeline modules are imported from the `osl_dynamics.meeg` subpackage.

```bash
conda activate osld

python 1_preproc.py
python 2_surfaces.py
python 3_coreg.py
python 4_source_recon_and_parc.py
```

## Configuration

Each script has a config block at the top. Edit these variables before running:

```python
input_dir = Path("BIDS")
output_dir = Path("BIDS/derivatives")
plots_dir = Path("plots")
log_dir = Path("logs/1_preproc")
sessions = {
    "sub-01_task-rest": {"subject": "sub-01", "file": "sub-01_task-rest.fif"},
    "sub-02_task-rest": {"subject": "sub-02", "file": "sub-02_task-rest.fif"},
}
n_workers = 4
```

- `input_dir` — Path to your BIDS directory containing the raw data.
- `output_dir` — Path to the output directory for derivatives.
- `plots_dir` — Directory for QC plots and the HTML report.
- `log_dir` — Directory for per-session log files.
- `sessions` — Dictionary of sessions to process. Each key is a session ID used for naming output files and logs. Each value contains the `subject` (BIDS subject directory) and `file` (MEG filename).
- `n_workers` — Number of sessions to process in parallel.

Step 2 (`2_surfaces.py`) uses a `subjects` list instead of `sessions` since surface extraction only needs to run once per subject:

```python
subjects = ["sub-01", "sub-02", "sub-03", "sub-04"]
```

Some scripts have additional settings (e.g. `parcellation_file`, `gridstep`). See the config block in each script for details.

## Output Structure

```
BIDS/derivatives/
├── preprocessed/
│   ├── sub-01_task-rest_preproc-raw.fif
│   └── ...
├── anat_surfaces/
│   ├── sub-01/
│   │   ├── inskull.png
│   │   ├── outskin.png
│   │   ├── outskull.png
│   │   └── ...
│   └── ...
└── osl/
    ├── sub-01_task-rest/
    │   ├── bem/
    │   ├── coreg/
    │   │   ├── coreg.png
    │   │   └── model-fwd.fif
    │   ├── src/
    │   │   └── filters-lcmv.h5
    │   ├── lcmv-parc-raw.fif
    │   └── psd_topo.png
    └── ...

plots/
├── sub-01_task-rest/
│   ├── 1_summary.json
│   ├── 1_psd.png
│   ├── 1_sum_square.png
│   ├── 1_sum_square_exclude_bads.png
│   ├── 1_channel_stds.png
│   ├── 3_coreg.png
│   └── 4_psd_topo.png
├── sub-02_task-rest/
│   └── ...
└── report.html
```

## QC Report

A self-contained HTML report (`plots/report.html`) is automatically generated after steps 1, 3 and 4 complete. It contains tabs for each pipeline step with all session QC plots embedded. Open it in a browser to review results.

The report updates incrementally — after step 1 you'll see preprocessing plots, after step 3 coregistration and surfaces appear, etc. Surface extraction (step 2), coregistration (step 3), and parcellation (step 4) plots are automatically copied from the derivatives directory when the report is generated.

## Logging

Full verbose output (from MNE, osl-dynamics, etc.) is saved to per-session log files in the `log_dir` directory, e.g. `logs/1_preproc/sub-01_task-rest.log`.

## Notes

- Steps must be run sequentially (each depends on the output of the previous step), but within each step all sessions are processed in parallel.
- If you do not have a structural MRI for a subject, set `use_mni152 = True` and `allow_mri_scaling = True` in `3_coreg.py` (and `4_source_recon_and_parc.py`) to use the standard MNI152 brain (bundled in osl-dynamics at `osl_dynamics.files.mni152_surfaces`). You can skip `2_surfaces.py` in this case.
- Set `n_workers` based on the number of CPU cores available. For memory-intensive steps (source reconstruction, parcellation), you may need to reduce this.
