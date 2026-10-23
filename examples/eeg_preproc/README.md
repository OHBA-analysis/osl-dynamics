# EEG Preprocessing

This directory contains Python scripts for preparing EEG data for source reconstruction.

## Scripts

| Script | Description |
|--------|-------------|
| `standard_montage_coreg.py` | Fit a standard montage to the MNI152 standard brain by scaling the electrode positions (with `osl_dynamics.meeg.preproc.scale_eeg_positions`). Useful when you don't have digitised electrode positions or a structural MRI. |

## Usage

```bash
conda activate osld
python standard_montage_coreg.py
```

The coregistration plots are saved to `standard_montage_coreg/<id>/coreg/coreg.png`. Edit `scale` at the top of the script and re-run until the electrodes fit the scalp and the montage fiducials match the MRI fiducials.
