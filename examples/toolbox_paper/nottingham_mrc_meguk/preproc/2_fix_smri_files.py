"""Nottingham MRC MEGUK: Fix structural MRI files.

The sform header in the publicly available MRC MEGUK data is incompatible with OSL,
we create a copy and fix the MRI files in this script.
"""

from os import makedirs
from glob import glob
from pathlib import Path
from shutil import copyfile

import numpy as np
import nibabel as nib

from osl import source_recon

# Directories and files
PREPROC_DIR = "data/preproc"
FIXED_SMRI_DIR = "data/smri"

SMRI_FILE = "data/raw/Nottingham/{0}/anat/{0}_T1w.nii.gz"

# Look up which subjects we preprocessed to see what SMRI files we need to fix
smri_files = []
for path in sorted(glob(PREPROC_DIR + "/*/sub-*_preproc_raw.fif")):
    subject = Path(path).stem.split("_")[0]
    smri_file = SMRI_FILE.format(subject)
    if Path(smri_file).exists():
        smri_files.append(smri_file)

# Make output directory if it doesn't exist
makedirs(FIXED_SMRI_DIR, exist_ok=True)

for input_smri_file in smri_files:
    output_smri_file = FIXED_SMRI_DIR + "/" + Path(input_smri_file).name
    print("Saving output to:", output_smri_file)

    # Copy the original SMRI file to the output directory
    copyfile(input_smri_file, output_smri_file)

    # Load the output SMRI file
    smri = nib.load(output_smri_file)

    # Get the original sform header
    sform = smri.header.get_sform()
    sform_std = np.copy(sform)

    # Fix the sform header
    sform_std[0, 0:4] = [-1, 0, 0, 128]
    sform_std[1, 0:4] = [0, 1, 0, -128]
    sform_std[2, 0:4] = [0, 0, 1, -90]
    source_recon.rhino.utils.system_call(
        "fslorient -setsform {} {}".format(
            " ".join(map(str, sform_std.flatten())), output_smri_file
        )
    )
