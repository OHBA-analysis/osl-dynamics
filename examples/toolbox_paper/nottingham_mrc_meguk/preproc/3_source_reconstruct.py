"""Nottingham MRC MEGUK: Source reconstruction.

"""

from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import Client

from osl import source_recon, utils

# Directories
RAW_DIR = "data/raw/Nottingham"
PREPROC_DIR = "data/preproc"
SRC_DIR = "data/src"

# Files
SMRI_FILE = "data/smri/{0}_T1w.nii.gz"
PREPROC_FILE = PREPROC_DIR + "/{0}_task-resteyesopen_meg/{0}_task-resteyesopen_meg_preproc_raw.fif"
POS_FILE = RAW_DIR + "/{0}/meg/{0}_headshape.pos"

def save_polhemus_from_pos(src_dir, subject, preproc_file, smri_file, epoch_file):
    """Saves fiducials/headshape from a pos file."""

    # Load pos file. Note, it contains values in cm in polhemus space
    pos_file = POS_FILE.format(subject)
    utils.logger.log_or_print(f"Saving polhemus from {pos_file}")

    num_headshape_pnts = int(pd.read_csv(pos_file, header=None).to_numpy()[0])
    data = pd.read_csv(pos_file, header=None, skiprows=[0], delim_whitespace=True)

    # RHINO is going to work with distances in mm so we convert from cm to mm
    data.iloc[:, 1:4] = data.iloc[:, 1:4] * 10

    # Polhemus fiducial and headshape points in polhemus space in mm
    polhemus_nasion = data[data.iloc[:, 0].str.match("nasion")].iloc[0, 1:4].to_numpy().astype("float64").T
    polhemus_rpa = data[data.iloc[:, 0].str.match("right")].iloc[0, 1:4].to_numpy().astype("float64").T
    polhemus_lpa = data[data.iloc[:, 0].str.match("left")].iloc[0, 1:4].to_numpy().astype("float64").T
    polhemus_headshape = data[0:num_headshape_pnts].iloc[:, 1:4].to_numpy().astype("float64").T

    # Save
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)
    np.savetxt(filenames["polhemus_nasion_file"], polhemus_nasion)
    np.savetxt(filenames["polhemus_rpa_file"], polhemus_rpa)
    np.savetxt(filenames["polhemus_lpa_file"], polhemus_lpa)
    np.savetxt(filenames["polhemus_headshape_file"], polhemus_headshape)

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup parallel workers
    client = Client(n_workers=16, threads_per_worker=1)

    # Settings
    config = """
        source_recon:
        - save_polhemus_from_pos: {}
        - compute_surfaces_coregister_and_forward_model:
            include_nose: true
            use_nose: true
            use_headshape: true
            model: Single Layer
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: mag
            rank: {mag: 120}
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
    """

    # Get input files
    subjects = []
    smri_files = []
    preproc_files = []
    for path in sorted(glob(PREPROC_FILE.replace("{0}", "*"))):
        subject = Path(path).stem.split("_")[0]
        subjects.append(subject)
        preproc_files.append(PREPROC_FILE.format(subject))
        smri_files.append(SMRI_FILE.format(subject))

    # Run batch source reconstruction
    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[save_polhemus_from_pos],
        dask_client=True,
    )
