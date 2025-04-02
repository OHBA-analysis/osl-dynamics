"""Nottingham MRC MEGUK: Source reconstruction.

"""

from glob import glob
from pathlib import Path
from dask.distributed import Client

from osl_ephys import source_recon, utils

# Directories
rawdir = "data/raw/Nottingham"
outdir = "data/preproc"

# Files
smri_file = "data/smri/{0}_T1w.nii.gz"
preproc_file = outdir + "/{0}/{0}_preproc-raw.fif"
pos_filepath = rawdir + "/{subject}/meg/{subject}_headshape.pos"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup parallel workers
    client = Client(n_workers=16, threads_per_worker=1)

    # Settings
    config = f"""
        source_recon:
        - extract_polhemus_from_pos:
            filepath: {pos_filepath}
        - compute_surfaces:
            include_nose: true
        - coregister:
            use_nose: true
            use_headshape: true
        - forward_model:
            model: Single Layer
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: mag
            rank: {{mag: 120}}
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
    """

    # Get input files
    subjects = []
    smri_files = []
    preproc_files = []
    for path in sorted(glob(preproc_file.replace("{0}", "*"))):
        subject = Path(path).stem.split("_")[0]
        subjects.append(subject)
        preproc_files.append(preproc_file.format(subject))
        smri_files.append(smri_file.format(subject))

    # Run batch source reconstruction
    source_recon.run_src_batch(
        config,
        outdir=outdir,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        #dask_client=True,
    )
