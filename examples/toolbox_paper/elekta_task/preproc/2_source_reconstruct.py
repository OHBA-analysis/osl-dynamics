
"""Wakeman-Henson: Source Reconstruction.

"""

import numpy as np
from dask.distributed import Client

from osl_ephys import source_recon, utils

def fix_headshape_points(outdir, subject):
    filenames = source_recon.rhino.get_coreg_filenames(outdir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Remove headshape points on the nose
    remove = np.logical_and(hs[1] > max(lpa[1], rpa[1]), hs[2] < nas[2])
    hs = hs[:, ~remove]

    # Overwrite headshape file
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup parallel workers
    client = Client(n_workers=16, threads_per_worker=1)

    config = """
        source_recon:
        - extract_fiducials_from_fif: {}
        - fix_headshape_points: {}
        - compute_surfaces:
            include_nose: False
        - coregister:
            use_nose: False
            use_headshape: True
        - forward_model:
            model: Single Layer
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: [mag, grad]
            rank: {meg: 60}
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
    """

    # Directories
    rawdir = "data/ds117"
    outdir = "data/preproc"

    # Get input files for source reconstruction
    subjects = []
    preproc_files = []
    smri_files = []
    for sub in range(1, 20):
        for run in range(1, 7):
            subject = f"sub-{sub:02d}_run-{run:02d}"
            subjects.append(subject)
            preproc_files.append(f"{outdir}/{subject}/{subject}_preproc-raw.fif")
            smri_files.append(f"{rawdir}/sub{sub:03d}/anatomy/highres001.nii.gz")

    # Run batch source reconstruction
    source_recon.run_src_batch(
        config,
        outdir=outdir,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[fix_headshape_points],
        dask_client=True,
    )
