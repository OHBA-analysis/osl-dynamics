"""Wakeman-Henson: Source Reconstruction.

"""

from dask.distributed import Client

from osl import source_recon, utils

# Directories
raw_dir = "data/ds117"
preproc_dir = "data/preproc"
src_dir = "data/src"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup parallel workers
    client = Client(n_workers=20, threads_per_worker=1)

    config = """
        source_recon:
        - extract_fiducials_from_fif: {}
        - compute_surfaces_coregister_and_forward_model:
            include_nose: false
            use_nose: false
            use_headshape: false
            model: Single Layer
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: [mag, grad]
            rank: {meg: 60}
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
    """

    # Get input files for source reconstruction
    subjects = []
    preproc_files = []
    smri_files = []
    for sub in range(1, 20):
        for run in range(1, 7):
            subjects.append(f"sub{sub:03d}_run{run:02d}")
            preproc_files.append(
                f"{preproc_dir}/sub{sub:03d}/run_{run:02d}_sss/run_{run:02d}_sss_preproc_raw.fif"
            )
            smri_files.append(f"{raw_dir}/sub{sub:03d}/anatomy/highres001.nii.gz")

    # Run batch source reconstruction
    source_recon.run_src_batch(
        config,
        src_dir=src_dir,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        dask_client=True,
    )
