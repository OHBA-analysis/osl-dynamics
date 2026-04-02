"""Step 5: Parcellation."""

from pathlib import Path
import mne
import matplotlib
matplotlib.use("Agg")
from osl_dynamics.meeg import parallel, source_recon, parcellation
from osl_dynamics.utils.filenames import OSLFilenames

# ----------------------------------------------------------------------------
input_dir = Path("BIDS")
output_dir = Path("derivatives")
plots_dir = Path("plots")
log_dir = Path("logs/5_parc")

sessions = [
    {"id": "sub-01_task-rest", "subject": "sub-01", "raw_file": "sub-01_task-rest.fif"},
    {"id": "sub-02_task-rest", "subject": "sub-02", "raw_file": "sub-02_task-rest.fif"},
    {"id": "sub-03_task-rest", "subject": "sub-03", "raw_file": "sub-03_task-rest.fif"},
    {"id": "sub-04_task-rest", "subject": "sub-04", "raw_file": "sub-04_task-rest.fif"},
]

parcellation_file = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"
parcellation_method = "spatial_basis"
orthogonalisation = "symmetric"
use_mni152 = False
# ----------------------------------------------------------------------------


def process_session(session, logger):
    """Parcellate a single session."""

    preproc_file = output_dir / "preprocessed" / f"{session['id']}_preproc-raw.fif"

    if use_mni152:
        from osl_dynamics import files
        surfaces_dir = files.mni152_surfaces.directory
    else:
        surfaces_dir = str(output_dir / "anat_surfaces" / session["subject"])

    fns = OSLFilenames(
        outdir=str(output_dir / "osl"),
        id=session["id"],
        preproc_file=str(preproc_file),
        surfaces_dir=surfaces_dir,
    )

    logger.log("Applying LCMV beamformer...")
    voxel_data, voxel_coords = source_recon.apply_lcmv_beamformer(fns)

    logger.log("Parcellating...")
    parcel_data = parcellation.parcellate(
        fns,
        voxel_data,
        voxel_coords,
        method=parcellation_method,
        orthogonalisation=orthogonalisation,
        parcellation_file=parcellation_file,
    )

    logger.log("Saving parcellated data...")
    raw = mne.io.read_raw_fif(str(preproc_file), preload=True)
    parc_fif = str(output_dir / "osl" / session["id"] / "lcmv-parc-raw.fif")
    parcellation.save_as_fif(
        parcel_data,
        raw,
        extra_chans="stim",
        filename=parc_fif,
    )

    logger.log("Saving QC plots...")
    parcellation.save_qc_plots(parc_fif, parcellation_file)

    logger.log("Done.")


if __name__ == "__main__":
    parallel.run(
        process_session,
        items=sessions,
        output_dir=output_dir,
        log_dir=log_dir,
        plots_dir=plots_dir,
        n_workers=4,
    )
