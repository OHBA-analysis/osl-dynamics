"""Nottingham MEGUK: Source reconstruction."""

from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import mne

from osl_dynamics.meeg import parallel, parcellation, rhino, source_recon
from osl_dynamics.utils.filenames import OSLFilenames

# ----------------------------------------------------------------------------
rawdir = Path("data/raw/Nottingham")
outdir = Path("data/preproc")
smri_dir = Path("data/smri")
log_dir = Path("logs/3_source_reconstruct")
n_workers = 16

parcellation_file = "atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz"
# ----------------------------------------------------------------------------


def process_session(session, logger):
    """Source reconstruct a single session."""

    subject = session["subject"]
    preproc_file = outdir / subject / f"{subject}_preproc-raw.fif"
    surfaces_dir = outdir / subject / "surfaces"
    pos_file = rawdir / subject / "meg" / f"{subject}_headshape.pos"

    fns = OSLFilenames(
        outdir=str(outdir),
        id=subject,
        preproc_file=str(preproc_file),
        surfaces_dir=str(surfaces_dir),
        pos_file=str(pos_file),
    )

    logger.log("Extracting fiducials and headshape from .pos file...")
    rhino.extract_fiducials_and_headshape_from_pos(fns)

    logger.log("Extracting surfaces...")
    rhino.extract_surfaces(
        mri_file=session["smri_file"],
        outdir=str(surfaces_dir),
        include_nose=True,
    )

    logger.log("Coregistering MEG to MRI...")
    rhino.coregister_head_and_mri(fns, use_nose=True, use_headshape=True)

    logger.log("Computing forward model...")
    rhino.forward_model(fns, model="Single Layer")

    logger.log("Computing LCMV beamformer...")
    source_recon.lcmv_beamformer(
        fns,
        chantypes="mag",
        rank={"mag": 120},
        frequency_range=[1, 45],
    )

    logger.log("Applying LCMV beamformer...")
    voxel_data, voxel_coords = source_recon.apply_lcmv_beamformer(fns)

    logger.log("Parcellating...")
    parcel_data = parcellation.parcellate(
        fns,
        voxel_data,
        voxel_coords,
        method="spatial_basis",
        orthogonalisation="symmetric",
        parcellation_file=parcellation_file,
    )

    logger.log("Saving parcellated data...")
    raw = mne.io.read_raw_fif(str(preproc_file), preload=True)
    parcellation.save_as_fif(
        parcel_data,
        raw,
        filename=str(outdir / subject / "lcmv-parc-raw.fif"),
        extra_chans="stim",
    )

    logger.log("Done.")


if __name__ == "__main__":
    sessions = []
    for path in sorted(glob(str(outdir / "*" / "sub-*_preproc-raw.fif"))):
        subject = Path(path).stem.split("_")[0]
        smri_file = smri_dir / f"{subject}_T1w.nii.gz"
        if smri_file.exists():
            sessions.append(
                {"id": subject, "subject": subject, "smri_file": str(smri_file)}
            )

    parallel.run(
        process_session,
        items=sessions,
        n_workers=n_workers,
        log_dir=log_dir,
    )
