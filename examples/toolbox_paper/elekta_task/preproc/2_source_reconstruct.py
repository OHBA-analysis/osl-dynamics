"""Wakeman-Henson: Source Reconstruction."""

from glob import glob
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import mne

from osl_dynamics.meeg import parallel, parcellation, rhino, source_recon
from osl_dynamics.utils.filenames import OSLFilenames

# ----------------------------------------------------------------------------
rawdir = Path("data/ds117")
outdir = Path("data/preproc")
log_dir = Path("logs/2_source_reconstruct")
n_workers = 16

parcellation_file = "atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz"
# ----------------------------------------------------------------------------


def fix_headshape_points(fns):
    """Remove headshape points on the nose."""
    hs = np.loadtxt(fns.coreg.head_headshape_file)
    nas = np.loadtxt(fns.coreg.head_nasion_file)
    lpa = np.loadtxt(fns.coreg.head_lpa_file)
    rpa = np.loadtxt(fns.coreg.head_rpa_file)

    remove = np.logical_and(hs[1] > max(lpa[1], rpa[1]), hs[2] < nas[2])
    hs = hs[:, ~remove]

    print(f"overwriting {fns.coreg.head_headshape_file}")
    np.savetxt(fns.coreg.head_headshape_file, hs)


def process_session(session, logger):
    """Source reconstruct a single session."""

    session_id = session["id"]
    preproc_file = outdir / session_id / f"{session_id}_preproc-raw.fif"
    surfaces_dir = outdir / session["subject"] / "surfaces"

    fns = OSLFilenames(
        outdir=str(outdir),
        id=session_id,
        preproc_file=str(preproc_file),
        surfaces_dir=str(surfaces_dir),
    )

    logger.log("Extracting fiducials and headshape from fif...")
    rhino.extract_fiducials_and_headshape_from_fif(fns)

    logger.log("Removing headshape points on the nose...")
    fix_headshape_points(fns)

    logger.log("Extracting surfaces...")
    rhino.extract_surfaces(
        mri_file=session["smri_file"],
        outdir=str(surfaces_dir),
        include_nose=False,
    )

    logger.log("Coregistering MEG to MRI...")
    rhino.coregister_head_and_mri(fns, use_nose=False, use_headshape=True)

    logger.log("Computing forward model...")
    rhino.forward_model(fns, model="Single Layer")

    logger.log("Computing LCMV beamformer...")
    source_recon.lcmv_beamformer(
        fns,
        chantypes=["mag", "grad"],
        rank={"meg": 60},
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
        filename=str(outdir / session_id / "lcmv-parc-raw.fif"),
        extra_chans="stim",
    )

    logger.log("Done.")


if __name__ == "__main__":
    sessions = []
    for path in sorted(glob(str(outdir / "*" / "sub-*_run-*_preproc-raw.fif"))):
        session_id = Path(path).stem.split("_preproc")[0]
        subject = session_id.split("_")[0]
        sub_num = int(subject.split("-")[1])
        smri_file = rawdir / f"sub{sub_num:03d}" / "anatomy" / "highres001.nii.gz"
        sessions.append(
            {
                "id": session_id,
                "subject": subject,
                "smri_file": str(smri_file),
            }
        )

    parallel.run(
        process_session,
        items=sessions,
        n_workers=n_workers,
        log_dir=log_dir,
    )
