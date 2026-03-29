"""Step 3: Coregistration."""

from pathlib import Path
from osl_dynamics import files
from osl_dynamics.meeg import parallel, rhino
from osl_dynamics.utils.filenames import OSLFilenames

# ----------------------------------------------------------------------------
input_dir = Path("BIDS")
output_dir = Path("BIDS/derivatives")
plots_dir = Path("plots")
log_dir = Path("logs/3_coreg")

sessions = [
    {"id": "sub-01_task-rest", "subject": "sub-01", "raw_file": "sub-01_task-rest.fif"},
    {"id": "sub-02_task-rest", "subject": "sub-02", "raw_file": "sub-02_task-rest.fif"},
    {"id": "sub-03_task-rest", "subject": "sub-03", "raw_file": "sub-03_task-rest.fif"},
    {"id": "sub-04_task-rest", "subject": "sub-04", "raw_file": "sub-04_task-rest.fif"},
]

use_nose = False
allow_mri_scaling = False  # set True if using MNI152 standard brain
use_mni152 = False  # set True to use standard brain instead of subject MRI
# ----------------------------------------------------------------------------


def process_session(session, logger):
    """Coregister a single session."""

    preproc_file = output_dir / "preprocessed" / f"{session['id']}_preproc-raw.fif"

    if use_mni152:
        surfaces_dir = files.mni152_surfaces.directory
    else:
        surfaces_dir = str(output_dir / "anat_surfaces" / session["subject"])

    fns = OSLFilenames(
        outdir=str(output_dir / "osl"),
        id=session["id"],
        preproc_file=str(preproc_file),
        surfaces_dir=surfaces_dir,
    )

    logger.log("Extracting fiducials and headshape...")
    rhino.extract_fiducials_and_headshape_from_fif(fns)

    logger.log("Coregistering MEG to MRI...")
    rhino.coregister_head_and_mri(
        fns,
        use_nose=use_nose,
        allow_mri_scaling=allow_mri_scaling,
    )

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
