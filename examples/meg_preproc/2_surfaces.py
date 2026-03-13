"""Step 2: Surface Extraction."""

from pathlib import Path

from osl_dynamics.meeg import parallel, rhino

# ----------------------------------------------------------------------------
input_dir = Path("BIDS")
output_dir = Path("BIDS/derivatives")
log_dir = Path("logs/2_surfaces")
subjects = ["sub-01", "sub-02", "sub-03", "sub-04"]
include_nose = False
n_workers = 4
# ----------------------------------------------------------------------------


def process_subject(subject, _, logger, **kwargs):
    """Extract surfaces for a single subject."""
    logger.log("Extracting surfaces...")

    mri_file = input_dir / subject / "anat" / f"{subject}_T1w.nii.gz"
    outdir = output_dir / "anat_surfaces" / subject

    rhino.extract_surfaces(
        mri_file=str(mri_file),
        outdir=str(outdir),
        include_nose=include_nose,
    )

    logger.log("Done.")


if __name__ == "__main__":
    parallel.run(
        process_subject,
        items=subjects,
        n_workers=n_workers,
        log_dir=log_dir,
        step_name="Step 2",
    )
