"""Step 1: Preprocessing."""


from pathlib import Path

import mne
import matplotlib
matplotlib.use("Agg")

from osl_dynamics.meeg import parallel, preproc

# ----------------------------------------------------------------------------
input_dir = Path("BIDS")
output_dir = Path("BIDS/derivatives")
plots_dir = Path("plots")
log_dir = Path("logs/1_preproc")
sessions = {
    "sub-01_task-rest": {"subject": "sub-01", "file": "sub-01_task-rest.fif"},
    "sub-02_task-rest": {"subject": "sub-02", "file": "sub-02_task-rest.fif"},
    "sub-03_task-rest": {"subject": "sub-03", "file": "sub-03_task-rest.fif"},
    "sub-04_task-rest": {"subject": "sub-04", "file": "sub-04_task-rest.fif"},
}
n_workers = 4
# ----------------------------------------------------------------------------


def process_session(id, info, logger, **kwargs):
    """Preprocess a single session."""
    logger.log("Loading raw data...")
    raw_file = input_dir / info["subject"] / "meg" / info["file"]
    raw = mne.io.read_raw_fif(raw_file, preload=True)

    raw = raw.crop(tmax=60)

    logger.log("Filtering and downsampling...")
    raw = raw.resample(sfreq=250)
    raw = raw.filter(
        l_freq=1, h_freq=45,
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
    )
    raw = raw.notch_filter([50, 100])

    logger.log("Detecting bad segments...")
    raw = preproc.detect_bad_segments(raw, picks="mag")
    raw = preproc.detect_bad_segments(raw, picks="mag", mode="diff")
    raw = preproc.detect_bad_segments(raw, picks="grad")
    raw = preproc.detect_bad_segments(raw, picks="grad", mode="diff")

    logger.log("Detecting bad channels...")
    raw = preproc.detect_bad_channels(raw, picks="mag")
    raw = preproc.detect_bad_channels(raw, picks="grad")

    logger.log("Running ICA artefact rejection...")
    raw, ica, ic_labels = preproc.ica_label(raw, picks="meg")

    logger.log("Saving QC plots...")
    preproc.save_qc_plots(raw, plots_dir / id, ica=ica, ic_labels=ic_labels)

    logger.log("Saving preprocessed data...")
    preproc_out_dir = output_dir / "preprocessed"
    preproc_out_dir.mkdir(parents=True, exist_ok=True)
    outfile = preproc_out_dir / f"{id}_preproc-raw.fif"
    raw.save(outfile, overwrite=True)

    logger.log("Done.")


if __name__ == "__main__":
    parallel.run(
        process_session,
        items=sessions,
        n_workers=n_workers,
        log_dir=log_dir,
        plots_dir=plots_dir,
        output_dir=output_dir,
        step_name="Step 1",
    )
