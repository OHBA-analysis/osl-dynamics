"""Step 1: Preprocessing."""

from pathlib import Path
import mne
import matplotlib
matplotlib.use("Agg")
from osl_dynamics.meeg import parallel, preproc

# ----------------------------------------------------------------------------
input_dir = Path("BIDS")
output_dir = Path("derivatives")
qc_dir = Path("qc")
log_dir = Path("logs/1_preproc")

sessions = [
    {"id": "sub-01_task-rest", "subject": "sub-01", "raw_file": "sub-01_task-rest.fif"},
    {"id": "sub-02_task-rest", "subject": "sub-02", "raw_file": "sub-02_task-rest.fif"},
    {"id": "sub-03_task-rest", "subject": "sub-03", "raw_file": "sub-03_task-rest.fif"},
    {"id": "sub-04_task-rest", "subject": "sub-04", "raw_file": "sub-04_task-rest.fif"},
]
# ----------------------------------------------------------------------------


def process_session(session):
    """Preprocess a single session."""

    raw_file = input_dir / session["subject"] / "meg" / session["raw_file"]
    raw = mne.io.read_raw_fif(raw_file, preload=True)

    raw = raw.crop(tmax=60)

    raw = raw.notch_filter([50, 100])
    raw = raw.filter(
        l_freq=1,
        h_freq=45,
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
    )
    raw = raw.resample(sfreq=250)

    raw = preproc.detect_bad_segments(raw, picks="mag")
    raw = preproc.detect_bad_segments(raw, picks="mag", mode="diff")
    raw = preproc.detect_bad_segments(raw, picks="grad")
    raw = preproc.detect_bad_segments(raw, picks="grad", mode="diff")

    raw = preproc.detect_bad_channels(raw, picks="mag")
    raw = preproc.detect_bad_channels(raw, picks="grad")

    raw, ica, ic_labels = preproc.ica_label(raw, picks="meg")

    preproc.save_qc_plots(
        qc_dir,
        session["id"],
        raw,
        ica=ica,
        ic_labels=ic_labels,
    )

    preproc_out_dir = output_dir / "preprocessed"
    preproc_out_dir.mkdir(parents=True, exist_ok=True)
    outfile = preproc_out_dir / f"{session['id']}_preproc-raw.fif"
    raw.save(outfile, overwrite=True)


if __name__ == "__main__":
    parallel.run(
        process_session,
        items=sessions,
        output_dir=output_dir,
        log_dir=log_dir,
        qc_dir=qc_dir,
        n_workers=4,
    )
