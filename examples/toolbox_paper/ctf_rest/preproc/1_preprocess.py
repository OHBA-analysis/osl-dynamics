"""Nottingham MEGUK: Preprocessing."""

from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import mne

from osl_dynamics.meeg import parallel, preproc

# ----------------------------------------------------------------------------
rawdir = Path("data/raw/Nottingham")
outdir = Path("data/preproc")
log_dir = Path("logs/1_preprocess")
n_workers = 16
# ----------------------------------------------------------------------------


def process_session(session):
    """Preprocess a single session."""

    raw = mne.io.read_raw_ctf(session["raw_file"], preload=True)

    parallel.log("Setting channel types and picking...")
    raw = raw.set_channel_types(
        {"EEG057": "eog", "EEG058": "eog", "EEG059": "ecg"}
    )
    raw = raw.pick(["mag", "eog", "ecg"])

    raw = raw.filter(
        l_freq=1,
        h_freq=125,
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
    )
    raw = raw.notch_filter([50, 100])
    raw = raw.resample(sfreq=250)

    raw = preproc.detect_bad_segments(raw, picks="mag")
    raw = preproc.detect_bad_segments(raw, picks="mag", mode="diff")

    raw = preproc.detect_bad_channels(raw, picks="mag")

    raw, ica, ic_labels = preproc.ica_ecg_eog_correlation(
        raw, picks="mag", n_components=40
    )

    parallel.log("Interpolating bad channels...")
    raw = raw.interpolate_bads()

    out_subject_dir = outdir / session["subject"]
    out_subject_dir.mkdir(parents=True, exist_ok=True)
    raw.save(
        out_subject_dir / f"{session['subject']}_preproc-raw.fif", overwrite=True
    )


if __name__ == "__main__":
    sessions = []
    for directory in sorted(glob(str(rawdir / "sub-*"))):
        subject = Path(directory).name
        raw_file = rawdir / subject / "meg" / f"{subject}_task-resteyesopen_meg.ds"
        if raw_file.exists():
            sessions.append(
                {"id": subject, "subject": subject, "raw_file": str(raw_file)}
            )

    parallel.run(
        process_session,
        items=sessions,
        n_workers=n_workers,
        log_dir=log_dir,
    )
