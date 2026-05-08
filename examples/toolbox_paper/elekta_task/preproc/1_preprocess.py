"""Wakeman-Henson: Preprocessing."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import mne

from osl_dynamics.meeg import parallel, preproc

# ----------------------------------------------------------------------------
rawdir = Path("data/ds117")
outdir = Path("data/preproc")
log_dir = Path("logs/1_preprocess")
n_workers = 6
n_subjects = 19
n_runs = 6
# ----------------------------------------------------------------------------


def process_session(session, logger):
    """Preprocess a single session."""

    logger.log("Loading raw data...")
    raw = mne.io.read_raw_fif(session["raw_file"], preload=True)

    logger.log("Setting channel types...")
    raw = raw.set_channel_types(
        {"EEG061": "eog", "EEG062": "eog", "EEG063": "ecg"}
    )

    logger.log("Filtering and downsampling...")
    raw = raw.filter(
        l_freq=0.5,
        h_freq=125,
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
    )
    raw = raw.notch_filter([50, 100])
    raw = raw.resample(sfreq=250)

    logger.log("Detecting bad segments...")
    raw = preproc.detect_bad_segments(raw, picks="mag")
    raw = preproc.detect_bad_segments(raw, picks="grad")
    raw = preproc.detect_bad_segments(raw, picks="mag", mode="diff")
    raw = preproc.detect_bad_segments(raw, picks="grad", mode="diff")

    logger.log("Detecting bad channels...")
    raw = preproc.detect_bad_channels(raw, picks="mag")
    raw = preproc.detect_bad_channels(raw, picks="grad")

    logger.log("Running ICA artefact rejection...")
    raw, ica, ic_labels = preproc.ica_ecg_eog_correlation(
        raw, picks="meg", n_components=40
    )

    logger.log("Interpolating bad channels...")
    raw = raw.interpolate_bads()

    logger.log("Saving preprocessed data...")
    out_session_dir = outdir / session["id"]
    out_session_dir.mkdir(parents=True, exist_ok=True)
    raw.save(
        out_session_dir / f"{session['id']}_preproc-raw.fif", overwrite=True
    )

    logger.log("Done.")


if __name__ == "__main__":
    sessions = []
    for sub in range(1, n_subjects + 1):
        for run in range(1, n_runs + 1):
            session_id = f"sub-{sub:02d}_run-{run:02d}"
            raw_file = rawdir / f"sub{sub:03d}" / "MEG" / f"run_{run:02d}_sss.fif"
            if raw_file.exists():
                sessions.append(
                    {
                        "id": session_id,
                        "subject": f"sub-{sub:02d}",
                        "raw_file": str(raw_file),
                    }
                )

    parallel.run(
        process_session,
        items=sessions,
        n_workers=n_workers,
        log_dir=log_dir,
    )
