"""Nottingham MRC MEGUK: Preprocessing.

"""

from glob import glob
from pathlib import Path
from dask.distributed import Client

from osl_ephys import preprocessing, utils

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Settings
    config = """
        preproc:
        - set_channel_types: {EEG057: eog, EEG058: eog, EEG059: ecg}
        - pick: {picks: [mag, eog, ecg]}
        - filter: {l_freq: 1, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff}
        - bad_channels: {picks: mag}
        - ica_raw: {picks: mag, n_components: 40}
        - ica_autoreject: {picks: mag, ecgmethod: correlation, eogthreshold: auto}
        - interpolate_bads: {}
    """

    # Setup parallel processing
    client = Client(n_workers=16, threads_per_worker=1)

    # Directories and files
    rawdir = "data/raw/Nottingham"
    outdir = "data/preproc"

    raw_file = rawdir + "/{0}/meg/{0}_task-resteyesopen_meg.ds"

    # Setup input files
    inputs = []
    subjects = []
    for directory in sorted(glob(rawdir + "/sub-*")):
        subject = Path(directory).name
        raw_file = raw_file.format(subject)
        if Path(raw_file).exists():
            inputs.append(raw_file)
            subjects.append(subject)

    # Run batch preprocessing
    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=outdir,
        subjects=subjects,
        overwrite=True,
        dask_client=True,
    )
