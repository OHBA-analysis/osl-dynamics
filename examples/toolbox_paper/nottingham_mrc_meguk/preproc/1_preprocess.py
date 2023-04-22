"""Nottingham MRC MEGUK: Preprocessing.

"""

from glob import glob
from pathlib import Path
from dask.distributed import Client

from osl import preprocessing, utils

# Directories and files
RAW_DIR = "data/raw/Nottingham"
PREPROC_DIR = "data/preproc"

RAW_FILE = RAW_DIR + "/{0}/meg/{0}_task-resteyesopen_meg.ds"

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

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup parallel processing
    client = Client(n_workers=16, threads_per_worker=1)

    # Setup input files
    inputs = []
    for directory in sorted(glob(RAW_DIR + "/sub-*")):
        subject = Path(directory).name
        raw_file = RAW_FILE.format(subject)
        if Path(raw_file).exists():
            inputs.append(raw_file)

    # Run batch preprocessing
    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=PREPROC_DIR,
        overwrite=True,
        dask_client=True,
    )
