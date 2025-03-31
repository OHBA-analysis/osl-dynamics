"""Wakeman-Henson: Preprocessing.

"""

from dask.distributed import Client

from osl_ephys import preprocessing, utils

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup parallel workers
    client = Client(n_workers=6, threads_per_worker=1)

    config = """
        preproc:
        - set_channel_types: {EEG061: eog, EEG062: eog, EEG063: ecg}
        - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag}
        - bad_segments: {segment_len: 500, picks: grad}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff}
        - bad_segments: {segment_len: 500, picks: grad, mode: diff}
        - ica_raw: {picks: meg, n_components: 40}
        - ica_autoreject: {picks: meg, ecgmethod: correlation, eogthreshold: auto}
        - interpolate_bads: {}
    """

    # Directories
    rawdir = "data/ds117"
    outdir = "data/preproc"

    for sub in range(1, 20):
        # Get input files for preprocessing
        inputs = []
        subjects = []
        for run in range(1, 7):
            inputs.append(f"{rawdir}/sub{sub:03d}/MEG/run_{run:02d}_sss.fif")
            subjects.append(f"{sub-{sub:02d}}_run-{run:02d}")

        # Run batch preprocessing
        preprocessing.run_proc_batch(
            config,
            inputs,
            outdir=outdir,
            subjects=subjects,
            overwrite=True,
            dask_client=True,
        )
