"""Maxfiltering.

This module wraps the Elekta Maxfilter command-line tool for applying
Signal Space Separation (SSS) and temporal SSS (tSSS) to MEG data.

Note
----
This module requires a licensed installation of Elekta/MEGIN MaxFilter.
MaxFilter is proprietary software and is not included with osl-dynamics.

**OHBA users:** The ``scanner`` argument sets the correct calibration
files for OHBA scanners (``"VectorView"``, ``"VectorView2"`` or
``"Neo"``).  To check which scanner was used to record a FIF file, run
the following on an OHBA machine with a MaxFilter license::

    from subprocess import run, PIPE

    cmd = f"/neuro/bin/util/show_fiff -v -t 100:206 {path_to_fif}"
    result = run(cmd.split(), stdout=PIPE, stderr=PIPE)
    for line in result.stdout.decode().splitlines():
        print(line)

The output will show the recording date and scanner description.

Examples
--------
Standard multistage Maxfiltering with the newest OHBA scanner::

    run_maxfilter(
        files="sub-*_task-rest_meg.fif",
        outdir="derivatives/maxfiltered",
        scanner="Neo",
        mode="multistage",
        tsss=True,
        headpos=True,
        movecomp=True,
    )

Cambridge CBU 3-stage pipeline with a single file::

    run_maxfilter(
        files="sub-01_task-rest_meg.fif",
        outdir="derivatives/maxfiltered",
        mode="cbu",
    )
"""

from __future__ import annotations

import glob
import math
import os
import shutil
import subprocess
import tempfile

import mne
import numpy as np

# Keys copied from the parent options dict into each pipeline stage.
_COMMON_KEYS = ["maxpath", "scanner", "ctc", "cal", "dryrun", "overwrite", "outdir"]
_STAGE1_KEYS = _COMMON_KEYS + [
    "inorder",
    "outorder",
    "hpie",
    "hpig",
    "origin",
    "frame",
]
_STAGE2_KEYS = _COMMON_KEYS + [
    "tsss",
    "st",
    "corr",
    "inorder",
    "outorder",
    "hpig",
    "hpie",
    "movecomp",
    "movecompinter",
    "headpos",
    "origin",
    "frame",
    "skip",
]
_STAGE3_KEYS = _COMMON_KEYS + ["trans"]

# OHBA-specific calibration file paths.
_SCANNER_FILES = {
    "VectorView": {},
    "VectorView2": {
        "cal": "/net/aton/meg_pool/neuromag/databases/sss/sss_cal_3026_171220.dat",
        "ctc": "/net/aton/meg_pool/neuromag/databases/ctc/ct_sparse.fif",
    },
    "Neo": {
        "cal": "/vols/MEG/TriuxNeo/system/sss/sss_cal.dat",
        "ctc": "/vols/MEG/TriuxNeo/system/ctc/ct_sparse.fif",
    },
}

# Simple flag table: (options key, maxfilter flag, value-type).
# "bool" flags are appended when truthy; "value" flags are appended with
# their value when not None.
_FLAG_TABLE = [
    ("movecomp", "-movecomp", "bool"),
    ("hpie", "-hpie", "value"),
    ("hpig", "-hpig", "value"),
    ("hpisubt", "-hpisubt", "value"),
    ("linefreq", "-linefreq", "value"),
    ("badlimit", "-badlimit", "value"),
    ("bads", "-bad", "value"),
    ("force", "-force", "bool"),
    ("inorder", "-in", "value"),
    ("outorder", "-out", "value"),
    ("frame", "-frame", "value"),
    ("trans", "-trans", "value"),
    ("skip", "-skip", "value"),
]


def _pick_keys(source: dict, keys: list[str], overrides: dict | None = None) -> dict:
    """Build a stage-specific options dict from *source*.

    Parameters
    ----------
    source : dict
        The full options dict built by :func:`run_maxfilter`.
    keys : list of str
        Keys to copy from *source* (if present).
    overrides : dict, optional
        Extra key/value pairs that take precedence over *source*.

    Returns
    -------
    stage_options : dict
        Subset of *source* with *overrides* applied.
    """
    stage_options = {k: source[k] for k in keys if k in source}
    if overrides:
        stage_options.update(overrides)
    return stage_options


def _build_command(options: dict, input_file: str, output_file: str) -> str:
    """Assemble the full Maxfilter command string.

    Parameters
    ----------
    options : dict
        Maxfilter options.  Required keys: ``"maxpath"``.
        Optional keys correspond to the entries in ``_FLAG_TABLE``
        plus ``"headpos"``, ``"movecompinter"``, ``"autobad"``,
        ``"autobad_dur"``, ``"tsss"``/``"st"``/``"corr"``,
        ``"origin"``, ``"scanner"``, ``"ctc"``, ``"cal"``.
    input_file : str
        Input FIF path.
    output_file : str
        Output FIF path.

    Returns
    -------
    command : str
        Complete shell command string.
    """
    command = f"{options['maxpath']} -f {input_file} -o {output_file}"

    # Head position logging
    if options.get("headpos"):
        headpos_file = output_file.replace(".fif", "_headpos.log")
        command += f" -hp {headpos_file}"

    # Movement compensation with intermittent HPI
    if options.get("movecompinter"):
        command += " -movecomp inter"

    # Simple flags from the table
    for key, flag, kind in _FLAG_TABLE:
        val = options.get(key)
        if val is None:
            continue
        if kind == "bool" and val:
            command += f" {flag}"
        elif kind == "value":
            command += f" {flag} {val}"

    # Autobad: on/off toggle or duration
    if options.get("autobad_dur") is not None:
        command += f" -autobad {options['autobad_dur']}"
    elif options.get("autobad"):
        command += " -autobad on"
    else:
        command += " -autobad off"

    # tSSS
    if options.get("tsss"):
        command += f" -st {options['st']} -corr {options['corr']}"

    # Origin
    if options.get("origin") is not None:
        command += " -origin {0} {1} {2}".format(*options["origin"])

    # Scanner preset or explicit ctc/cal
    scanner = options.get("scanner")
    if scanner is not None:
        scanner_paths = _SCANNER_FILES.get(scanner, {})
        if "cal" in scanner_paths:
            command += f" -cal {scanner_paths['cal']}"
        if "ctc" in scanner_paths:
            command += f" -ctc {scanner_paths['ctc']}"
    else:
        if options.get("ctc"):
            command += f" -ctc {options['ctc']}"
        if options.get("cal"):
            command += f" -cal {options['cal']}"

    return command


_BIDS_DATA_SUFFIXES = ("_meg", "_eeg", "_ieeg", "_nirs")


def _output_name(input_file: str, outdir: str, *labels: str) -> str:
    """Build a BIDS-MEG output filename using the ``proc-`` entity.

    Multiple processing labels are joined with ``+`` and inserted into
    the ``proc-`` entity before the data-type suffix (``_meg``, ``_eeg``,
    ``_ieeg`` or ``_nirs``).

    Examples
    --------
    >>> _output_name("sub-01_task-rest_meg.fif", "out", "tsss")
    'out/sub-01_task-rest_proc-tsss_meg.fif'

    >>> _output_name("sub-01_task-rest_meg.fif", "out", "tsss", "trans")
    'out/sub-01_task-rest_proc-tsss+trans_meg.fif'

    For non-BIDS inputs (no recognised data-type suffix), the ``proc-``
    entity is appended at the end of the basename instead.
    """
    if not labels:
        raise ValueError("_output_name requires at least one label")
    basename = os.path.splitext(os.path.basename(input_file))[0]
    proc = "proc-" + "+".join(labels)
    for suffix in _BIDS_DATA_SUFFIXES:
        if basename.endswith(suffix):
            stem = basename[: -len(suffix)]
            return os.path.join(outdir, f"{stem}_{proc}{suffix}.fif")
    return os.path.join(outdir, f"{basename}_{proc}.fif")


def _parse_bad_channels(log_file: str) -> str | None:
    """Extract static bad channels from a Maxfilter log file.

    Parameters
    ----------
    log_file : str
        Path to a Maxfilter log file.

    Returns
    -------
    bads : str or None
        Space-separated channel numbers, or ``None`` if not found.
    """
    with open(log_file, "r") as f:
        for line in f:
            if line.startswith("Static bad channels"):
                parts = line.split(": ")[1].split()
                return " ".join(p.strip() for p in parts)
    return None


def _quick_load_dig(fname: str) -> list:
    """Extract digitization points from a FIF file.

    Uses low-level MNE readers, useful when the full raw file cannot
    be loaded.

    Parameters
    ----------
    fname : str
        Path to a FIF file.

    Returns
    -------
    dig : list
        List of digitization points.
    """
    from mne.io.constants import FIFF

    fid, tree, _ = mne.io.open.fiff_open(fname, preload=False)
    meas = mne.io.tree.dir_tree_find(tree, FIFF.FIFFB_MEAS)
    meas_info = mne.io.tree.dir_tree_find(meas, FIFF.FIFFB_MEAS_INFO)
    dig = mne.io._digitization._read_dig_fif(fid, meas_info)
    return dig


def _fit_cbu_origin(
    input_file: str,
    output_base: str | None = None,
    remove_nose: bool = True,
) -> np.ndarray:
    """Fit sphere origin from headshape points.

    This is used by the CBU pipeline to fit a sphere to the headshape
    points, optionally removing nose points first.

    Parameters
    ----------
    input_file : str
        Path to input FIF file.
    output_base : str, optional
        Format-string path for saving intermediate files.  Must contain
        a ``{0}`` placeholder that is replaced with the file suffix,
        e.g. ``"outdir/sub-01_{0}"``.
    remove_nose : bool, optional
        Whether to remove nose points before fitting.

    Returns
    -------
    origin : np.ndarray
        Fitted origin coordinates [x, y, z] in mm.
    """
    try:
        raw = mne.io.read_raw_fif(input_file)
        dig = raw.info["dig"]
    except ValueError:
        dig = _quick_load_dig(input_file)

    # Extract headshape points
    headshape = []
    for point in dig:
        if point["kind"]._name.find("EXTRA") > 0:
            headshape.append(point["r"])
    headshape = np.vstack(headshape)

    if remove_nose:
        keeps = np.where(
            np.logical_and(headshape[:, 2] < 0, headshape[:, 1] > 0)
            == False  # noqa: E712
        )[0]
        headshape = headshape[keeps, :]

    # Save headshape points and fit origin
    if output_base is None:
        headshape_path = tempfile.NamedTemporaryFile(
            prefix="CBU_MaxfilterOrigin_Headshapes"
        ).name
        fit_path = tempfile.NamedTemporaryFile(prefix="CBU_MaxfilterOrigin_Fit").name
    else:
        headshape_path = output_base.format("headshape.txt")
        fit_path = output_base.format("headorigin_fit.txt")

    np.savetxt(headshape_path, headshape)

    subprocess.run(
        f"/neuro/bin/util/fit_sphere_to_points {headshape_path} > {fit_path}",
        shell=True,
        check=True,
    )

    origin = np.loadtxt(fit_path)[:3] * 1000
    print(f"fitted origin is {origin}")

    return origin


def _run_single(
    input_file: str,
    output_file: str,
    options: dict,
) -> tuple[str, str]:
    """Run Maxfilter on a single file.

    Parameters
    ----------
    input_file : str
        Path to input FIF file.
    output_file : str
        Path to output FIF file.
    options : dict
        Maxfilter options.  Must contain ``"maxpath"`` and ``"dryrun"``.
        See :func:`_build_command` for the full set of recognised keys.

    Returns
    -------
    output_file : str
        Path to output FIF file.
    log_file : str
        Path to the stdout log.
    """
    command = _build_command(options, input_file, output_file)

    log_file = output_file.replace(".fif", ".log")
    error_log = output_file.replace(".fif", "_err.log")

    # Capture stdout and stderr into separate files
    command += f" -v > >(tee -a {log_file}) 2> >(tee -a {error_log} >&2)"

    print(command)
    print()

    if not options["dryrun"]:
        subprocess.run(command, shell=True, executable="/bin/bash", check=True)

    return output_file, log_file


def _run_multistage(input_file: str, outdir: str, options: dict) -> None:
    """Run Maxfilter in three sequential stages.

    1. **Bad channel detection** — run with ``-autobad on`` to identify
       bad channels automatically.
    2. **SSS/tSSS** — apply Signal Space Separation with the detected
       bad channels marked.
    3. **Head translation** (optional) — transform data to the head
       position in a reference file.  Only runs when ``options["trans"]``
       is set.

    Parameters
    ----------
    input_file : str
        Path to the *original* input FIF file.
    outdir : str
        Output directory.
    options : dict
        Full Maxfilter options from :func:`run_maxfilter`.
    """
    # Stage 1 - Find Bad Channels (skipped if bads are pre-provided,
    # e.g. by chain expansion which shares one autobad pass across pieces)
    pre_bads = options.get("bads")
    if pre_bads is None:
        output_file = _output_name(input_file, outdir, "autobad")
        if os.path.exists(output_file):
            os.remove(output_file)

        stage1_options = _pick_keys(options, _STAGE1_KEYS, overrides={"autobad": True})
        output_file, log_file = _run_single(input_file, output_file, stage1_options)
        bads = _parse_bad_channels(log_file) if not options["dryrun"] else None
    else:
        bads = pre_bads

    # Stage 2 - Signal Space Separation
    sss_label = "tsss" if options.get("tsss") else "sss"
    output_file = _output_name(input_file, outdir, sss_label)
    if os.path.exists(output_file):
        os.remove(output_file)

    stage2_options = _pick_keys(
        options, _STAGE2_KEYS, overrides={"autobad": None, "bads": bads}
    )
    stage2_output = output_file
    output_file, log_file = _run_single(input_file, output_file, stage2_options)

    # Stage 3 - Translate to reference file
    if options.get("trans") is not None:
        output_file = _output_name(input_file, outdir, sss_label, "trans")
        if os.path.exists(output_file):
            os.remove(output_file)

        stage3_options = _pick_keys(
            options,
            _STAGE3_KEYS,
            overrides={"autobad": None, "force": True},
        )
        _run_single(stage2_output, output_file, stage3_options)


def _run_cbu_3stage(input_file: str, outdir: str, options: dict) -> None:
    """Run the CBU 3-stage Maxfilter pipeline.

    0. **Origin fitting** — fit a sphere to headshape points (nose
       removed).
    1. **Bad channel detection** — run with ``-autobad on``.
    2. **tSSS + movecomp** — apply tSSS with movement compensation.
    3. **Head translation** — transform data to default head position.

    Parameters
    ----------
    input_file : str
        Path to the *original* input FIF file.
    outdir : str
        Output directory.
    options : dict
        Full Maxfilter options from :func:`run_maxfilter`.

    References
    ----------
    https://imaging.mrc-cbu.cam.ac.uk/meg/Maxfilter
    https://imaging.mrc-cbu.cam.ac.uk/meg/maxbugs
    """
    basename = os.path.splitext(os.path.basename(input_file))[0]
    output_base = os.path.join(outdir, basename + "_{0}")

    # Stage 0 - Fit Origin without nose (skipped if pre-provided, e.g. by
    # chain expansion which shares one origin fit across pieces)
    pre_origin = options.get("origin")
    if pre_origin is not None:
        origin = np.asarray(pre_origin, dtype=float)
    else:
        origin = _fit_cbu_origin(input_file, output_base, remove_nose=True)

    # Stage 1 - Find Bad Channels (skipped if bads are pre-provided)
    pre_bads = options.get("bads")
    if pre_bads is None:
        output_file = _output_name(input_file, outdir, "autobad")
        if os.path.exists(output_file):
            os.remove(output_file)

        stage1_options = _pick_keys(
            options,
            _STAGE1_KEYS,
            overrides={
                "autobad": True,
                "origin": origin,
                "frame": "head",
                "autobad_dur": 1800,
                "badlimit": 7,
                "linefreq": 50,
                "hpisubt": "amp",
            },
        )
        output_file, log_file = _run_single(input_file, output_file, stage1_options)
        bads = _parse_bad_channels(log_file) if not options["dryrun"] else None
    else:
        bads = pre_bads

    # Stage 2 - Signal Space Separation
    output_file = _output_name(input_file, outdir, "tsss")
    if os.path.exists(output_file):
        os.remove(output_file)

    stage2_options = _pick_keys(
        options,
        _STAGE2_KEYS,
        overrides={
            "autobad": None,
            "bads": bads,
            "origin": origin,
            "frame": "head",
            "movecompinter": True,
            "st": 10,
            "corr": 0.98,
            "tsss": True,
            "linefreq": 50,
            "hpisubt": "amp",
        },
    )
    if options.get("nomovecompinter"):
        stage2_options["movecompinter"] = False

    stage2_output = output_file
    output_file, log_file = _run_single(input_file, output_file, stage2_options)

    # Stage 3 - Translate to default
    output_file = _output_name(input_file, outdir, "tsss", "trans")
    if os.path.exists(output_file):
        os.remove(output_file)

    translated_origin = [origin[0], origin[1] - 13, origin[2] + 6]
    stage3_options = _pick_keys(
        options,
        _STAGE3_KEYS,
        overrides={
            "autobad": None,
            "trans": "default",
            "origin": list(translated_origin),
            "frame": "head",
            "force": True,
        },
    )
    _run_single(stage2_output, output_file, stage3_options)


def _estimate_output_size(input_file: str) -> tuple[int, float]:
    """Estimate the float32 maxfiltered output size of a FIF recording.

    Parameters
    ----------
    input_file : str
        Path to the input FIF file (the chain head if the recording is
        split across multiple files).

    Returns
    -------
    nbytes : int
        Estimated output size in bytes (``nchan × sfreq × duration × 4``).
    duration : float
        Recording duration in seconds.
    """
    raw = mne.io.read_raw_fif(input_file, allow_maxshield="yes", verbose="error")
    duration = float(raw.times[-1])
    n_chan = raw.info["nchan"]
    sfreq = raw.info["sfreq"]
    return int(n_chan * sfreq * duration * 4), duration


def _needs_chunking(input_file: str, options: dict) -> bool:
    """Return True if *input_file* would exceed ``size_limit_gb`` after maxfilter."""
    size_limit_bytes = options["size_limit_gb"] * 1024**3
    nbytes, _ = _estimate_output_size(input_file)
    return nbytes > size_limit_bytes


def _chunk_boundaries(
    duration: float, max_chunk_dur: float
) -> list[tuple[float, float]]:
    """Uniform ``[t_start, t_end]`` pairs covering ``[0, duration]``."""
    n_chunks = max(1, math.ceil(duration / max_chunk_dur))
    chunk_dur = duration / n_chunks
    return [
        (i * chunk_dur, (i + 1) * chunk_dur if i < n_chunks - 1 else duration)
        for i in range(n_chunks)
    ]


def _skip_arg(t_start: float, t_end: float, duration: float) -> str | None:
    """Build a ``-skip`` argument string that keeps only ``[t_start, t_end]``.

    Returns ``None`` if the chunk covers the entire recording (no skip
    needed).
    """
    intervals: list[float] = []
    if t_start > 0:
        intervals.extend([0.0, t_start])
    if t_end < duration:
        intervals.extend([t_end, duration])
    if not intervals:
        return None
    return " ".join(f"{x:.3f}" for x in intervals)


def _validate_chunked_options(input_file: str, mode: str, options: dict) -> None:
    """Raise a clear error for unsafe option combinations under chunking.

    The CBU 3-stage pipeline is exempt because it always runs
    ``-trans default`` in its final stage, so its chunks are guaranteed
    to share a common head-position reference.
    """
    if mode == "cbu":
        return
    if options.get("movecomp") and not options.get("trans"):
        raise ValueError(
            f"Auto-chunking input '{os.path.basename(input_file)}' with "
            "movecomp=True but no trans reference is unsafe: each chunk's "
            "movement compensation would reference its own first sample, so "
            "the concatenated output would have head-position discontinuities "
            "at chunk boundaries. Pass trans='default' (transforms all "
            "chunks to a standard head position) or "
            "trans='/path/to/reference.fif' (transforms to a specific "
            "reference) to fix this."
        )


def _detect_bads_on_window(
    input_file: str,
    scratch_dir: str,
    options: dict,
    extra: dict | None = None,
) -> str | None:
    """Run autobad on a short window of *input_file* and return static bads.

    Uses the existing :func:`_run_single` machinery with ``-autobad on``
    and a ``-skip`` argument that keeps only the first
    ``options['autobad_scan_dur']`` seconds. The output FIF is small (well
    under the 2 GB limit) and discarded; only the log is parsed for the
    static bad channel list.

    Parameters
    ----------
    input_file : str
        Original input FIF file.
    scratch_dir : str
        Directory to write the throw-away autobad output and log into.
    options : dict
        Full options dict from :func:`run_maxfilter`. Must contain
        ``autobad_scan_dur``.
    extra : dict, optional
        Extra option overrides applied to the autobad call (used by the
        CBU pipeline to inject ``origin``, ``frame``, ``linefreq``, etc.).

    Returns
    -------
    bads : str or None
        Space-separated bad channel list, or ``None`` if no bads found
        or running in dryrun mode.
    """
    os.makedirs(scratch_dir, exist_ok=True)

    _, duration = _estimate_output_size(input_file)
    autobad_skip = _skip_arg(0.0, options["autobad_scan_dur"], duration)

    overrides: dict = {
        "autobad": True,
        "tsss": False,
        "skip": autobad_skip,
        "outdir": scratch_dir,
        "overwrite": True,
    }
    if extra:
        overrides.update(extra)

    autobad_options = _pick_keys(options, _STAGE1_KEYS, overrides=overrides)

    output_file = _output_name(input_file, scratch_dir, "autobad")
    if os.path.exists(output_file):
        os.remove(output_file)
    _, log_file = _run_single(input_file, output_file, autobad_options)

    if options["dryrun"]:
        return None
    return _parse_bad_channels(log_file)


def _concat_and_save_bids(chunk_outputs: list[str], final_path: str) -> None:
    """Concatenate per-chunk maxfiltered FIFs and save as a BIDS split chain.

    Reads each chunk with :func:`mne.io.read_raw_fif`, concatenates them
    via :func:`mne.concatenate_raws`, and writes the result with
    ``split_naming='bids'`` and ``fmt='single'`` (float32) so the
    downstream code reads it as one logical Raw.
    """
    raws = [
        mne.io.read_raw_fif(p, allow_maxshield="yes", verbose="error")
        for p in chunk_outputs
    ]
    combined = mne.concatenate_raws(raws)
    combined.save(
        final_path,
        split_size="1.9GB",
        split_naming="bids",
        fmt="single",
        overwrite=True,
    )


def _persist_headpos_logs(
    scratch: str, outdir: str, input_file: str, n_chunks: int, *labels: str
) -> None:
    """Copy each chunk's head-position log alongside the final output.

    The chunk's headpos log lives next to its stage-2 output FIF inside
    the per-chunk scratch dir, named ``<output>_headpos.log``. This
    helper copies it to the session outdir as
    ``<output>_headpos_partNN.log``.
    """
    for i in range(n_chunks):
        chunk_dir = os.path.join(scratch, f"part{i + 1:02d}")
        chunk_fif = _output_name(input_file, chunk_dir, *labels)
        src = chunk_fif.replace(".fif", "_headpos.log")
        if os.path.exists(src):
            base_fif = _output_name(input_file, outdir, *labels)
            dst = base_fif.replace(".fif", f"_headpos_part{i + 1:02d}.log")
            shutil.copy(src, dst)


def _persist_chain_headpos_logs(
    pieces: list[str],
    piece_outdirs: list[str],
    outdir: str,
    input_file: str,
    sss_label: str,
    final_labels: tuple[str, ...],
) -> None:
    """Copy each chain piece's head-position log alongside the final output.

    Each piece writes its SSS/tSSS stage headpos log next to the
    stage-2 output FIF in its per-piece scratch dir. This helper copies
    each one to the session outdir, named using the final output's
    basename with a ``_headpos_pieceNN.log`` suffix, so they survive
    the ``shutil.rmtree`` of the piece scratch directory.
    """
    base_fif = _output_name(input_file, outdir, *final_labels)
    for i, (piece, piece_outdir) in enumerate(zip(pieces, piece_outdirs)):
        piece_sss_fif = _output_name(piece, piece_outdir, sss_label)
        src = piece_sss_fif.replace(".fif", "_headpos.log")
        if os.path.exists(src):
            dst = base_fif.replace(".fif", f"_headpos_piece{i + 1:02d}.log")
            shutil.copy(src, dst)


def _run_standard_chunked(input_file: str, outdir: str, options: dict) -> None:
    """Chunked variant of ``mode='standard'``.

    Detects static bads once on a window if ``autobad=True`` (and the
    caller did not pre-supply ``bads``), then runs ``_run_single`` once
    per time chunk with the appropriate ``-skip`` and shared ``-bad``,
    and finally concatenates the chunks into a BIDS split chain.
    """
    nbytes, duration = _estimate_output_size(input_file)
    max_chunk_dur = options["size_limit_gb"] * 1024**3 / (nbytes / duration)
    chunks = _chunk_boundaries(duration, max_chunk_dur)

    basename = os.path.splitext(os.path.basename(input_file))[0]
    scratch = os.path.join(outdir, f".chunks_{basename}")
    os.makedirs(scratch, exist_ok=True)

    print(f"  auto-chunking into {len(chunks)} parts (duration {duration:.0f}s)")

    # Detect bads once if requested.
    bads = options.get("bads")
    if options.get("autobad") and bads is None:
        bads = _detect_bads_on_window(
            input_file, os.path.join(scratch, "autobad"), options
        )

    label = "tsss" if options.get("tsss") else "sss"
    chunk_outputs: list[str] = []
    for i, (t_start, t_end) in enumerate(chunks):
        chunk_dir = os.path.join(scratch, f"part{i + 1:02d}")
        os.makedirs(chunk_dir, exist_ok=True)

        chunk_options = dict(options)
        chunk_options["outdir"] = chunk_dir
        chunk_options["skip"] = _skip_arg(t_start, t_end, duration)
        chunk_options["autobad"] = False
        chunk_options["bads"] = bads

        output_file = _output_name(input_file, chunk_dir, label)
        if os.path.exists(output_file):
            os.remove(output_file)
        _run_single(input_file, output_file, chunk_options)
        chunk_outputs.append(output_file)

    if not options["dryrun"]:
        final_out = _output_name(input_file, outdir, label)
        _concat_and_save_bids(chunk_outputs, final_out)
        if options.get("headpos"):
            _persist_headpos_logs(scratch, outdir, input_file, len(chunks), label)
        shutil.rmtree(scratch)


def _run_multistage_chunked(input_file: str, outdir: str, options: dict) -> None:
    """Chunked variant of ``_run_multistage``.

    Stage 1 (autobad) is run once on a short scan window. Stage 2
    (SSS/tSSS) is run once per chunk with the shared bad list and the
    chunk's ``-skip``. Stage 3 (trans) is optionally run on each chunk's
    stage-2 output. The chunked outputs are concatenated and saved as a
    BIDS split chain.
    """
    nbytes, duration = _estimate_output_size(input_file)
    max_chunk_dur = options["size_limit_gb"] * 1024**3 / (nbytes / duration)
    chunks = _chunk_boundaries(duration, max_chunk_dur)

    basename = os.path.splitext(os.path.basename(input_file))[0]
    scratch = os.path.join(outdir, f".chunks_{basename}")
    os.makedirs(scratch, exist_ok=True)

    print(f"  auto-chunking into {len(chunks)} parts (duration {duration:.0f}s)")

    # Stage 1 - Find bad channels on a short window
    bads = _detect_bads_on_window(input_file, os.path.join(scratch, "autobad"), options)

    sss_label = "tsss" if options.get("tsss") else "sss"
    final_labels = (
        (sss_label, "trans") if options.get("trans") is not None else (sss_label,)
    )
    chunk_outputs: list[str] = []

    for i, (t_start, t_end) in enumerate(chunks):
        chunk_dir = os.path.join(scratch, f"part{i + 1:02d}")
        os.makedirs(chunk_dir, exist_ok=True)
        skip = _skip_arg(t_start, t_end, duration)

        # Stage 2 - Signal Space Separation
        stage2_output = _output_name(input_file, chunk_dir, sss_label)
        if os.path.exists(stage2_output):
            os.remove(stage2_output)
        stage2_options = _pick_keys(
            options,
            _STAGE2_KEYS,
            overrides={
                "autobad": None,
                "bads": bads,
                "skip": skip,
                "outdir": chunk_dir,
                "overwrite": True,
            },
        )
        _run_single(input_file, stage2_output, stage2_options)

        # Stage 3 - Translate to reference
        if options.get("trans") is not None:
            trans_output = _output_name(input_file, chunk_dir, sss_label, "trans")
            if os.path.exists(trans_output):
                os.remove(trans_output)
            stage3_options = _pick_keys(
                options,
                _STAGE3_KEYS,
                overrides={
                    "autobad": None,
                    "force": True,
                    "outdir": chunk_dir,
                    "overwrite": True,
                },
            )
            _run_single(stage2_output, trans_output, stage3_options)
            chunk_outputs.append(trans_output)
        else:
            chunk_outputs.append(stage2_output)

    if not options["dryrun"]:
        final_out = _output_name(input_file, outdir, *final_labels)
        _concat_and_save_bids(chunk_outputs, final_out)
        if options.get("headpos"):
            _persist_headpos_logs(scratch, outdir, input_file, len(chunks), sss_label)
        shutil.rmtree(scratch)


def _run_cbu_3stage_chunked(input_file: str, outdir: str, options: dict) -> None:
    """Chunked variant of ``_run_cbu_3stage``.

    Stage 0 (origin fitting) and stage 1 (autobad on a window) run once.
    Stages 2 (tSSS+movecomp) and 3 (trans-to-default) run per chunk
    with the shared bads, origin, and chunk-specific ``-skip``. Outputs
    are concatenated and saved as a BIDS split chain.
    """
    nbytes, duration = _estimate_output_size(input_file)
    max_chunk_dur = options["size_limit_gb"] * 1024**3 / (nbytes / duration)
    chunks = _chunk_boundaries(duration, max_chunk_dur)

    basename = os.path.splitext(os.path.basename(input_file))[0]
    scratch = os.path.join(outdir, f".chunks_{basename}")
    os.makedirs(scratch, exist_ok=True)

    print(f"  auto-chunking into {len(chunks)} parts (duration {duration:.0f}s)")

    # Stage 0 - Fit origin (one-time, time-independent)
    output_base = os.path.join(outdir, basename + "_{0}")
    origin = _fit_cbu_origin(input_file, output_base, remove_nose=True)

    # Stage 1 - Find bad channels on a short window with CBU options
    bads = _detect_bads_on_window(
        input_file,
        os.path.join(scratch, "autobad"),
        options,
        extra={
            "origin": origin,
            "frame": "head",
            "autobad_dur": 1800,
            "badlimit": 7,
            "linefreq": 50,
            "hpisubt": "amp",
        },
    )

    chunk_outputs: list[str] = []
    translated_origin = [origin[0], origin[1] - 13, origin[2] + 6]

    for i, (t_start, t_end) in enumerate(chunks):
        chunk_dir = os.path.join(scratch, f"part{i + 1:02d}")
        os.makedirs(chunk_dir, exist_ok=True)
        skip = _skip_arg(t_start, t_end, duration)

        # Stage 2 - tSSS + movecomp
        stage2_output = _output_name(input_file, chunk_dir, "tsss")
        if os.path.exists(stage2_output):
            os.remove(stage2_output)
        stage2_options = _pick_keys(
            options,
            _STAGE2_KEYS,
            overrides={
                "autobad": None,
                "bads": bads,
                "origin": origin,
                "frame": "head",
                "movecompinter": True,
                "st": 10,
                "corr": 0.98,
                "tsss": True,
                "linefreq": 50,
                "hpisubt": "amp",
                "skip": skip,
                "outdir": chunk_dir,
                "overwrite": True,
            },
        )
        if options.get("nomovecompinter"):
            stage2_options["movecompinter"] = False
        _run_single(input_file, stage2_output, stage2_options)

        # Stage 3 - Translate to default
        trans_output = _output_name(input_file, chunk_dir, "tsss", "trans")
        if os.path.exists(trans_output):
            os.remove(trans_output)
        stage3_options = _pick_keys(
            options,
            _STAGE3_KEYS,
            overrides={
                "autobad": None,
                "trans": "default",
                "origin": list(translated_origin),
                "frame": "head",
                "force": True,
                "outdir": chunk_dir,
                "overwrite": True,
            },
        )
        _run_single(stage2_output, trans_output, stage3_options)
        chunk_outputs.append(trans_output)

    if not options["dryrun"]:
        final_out = _output_name(input_file, outdir, "tsss", "trans")
        _concat_and_save_bids(chunk_outputs, final_out)
        if options.get("headpos"):
            _persist_headpos_logs(scratch, outdir, input_file, len(chunks), "tsss")
        shutil.rmtree(scratch)


def _is_split_chain(input_file: str) -> bool:
    """Return True if *input_file* is the head of a multi-file FIF chain.

    MNE traverses the chain when opening the head file, so
    ``raw.filenames`` contains one entry per split. Maxfilter-2.2, in
    contrast, only processes the head file, which is why chains must
    be pre-sliced before being handed to maxfilter.
    """
    raw = mne.io.read_raw_fif(input_file, allow_maxshield="yes", verbose="error")
    return len(raw.filenames) > 1


def _expand_split_chain(
    input_file: str, scratch_dir: str, size_limit_gb: float
) -> list[str]:
    """Re-emit a split FIF chain as a list of standalone single-file FIFs.

    Maxfilter-2.2 only processes the *head* file of a split chain, so any
    pipeline that points maxfilter at a chain silently drops the
    continuation splits. This helper uses MNE (which does traverse the
    chain) to crop the recording into independent FIFs, each sized under
    ``size_limit_gb`` so that downstream maxfilter calls never need
    ``-skip`` chunking on a piece.

    Parameters
    ----------
    input_file : str
        Head file of a split FIF chain.
    scratch_dir : str
        Directory to write the piece FIFs into.
    size_limit_gb : float
        Upper bound (in GiB) on the float32 maxfiltered output of each
        piece. Pieces are sized at 90% of this limit to leave headroom.

    Returns
    -------
    piece_paths : list of str
        Paths to the standalone single-file FIFs written into
        ``scratch_dir``, one per piece, in time order.
    """
    os.makedirs(scratch_dir, exist_ok=True)

    raw = mne.io.read_raw_fif(input_file, allow_maxshield="yes", verbose="error")

    n_chan = raw.info["nchan"]
    sfreq = raw.info["sfreq"]
    bytes_per_sec = n_chan * sfreq * 4
    target_bytes = 0.9 * size_limit_gb * 1024**3
    max_piece_dur = target_bytes / bytes_per_sec

    duration = float(raw.times[-1])
    n_pieces = max(1, math.ceil(duration / max_piece_dur))
    piece_dur = duration / n_pieces

    basename = os.path.splitext(os.path.basename(input_file))[0]
    piece_paths: list[str] = []
    print(
        f"  pre-slicing chain of {len(raw.filenames)} split(s) into {n_pieces} "
        f"standalone piece(s)"
    )
    for i in range(n_pieces):
        t_start = i * piece_dur
        t_end = (i + 1) * piece_dur if i < n_pieces - 1 else duration

        piece_raw = mne.io.read_raw_fif(
            input_file, allow_maxshield="yes", verbose="error"
        )
        piece_raw.crop(tmin=t_start, tmax=t_end)

        piece_path = os.path.join(scratch_dir, f"{basename}_piece{i + 1:02d}.fif")
        piece_raw.save(
            piece_path,
            fmt="single",
            overwrite=True,
            verbose="error",
        )
        piece_paths.append(piece_path)

    return piece_paths


def _run_pipeline_on_file(
    input_file: str, outdir: str, options: dict, mode: str
) -> None:
    """Dispatch a standalone FIF (not a chain) to the requested mode.

    Auto-chunks oversized recordings via ``-skip`` chunking if the
    estimated float32 maxfiltered output exceeds ``size_limit_gb``.
    Split chains must be pre-expanded by :func:`_expand_split_chain`
    before calling this helper, since maxfilter does not follow split
    chains.
    """
    if _needs_chunking(input_file, options):
        _validate_chunked_options(input_file, mode, options)
        if mode == "standard":
            _run_standard_chunked(input_file, outdir, options)
        elif mode == "multistage":
            _run_multistage_chunked(input_file, outdir, options)
        elif mode == "cbu":
            _run_cbu_3stage_chunked(input_file, outdir, options)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'standard', 'multistage' or 'cbu'."
            )
        return

    if mode == "standard":
        sss_label = "tsss" if options.get("tsss") else "sss"
        output_file = _output_name(input_file, outdir, sss_label)
        _run_single(input_file, output_file, options)
    elif mode == "multistage":
        _run_multistage(input_file, outdir, options)
    elif mode == "cbu":
        _run_cbu_3stage(input_file, outdir, options)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use 'standard', 'multistage' or 'cbu'."
        )


def run_maxfilter(
    files: str | list[str],
    outdir: str,
    maxpath: str = "/neuro/bin/util/maxfilter-2.2",
    mode: str = "standard",
    scanner: str | None = None,
    tsss: bool = False,
    st: float = 10,
    corr: float = 0.98,
    headpos: bool = False,
    movecomp: bool = False,
    movecompinter: bool = False,
    nomovecompinter: bool = False,
    autobad: bool = False,
    autobad_dur: int | None = None,
    bads: str | None = None,
    badlimit: int | None = None,
    trans: str | None = None,
    origin: list[float] | None = None,
    frame: str | None = None,
    force: bool = False,
    inorder: int | None = None,
    outorder: int | None = None,
    hpie: int | None = None,
    hpig: float | None = None,
    ctc: str | None = None,
    cal: str | None = None,
    linefreq: int | None = None,
    hpisubt: str | None = None,
    skip: str | None = None,
    size_limit_gb: float = 1.9,
    autobad_scan_dur: float = 600.0,
    overwrite: bool = False,
    dryrun: bool = False,
) -> None:
    """Run Maxfilter on one or more FIF files.

    Parameters
    ----------
    files : str or list of str
        Path to a FIF file, a glob pattern, or a list of file paths.
    outdir : str
        Output directory.
    maxpath : str, optional
        Path to the maxfilter binary.
    mode : str, optional
        Running mode: ``"standard"``, ``"multistage"`` or ``"cbu"``.
    scanner : str, optional
        OHBA scanner name (``"VectorView"``, ``"VectorView2"`` or ``"Neo"``).
        Sets CTC and Cal files automatically. Overrides ``ctc`` and ``cal``.
    tsss : bool, optional
        Apply temporal extension of maxfilter (tSSS).
    st : float, optional
        Data buffer length for tSSS processing (default 10).
    corr : float, optional
        Subspace correlation limit for tSSS (default 0.98).
    headpos : bool, optional
        Output additional head movement parameter file.
    movecomp : bool, optional
        Apply movement compensation.
    movecompinter : bool, optional
        Apply movement compensation on data with intermittent HPI.
    nomovecompinter : bool, optional
        Remove default movecomp in the CBU 3-stage pipeline.
    autobad : bool, optional
        Apply automatic bad channel detection.
    autobad_dur : int, optional
        Set autobad with a specific duration.
    bads : str, optional
        Static bad channels as a space-separated string of channel numbers,
        e.g. ``"2233 2312"``.
    badlimit : int, optional
        Upper limit for number of bad channels to be removed.
    trans : str, optional
        Transform data to the head position in the specified file, or
        ``"default"`` for the default head position.
    origin : list of float, optional
        Custom sphere origin [x, y, z].
    frame : str, optional
        Coordinate frame for the origin.
    force : bool, optional
        Ignore program warnings.
    inorder : int, optional
        Order of the inside expansion.
    outorder : int, optional
        Order of the outside expansion.
    hpie : int, optional
        Error limit for HPI coil fitting (mm).
    hpig : float, optional
        Goodness-of-fit limit for HPI coil fitting.
    ctc : str, optional
        Path to cross-talk calibration file.
    cal : str, optional
        Path to fine-calibration file.
    linefreq : int, optional
        Line interference frequency (50 or 60 Hz).
    hpisubt : str, optional
        HPI signal subtraction mode.
    skip : str, optional
        Time intervals (in seconds) to skip from processing, given as a
        space-separated list of start/stop pairs, e.g. ``"0 30 120 150"``.
        Maps to MaxFilter's ``-skip`` flag.
    size_limit_gb : float, optional
        Threshold above which oversized recordings are automatically
        chunked using ``-skip`` (default 1.9 GB, leaving 5% headroom
        under the 2 GB single-FIF float32 limit). Each input file's
        estimated float32 maxfiltered output is checked; if it exceeds
        this limit, the recording is processed in time chunks and the
        chunked outputs are concatenated and saved as a BIDS split chain
        so :func:`mne.io.read_raw_fif` reads the result as a single Raw.
        Static bad channels are detected once on a short window (see
        ``autobad_scan_dur``) and shared across chunks. Movement
        compensation across chunks requires an explicit trans reference;
        passing ``movecomp=True`` without ``trans`` will raise.
    autobad_scan_dur : float, optional
        Duration in seconds of the autobad scan window used for
        chunked sessions (default 600 s = 10 min). Long enough for
        reliable bad-channel detection, short enough that the autobad
        FIF output stays well under the 2 GB limit at typical channel
        counts.
    overwrite : bool, optional
        Overwrite existing output files.
    dryrun : bool, optional
        Print commands without executing.

    Notes
    -----
    The recommended use at OHBA is multistage maxfiltering::

        from osl_dynamics.meeg.maxfilter import run_maxfilter

        run_maxfilter(
            files="sub-01_task-rest_meg.fif",
            outdir="derivatives/maxfiltered",
            scanner="Neo",
            mode="multistage",
            tsss=True,
            headpos=True,
            movecomp=True,
        )

    Output files follow the BIDS-MEG ``proc-`` entity convention::

        sub-01_task-rest_proc-sss_meg.fif          # SSS
        sub-01_task-rest_proc-tsss_meg.fif         # tSSS
        sub-01_task-rest_proc-tsss+trans_meg.fif   # tSSS + trans
    """
    # Build options dict from all parameters
    options = {k: v for k, v in locals().items() if k != "files"}

    # Resolve file list
    if isinstance(files, str):
        input_files = sorted(glob.glob(files))
        if not input_files:
            raise FileNotFoundError(f"No files found matching: {files}")
    elif isinstance(files, list):
        input_files = files
    else:
        raise ValueError("files must be a string or list of strings")

    # Validate trans file if specified
    if trans is not None and trans != "default":
        if not os.path.isfile(trans):
            raise FileNotFoundError(f"Trans file not found: {trans}")

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    print("\n\nMaxfilter")
    print("---------\n")
    print(f"Processing {len(input_files)} file(s)")
    print(f"Output directory: {outdir}\n")

    for idx, input_file in enumerate(input_files):
        print(f"Processing {idx + 1}/{len(input_files)} : {input_file}")

        if not os.path.isfile(input_file):
            print(f"Input file not found, skipping ({input_file})")
            continue

        # Determine the final output filename for the skip check and for
        # chain-expansion concatenation. Depends on the mode and whether
        # trans is requested.
        sss_label = "tsss" if tsss else "sss"
        if mode == "cbu":
            final_labels = ("tsss", "trans")
        elif mode == "multistage" and trans is not None:
            final_labels = (sss_label, "trans")
        else:
            final_labels = (sss_label,)
        final_output = _output_name(input_file, outdir, *final_labels)

        # Skip if output exists and we don't want to overwrite
        if os.path.isfile(final_output) and not overwrite:
            print(f"Output exists, skipping ({final_output})")
            continue

        # Split FIF chains need to be pre-sliced: maxfilter-2.2 only
        # processes the head file of a chain, so pointing it at a chain
        # silently drops the continuation splits. We use MNE (which does
        # follow the chain) to re-emit the recording as independent
        # single-file pieces, run the full pipeline on each piece, and
        # concatenate the piece outputs into the final BIDS split chain.
        if _is_split_chain(input_file):
            _validate_chunked_options(input_file, mode, options)
            basename = os.path.splitext(os.path.basename(input_file))[0]
            piece_scratch = os.path.join(outdir, f".pieces_{basename}")
            os.makedirs(piece_scratch, exist_ok=True)

            # Pre-detect bad channels (and origin for CBU) once across the
            # whole chain so every piece shares the same list. The pieces
            # come from the same recording session, so their bads are
            # identical
            piece_options = dict(options)
            if mode == "cbu":
                if piece_options.get("origin") is None:
                    shared_origin = _fit_cbu_origin(
                        input_file,
                        os.path.join(outdir, basename + "_{0}"),
                        remove_nose=True,
                    )
                    piece_options["origin"] = list(shared_origin)
                    piece_options["frame"] = "head"
                if piece_options.get("bads") is None:
                    piece_options["bads"] = _detect_bads_on_window(
                        input_file,
                        os.path.join(piece_scratch, "autobad"),
                        options,
                        extra={
                            "origin": piece_options["origin"],
                            "frame": "head",
                            "autobad_dur": 1800,
                            "badlimit": 7,
                            "linefreq": 50,
                            "hpisubt": "amp",
                        },
                    )
            elif autobad and piece_options.get("bads") is None:
                piece_options["bads"] = _detect_bads_on_window(
                    input_file,
                    os.path.join(piece_scratch, "autobad"),
                    options,
                )
            # Prevent per-piece pipelines from re-running autobad now that
            # the shared bad-channel list is set.
            piece_options["autobad"] = False

            pieces = _expand_split_chain(input_file, piece_scratch, size_limit_gb)
            piece_outputs: list[str] = []
            piece_outdirs: list[str] = []
            for piece in pieces:
                piece_name = os.path.splitext(os.path.basename(piece))[0]
                piece_outdir = os.path.join(piece_scratch, f"{piece_name}_out")
                os.makedirs(piece_outdir, exist_ok=True)
                _run_pipeline_on_file(piece, piece_outdir, piece_options, mode)
                piece_outputs.append(_output_name(piece, piece_outdir, *final_labels))
                piece_outdirs.append(piece_outdir)

            if not dryrun:
                _concat_and_save_bids(piece_outputs, final_output)
                if headpos:
                    piece_sss_label = "tsss" if mode == "cbu" else sss_label
                    _persist_chain_headpos_logs(
                        pieces,
                        piece_outdirs,
                        outdir,
                        input_file,
                        piece_sss_label,
                        final_labels,
                    )
                shutil.rmtree(piece_scratch)
            continue

        _run_pipeline_on_file(input_file, outdir, options, mode)

    print("\nMaxfiltering complete.\n")
