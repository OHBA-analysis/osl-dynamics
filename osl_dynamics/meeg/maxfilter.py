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

CBU 3-stage pipeline with a single file::

    run_maxfilter(
        files="sub-01_task-rest_meg.fif",
        outdir="derivatives/maxfiltered",
        mode="cbu",
    )
"""

from __future__ import annotations

import glob
import os
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


def _output_name(input_file: str, outdir: str, label: str) -> str:
    """Build a BIDS-like output filename.

    Given ``input_file="path/to/sub-01_task-rest_meg.fif"`` and
    ``label="tsss"``, returns ``"outdir/sub-01_task-rest_raw_tsss.fif"``.
    """
    basename = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(outdir, f"{basename}_raw_{label}.fif")


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
    log_tag: str = "",
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
    log_tag : str, optional
        Tag appended to the log file stem to differentiate pipeline stages
        (e.g. ``"_trans"``).

    Returns
    -------
    output_file : str
        Path to output FIF file.
    log_file : str
        Path to the stdout log.
    """
    command = _build_command(options, input_file, output_file)

    log_file = output_file.replace(".fif", f"{log_tag}.log")
    error_log = output_file.replace(".fif", f"{log_tag}_err.log")

    # Capture stdout and stderr into separate files
    command += f" -v > >(tee -a {log_file}) 2> >(tee -a {error_log} >&2)"

    print(command)
    print()

    if not options["dryrun"]:
        subprocess.run(command, shell=True, executable="/bin/bash")

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
    # Stage 1 - Find Bad Channels
    output_file = _output_name(input_file, outdir, "autobad")
    if os.path.exists(output_file):
        os.remove(output_file)

    stage1_options = _pick_keys(options, _STAGE1_KEYS, overrides={"autobad": True})
    output_file, log_file = _run_single(input_file, output_file, stage1_options)

    bads = _parse_bad_channels(log_file) if not options["dryrun"] else None

    # Stage 2 - Signal Space Separation
    label = "tsss" if options.get("tsss") else "sss"
    output_file = _output_name(input_file, outdir, label)
    if os.path.exists(output_file):
        os.remove(output_file)

    stage2_options = _pick_keys(
        options, _STAGE2_KEYS, overrides={"autobad": None, "bads": bads}
    )
    stage2_output = output_file
    output_file, log_file = _run_single(input_file, output_file, stage2_options)

    # Stage 3 - Translate to reference file
    if options.get("trans") is not None:
        output_file = _output_name(input_file, outdir, "trans")
        if os.path.exists(output_file):
            os.remove(output_file)

        stage3_options = _pick_keys(
            options,
            _STAGE3_KEYS,
            overrides={"autobad": None, "force": True},
        )
        _run_single(stage2_output, output_file, stage3_options, log_tag="_trans")


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

    # Stage 0 - Fit Origin without nose
    origin = _fit_cbu_origin(input_file, output_base, remove_nose=True)

    # Stage 1 - Find Bad Channels
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
    output_file = _output_name(input_file, outdir, "trans")
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
    _run_single(stage2_output, output_file, stage3_options, log_tag="_trans")


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

    Output files follow the MNE-Python/BIDS naming convention::

        sub-01_task-rest_raw_sss.fif     # SSS
        sub-01_task-rest_raw_tsss.fif    # tSSS
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

        # Build output path (for skip check only in standard mode)
        output_file = _output_name(input_file, outdir, "tsss" if tsss else "sss")

        # Skip if output exists and we don't want to overwrite
        if os.path.isfile(output_file) and not overwrite:
            print(f"Output exists, skipping ({output_file})")
            continue

        if mode == "standard":
            _run_single(input_file, output_file, options)
        elif mode == "multistage":
            _run_multistage(input_file, outdir, options)
        elif mode == "cbu":
            _run_cbu_3stage(input_file, outdir, options)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'standard', 'multistage'" " or 'cbu'."
            )

    print("\nMaxfiltering complete.\n")
