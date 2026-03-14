"""Generate a QC summary HTML report from pipeline plots."""

import base64
import html
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

STEPS = {
    1: {
        "name": "Preprocessing",
        "files": [
            "1_psd.png",
            "1_sum_square.png",
            "1_sum_square_exclude_bads.png",
            "1_channel_stds.png",
        ],
        "large": ["1_psd.png"],
    },
    2: {
        "name": "Surfaces",
        "files": [
            "2_inskull.png",
            "2_outskin.png",
            "2_outskull.png",
        ],
    },
    3: {
        "name": "Coregistration",
        "files": ["3_coreg.png"],
        "large": ["3_coreg.png"],
    },
    4: {
        "name": "Source Recon & Parcellation",
        "files": [
            "4_psd_topo.png",
        ],
        "large": ["4_psd_topo.png"],
    },
}

CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0;
    padding: 20px;
    background: #f5f5f5;
    color: #333;
}
h1 {
    margin: 0 0 20px 0;
}
.tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 20px;
    border-bottom: 2px solid #ddd;
}
.tab-btn {
    flex: 1;
    padding: 10px 20px;
    border: none;
    background: #e0e0e0;
    cursor: pointer;
    font-size: 14px;
    border-radius: 6px 6px 0 0;
    transition: background 0.2s;
    text-align: center;
}
.tab-btn:hover {
    background: #d0d0d0;
}
.tab-btn.active {
    background: #fff;
    font-weight: bold;
    border-bottom: 2px solid #fff;
    margin-bottom: -2px;
}
.tab-content {
    display: none;
    background: #fff;
    padding: 20px;
    border-radius: 0 0 6px 6px;
    max-height: calc(100vh - 160px);
    overflow-y: auto;
}
.tab-content.active {
    display: block;
}
.session-nav {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding: 10px;
    background: #f0f0f0;
    border-radius: 6px;
}
.session-nav button {
    padding: 6px 14px;
    border: 1px solid #ccc;
    background: #fff;
    cursor: pointer;
    border-radius: 4px;
    font-size: 18px;
    line-height: 1;
}
.session-nav button:hover {
    background: #e8e8e8;
}
.session-nav input {
    padding: 6px 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 14px;
    font-family: monospace;
    width: 300px;
}
.session-nav .counter {
    font-size: 13px;
    color: #888;
    margin-left: auto;
}
.session-panel {
    display: none;
}
.session-panel.active {
    display: block;
}
.session-panel img {
    max-width: 100%;
    max-height: 250px;
    display: block;
    margin: 4px auto;
    border: 1px solid #eee;
}
.session-panel img.large {
    max-height: 400px;
}
.session-panel iframe {
    width: 100%;
    height: 350px;
    border: 1px solid #ddd;
    border-radius: 4px;
    margin: 4px 0;
}
.image-row {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 12px 0;
}
.image-row-item {
    flex: 1;
    min-width: 0;
    text-align: center;
}
.image-row-item img {
    max-width: 100%;
    max-height: 200px;
    border: 1px solid #eee;
}
.placeholder {
    background: #eee;
    color: #999;
    padding: 40px;
    text-align: center;
    border-radius: 4px;
    margin: 8px auto;
    font-style: italic;
}
.file-label {
    font-size: 12px;
    color: #888;
    margin: 12px 0 2px 0;
    text-align: center;
}
.summary-box {
    background: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 12px;
    font-size: 13px;
    line-height: 1.6;
}
.summary-box .label {
    color: #888;
    font-size: 12px;
}
.summary-box .bad {
    color: #c0392b;
    font-weight: bold;
}
.footer {
    margin-top: 30px;
    padding-top: 15px;
    border-top: 1px solid #ddd;
    font-size: 12px;
    color: #999;
}
"""

JS = """
var sessions = SESSION_LIST;
var currentStep = 1;
var currentIdx = {};  // per-step session index

// Initialise each step to session 0
STEP_NUMS.forEach(function(s) { currentIdx[s] = 0; });

function switchTab(step) {
    currentStep = step;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('btn-' + step).classList.add('active');
    document.getElementById('tab-' + step).classList.add('active');
    showSession(step, currentIdx[step]);
}

function showSession(step, idx) {
    if (idx < 0) idx = 0;
    if (idx >= sessions.length) idx = sessions.length - 1;
    currentIdx[step] = idx;

    // Hide all session panels for this step
    var panels = document.querySelectorAll('#tab-' + step + ' .session-panel');
    panels.forEach(p => p.classList.remove('active'));

    // Show the selected one
    var panel = document.getElementById('step-' + step + '-session-' + idx);
    if (panel) panel.classList.add('active');

    // Update input and counter
    var input = document.getElementById('input-' + step);
    var counter = document.getElementById('counter-' + step);
    input.value = sessions[idx];
    counter.textContent = (idx + 1) + ' / ' + sessions.length;
}

function prevSession(step) {
    showSession(step, currentIdx[step] - 1);
}

function nextSession(step) {
    showSession(step, currentIdx[step] + 1);
}

function jumpToSession(step) {
    var input = document.getElementById('input-' + step);
    var val = input.value.trim().toLowerCase();
    for (var i = 0; i < sessions.length; i++) {
        if (sessions[i].toLowerCase() === val) {
            showSession(step, i);
            return;
        }
    }
    // Partial match: find first session containing the input
    for (var i = 0; i < sessions.length; i++) {
        if (sessions[i].toLowerCase().indexOf(val) !== -1) {
            showSession(step, i);
            return;
        }
    }
}

document.addEventListener('keydown', function(e) {
    // Ignore if user is typing in the input
    if (document.activeElement.tagName === 'INPUT') {
        if (e.key === 'Enter') {
            e.preventDefault();
            jumpToSession(currentStep);
        }
        return;
    }
    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        prevSession(currentStep);
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        nextSession(currentStep);
    }
});
"""


def _build_summary(session_dir: Path) -> str:
    """Build HTML for a preprocessing summary box, if summary exists."""
    summary_file = session_dir / "1_summary.json"
    if not summary_file.exists():
        return ""
    with open(summary_file) as f:
        s = json.load(f)
    total = s["total_duration_s"]
    bad = s["bad_duration_s"]
    pct = s["bad_percent"]
    n_bad_ch = s["n_bad_channels"]
    bad_chs = ", ".join(s["bad_channels"]) if s["bad_channels"] else "none"
    return (
        f'<div class="summary-box">'
        f'<span class="label">Duration:</span> {total:.1f}s &nbsp; | &nbsp; '
        f'<span class="label">Bad segments:</span> '
        f'<span class="bad">{bad:.1f}s ({pct:.1f}%)</span> &nbsp; | &nbsp; '
        f'<span class="label">Bad channels ({n_bad_ch}):</span> {bad_chs}'
        f"</div>"
    )


def _embed_png(filepath: Path, css_class: str = "") -> str:
    """Encode a PNG file as a base64 data URI."""
    data = filepath.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    cls = f' class="{css_class}"' if css_class else ""
    return f'<img{cls} src="data:image/png;base64,{b64}">'


def _embed_html(filepath: Path) -> str:
    """Embed an HTML file as an iframe with srcdoc."""
    content = filepath.read_text(errors="replace")
    escaped = html.escape(content, quote=True)
    return f'<iframe srcdoc="{escaped}" loading="lazy"></iframe>'


def _build_step_tab(
    step_num: int,
    step_info: Dict,
    plots_dir: Path,
    session_ids: List[str],
) -> Tuple[str, int, int]:
    """Build the HTML content for a single step tab."""
    files = step_info["files"]
    large_files = step_info.get("large", [])
    row_files = step_info.get("row_files", [])
    all_filenames = files + row_files

    # Count how many sessions have at least one file for this step
    done = 0
    for session_id in session_ids:
        session_dir = plots_dir / session_id
        if any((session_dir / f).exists() for f in all_filenames):
            done += 1
    total = len(session_ids)

    parts = []

    # Session navigator
    parts.append(f'<div class="session-nav">')
    parts.append(f'<button onclick="prevSession({step_num})">&#9664;</button>')
    parts.append(
        f'<input type="text" id="input-{step_num}" '
        f'value="{session_ids[0] if session_ids else ""}" '
        f"onkeydown=\"if(event.key==='Enter')jumpToSession({step_num})\" "
        f'placeholder="Type session ID...">'
    )
    parts.append(f'<button onclick="nextSession({step_num})">&#9654;</button>')
    parts.append(
        f'<span class="counter" id="counter-{step_num}">' f"1 / {total}</span>"
    )
    parts.append("</div>")

    # Session panels (one per session, only one visible at a time)
    for idx, session_id in enumerate(session_ids):
        session_dir = plots_dir / session_id
        active = " active" if idx == 0 else ""
        parts.append(
            f'<div class="session-panel{active}" '
            f'id="step-{step_num}-session-{idx}">'
        )

        # Add preprocessing summary if this is step 1
        if step_num == 1:
            parts.append(_build_summary(session_dir))

        for filename in files:
            filepath = session_dir / filename
            parts.append(f'<div class="file-label">{filename}</div>')
            if filepath.exists():
                if filename.endswith(".png"):
                    css_class = "large" if filename in large_files else ""
                    parts.append(_embed_png(filepath, css_class=css_class))
                elif filename.endswith(".html"):
                    parts.append(_embed_html(filepath))
            else:
                parts.append('<div class="placeholder">Pending</div>')

        # Render row files (e.g. power maps in a horizontal row)
        if row_files:
            row_labels = step_info.get("row_labels", [])
            parts.append('<div class="image-row">')
            for i, filename in enumerate(row_files):
                filepath = session_dir / filename
                parts.append('<div class="image-row-item">')
                if i < len(row_labels):
                    parts.append(f'<div class="file-label">{row_labels[i]}</div>')
                if filepath.exists():
                    parts.append(_embed_png(filepath))
                else:
                    parts.append('<div class="placeholder">Pending</div>')
                parts.append("</div>")
            parts.append("</div>")

        parts.append("</div>")

    return "\n".join(parts), done, total


def _copy_surface_plots(
    plots_dir: Path,
    sessions: Dict,
    output_dir: Path,
) -> None:
    """Copy surface extraction PNGs from the derivatives directory.

    Looks up the subject for each session and copies any surface PNGs
    from output_dir/anat_surfaces/<subject>/ into the session's plots
    directory with a ``2_`` prefix.

    Parameters
    ----------
    plots_dir : Path
        Path to the plots directory.
    sessions : dict
        Sessions dictionary mapping session IDs to info dicts.
    output_dir : Path
        Path to the derivatives directory.
    """
    for session_id, info in sessions.items():
        subject = info.get("subject")
        if subject is None:
            continue
        surfaces_dir = output_dir / "anat_surfaces" / subject
        if not surfaces_dir.exists():
            continue
        session_plots_dir = plots_dir / session_id
        session_plots_dir.mkdir(parents=True, exist_ok=True)
        for png in surfaces_dir.glob("*.png"):
            shutil.copy(png, session_plots_dir / f"2_{png.name}")


def _copy_coreg_plots(
    plots_dir: Path,
    sessions: Dict,
    output_dir: Path,
) -> None:
    """Copy coregistration PNGs from the derivatives directory.

    Copies ``coreg.png`` from ``output_dir/osl/<session_id>/coreg/``
    into the session's plots directory as ``3_coreg.png``.

    Parameters
    ----------
    plots_dir : Path
        Path to the plots directory.
    sessions : dict
        Sessions dictionary mapping session IDs to info dicts.
    output_dir : Path
        Path to the derivatives directory.
    """
    for session_id in sessions:
        coreg_png = output_dir / "osl" / session_id / "coreg" / "coreg.png"
        if not coreg_png.exists():
            continue
        session_plots_dir = plots_dir / session_id
        session_plots_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(coreg_png, session_plots_dir / "3_coreg.png")


def _copy_parc_plots(
    plots_dir: Path,
    sessions: Dict,
    output_dir: Path,
) -> None:
    """Copy parcellation QC PNGs from the derivatives directory.

    Copies ``psd_topo.png`` from ``output_dir/osl/<session_id>/``
    into the session's plots directory as ``4_psd_topo.png``.

    Parameters
    ----------
    plots_dir : Path
        Path to the plots directory.
    sessions : dict
        Sessions dictionary mapping session IDs to info dicts.
    output_dir : Path
        Path to the derivatives directory.
    """
    for session_id in sessions:
        session_plots_dir = plots_dir / session_id
        osl_dir = output_dir / "osl" / session_id
        for src_name, dst_name in [
            ("psd_topo.png", "4_psd_topo.png"),
            ("power_maps.png", "4_power_maps.png"),
        ]:
            src = osl_dir / src_name
            if src.exists():
                session_plots_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, session_plots_dir / dst_name)


def generate_report(
    plots_dir: Union[str, Path],
    sessions: Dict,
    output_dir: Optional[Union[str, Path]] = None,
    output_file: str = "report.html",
) -> None:
    """Generate a QC summary HTML report.

    Scans the plots directory for existing QC files and builds a
    self-contained HTML report with tabs for each pipeline step.
    Sessions are navigated one at a time using arrow keys or a
    text input field.

    Parameters
    ----------
    plots_dir : str or Path
        Path to the plots directory containing per-session subdirectories.
    sessions : dict
        Dictionary of sessions (same format as the pipeline scripts).
    output_dir : str or Path, optional
        Path to the derivatives directory. If provided, surface extraction,
        coregistration, and parcellation plots are copied from here into
        the plots directory.
    output_file : str, optional
        Filename for the report. Written to plots_dir/output_file.
    """
    plots_dir = Path(plots_dir)
    session_ids = list(sessions.keys())

    # Copy plots from derivatives if available
    if output_dir is not None:
        _copy_surface_plots(plots_dir, sessions, Path(output_dir))
        _copy_coreg_plots(plots_dir, sessions, Path(output_dir))
        _copy_parc_plots(plots_dir, sessions, Path(output_dir))

    # Add power maps to step 4 if any session has them
    has_power_maps = any(
        (plots_dir / sid / "4_power_maps.png").exists() for sid in session_ids
    )
    steps = {k: dict(v) for k, v in STEPS.items()}
    if has_power_maps:
        steps[4] = dict(steps[4])
        steps[4]["files"] = list(steps[4]["files"]) + ["4_power_maps.png"]
        steps[4]["large"] = list(steps[4].get("large", [])) + ["4_power_maps.png"]

    # Build tab buttons and content
    tab_buttons = []
    tab_contents = []

    first = True
    for step_num, step_info in steps.items():
        content, done, total = _build_step_tab(
            step_num, step_info, plots_dir, session_ids
        )
        active = " active" if first else ""
        tab_buttons.append(
            f'<button class="tab-btn{active}" id="btn-{step_num}" '
            f'onclick="switchTab({step_num})">'
            f'Step {step_num}: {step_info["name"]} ({done}/{total})'
            f"</button>"
        )
        tab_contents.append(
            f'<div class="tab-content{active}" id="tab-{step_num}">' f"{content}</div>"
        )
        first = False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Inject session list and step numbers into JS
    session_list_js = json.dumps(session_ids)
    step_nums_js = json.dumps(list(STEPS.keys()))
    js = JS.replace("SESSION_LIST", session_list_js)
    js = js.replace("STEP_NUMS", step_nums_js)

    report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>QC Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>QC Report</h1>
<div class="tabs">
{"".join(tab_buttons)}
</div>
{"".join(tab_contents)}
<div class="footer">Generated: {timestamp}</div>
<script>{js}</script>
</body>
</html>"""

    output_path = plots_dir / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report saved: {output_path}")
