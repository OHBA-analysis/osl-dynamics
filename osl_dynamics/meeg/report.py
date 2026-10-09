"""Generate a QC summary HTML report from the pipeline's QC files.

The report is a table with one row per session, holding the numbers each step
saves (bad segments, MNI registration, coregistration error, ...), next to the
QC plots of the selected session. Sort the table by a metric to see the worst
sessions first. Only the plots of the selected session are loaded, so the
report works for datasets with tens of thousands of sessions. The report and
its plots are all in the QC directory, which can be moved or served on its
own.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

# Plots shown in each tab, relative to the QC directory, which has a
# directory for each step holding a directory for each session, subject or
# head model. Plots that have not been saved are not shown.
TABS = {
    "Preprocessing": [
        "preproc/{id}/psd.webp",
        "preproc/{id}/sum_square.webp",
        "preproc/{id}/sum_square_exclude_bads.webp",
        "preproc/{id}/channel_stds.webp",
        "preproc/{id}/ica_components.webp",
    ],
    "Surfaces": [
        "surfaces/{subject}/surfaces.webp",
    ],
    "MNI Registration": [
        "surfaces/{subject}/mni_registration.webp",
    ],
    "Coregistration": [
        "coreg/{head_model}/coreg.webp",
    ],
    "Parcellation": [
        "parc/{id}/psd_topo.webp",
        "parc/{id}/power_maps.webp",
    ],
}

# Notes shown under the plots of a tab
CAPTIONS = {
    "Surfaces": (
        "Yellow: brain surface, cyan: inner skull, magenta: scalp. Each should "
        "follow its boundary and lie inside the next."
    ),
    "MNI Registration": (
        "Red: edges of the MNI152 template. They should follow the anatomy."
    ),
}

CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    background: #f5f5f5;
    color: #333;
    font-size: 13px;
}
header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 16px;
    background: #fff;
    border-bottom: 1px solid #ddd;
}
header h1 {
    font-size: 18px;
    margin: 0;
}
header input[type=text] {
    padding: 5px 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-family: monospace;
    width: 220px;
}
header .info {
    color: #888;
    margin-left: auto;
}
main {
    flex: 1;
    display: flex;
    min-height: 0;
}
#table-pane {
    max-width: 60%;
    display: flex;
    flex-direction: column;
    background: #fff;
    border-right: 1px solid #ddd;
}
#table-scroll {
    flex: 1;
    overflow: auto;
}
table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    padding: 3px 7px;
    text-align: right;
    white-space: nowrap;
}
th:first-child, td:first-child {
    text-align: left;
    font-family: monospace;
}
th {
    position: sticky;
    top: 0;
    background: #eee;
    cursor: pointer;
    user-select: none;
}
th .count {
    display: block;
    color: #999;
    font-weight: normal;
    font-size: 11px;
}
tbody tr {
    cursor: pointer;
    border-bottom: 1px solid #f0f0f0;
}
tbody tr:hover {
    background: #f5f9ff;
}
tbody tr.selected {
    background: #dbe9ff;
}
td.flag {
    color: #c62828;
    font-weight: bold;
}
td.missing {
    color: #ccc;
}
#pager {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 6px;
    border-top: 1px solid #ddd;
    color: #888;
}
button {
    padding: 4px 12px;
    border: 1px solid #ccc;
    background: #fff;
    border-radius: 4px;
    cursor: pointer;
}
button:hover {
    background: #e8e8e8;
}
#plot-pane {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
}
#tabs {
    display: flex;
    gap: 4px;
    padding: 8px 12px 0 12px;
    border-bottom: 1px solid #ddd;
}
#tabs button {
    border-radius: 6px 6px 0 0;
    border-bottom: none;
    background: #e0e0e0;
}
#tabs button.active {
    background: #fff;
    font-weight: bold;
}
#plots {
    flex: 1;
    overflow: auto;
    padding: 12px;
    background: #fff;
}
#plots h2 {
    font-size: 14px;
    font-family: monospace;
    margin: 0 0 10px 0;
}
#plots img {
    max-width: 100%;
    display: block;
    margin: 0 auto 10px auto;
}
.caption {
    color: #666;
    text-align: center;
    margin: 0;
}
.placeholder {
    background: #eee;
    color: #999;
    padding: 40px;
    text-align: center;
    border-radius: 4px;
    font-style: italic;
}
"""

JS = """
const columns = DATA.columns, rows = DATA.rows, tabs = Object.keys(DATA.tabs);
const nInfo = 3;  // id, subject and head model come before the metrics
const pageSize = 200;
let order = [], selected = 0, tab = 0, sortColumn = null, ascending = true;

// Values far from the rest of a column are highlighted: more than 5 scaled
// median absolute deviations from the median (and anything that is true)
const medians = {}, mads = {};
for (let c = nInfo; c < columns.length; c++) {
    const x = rows.map(r => r[c]).filter(v => typeof v === 'number');
    if (x.length === 0) continue;
    medians[c] = median(x);
    mads[c] = 1.4826 * median(x.map(v => Math.abs(v - medians[c])));
}

function median(x) {
    const s = [...x].sort((a, b) => a - b), m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function flagged(row, c) {
    const v = row[c];
    if (v === true) return true;
    if (typeof v !== 'number' || !mads[c]) return false;
    return Math.abs(v - medians[c]) / mads[c] > 5;
}

function format(v) {
    if (v === null) return '-';
    if (typeof v === 'number') return String(parseFloat(v.toPrecision(3)));
    return String(v);
}

function update() {
    const text = document.getElementById('filter').value.toLowerCase();
    const onlyFlagged = document.getElementById('only-flagged').checked;
    order = [];
    rows.forEach((row, i) => {
        if (text && !row[0].toLowerCase().includes(text)) return;
        if (onlyFlagged && !columns.some((_, c) => c >= nInfo && flagged(row, c))) return;
        order.push(i);
    });
    if (sortColumn !== null) {
        order.sort((a, b) => {
            const x = rows[a][sortColumn], y = rows[b][sortColumn];
            if (x === y) return a - b;
            if (x === null) return 1;
            if (y === null) return -1;
            return (x < y ? -1 : 1) * (ascending ? 1 : -1);
        });
    }
    selected = 0;
    render();
}

function sortBy(c) {
    ascending = sortColumn === c ? !ascending : true;
    sortColumn = c;
    update();
}

function render() {
    // Table, one page at a time
    const page = Math.floor(selected / pageSize);
    const nPages = Math.max(1, Math.ceil(order.length / pageSize));
    let html = '<thead><tr>';
    columns.forEach((name, c) => {
        if (c > 0 && c < nInfo) return;
        const n = rows.filter(r => r[c] !== null).length;
        const arrow = sortColumn === c ? (ascending ? ' &#9650;' : ' &#9660;') : '';
        html += `<th onclick="sortBy(${c})">${name}${arrow}<span class="count">${n}</span></th>`;
    });
    html += '</tr></thead><tbody>';
    for (let i = page * pageSize; i < Math.min(order.length, (page + 1) * pageSize); i++) {
        const row = rows[order[i]];
        html += `<tr id="row-${i}" class="${i === selected ? 'selected' : ''}" onclick="select(${i})">`;
        row.forEach((v, c) => {
            if (c > 0 && c < nInfo) return;
            const cls = v === null ? 'missing' : (c >= nInfo && flagged(row, c) ? 'flag' : '');
            html += `<td class="${cls}">${format(v)}</td>`;
        });
        html += '</tr>';
    }
    document.getElementById('table').innerHTML = html + '</tbody>';
    document.getElementById('page').textContent = `Page ${page + 1} / ${nPages}`;
    document.getElementById('count').textContent = `${order.length} / ${rows.length} sessions`;
    const tr = document.getElementById(`row-${selected}`);
    if (tr) tr.scrollIntoView({block: 'nearest'});

    // Tabs
    document.getElementById('tabs').innerHTML = tabs.map((name, t) =>
        `<button class="${t === tab ? 'active' : ''}" onclick="showTab(${t})">${name}</button>`
    ).join('');

    // Plots of the selected session
    const plots = document.getElementById('plots');
    if (order.length === 0) {
        plots.innerHTML = '<div class="placeholder">No sessions</div>';
        return;
    }
    const [id, subject, headModel] = rows[order[selected]];
    const path = file => file.replace('{id}', id).replace('{subject}', subject)
        .replace('{head_model}', headModel);
    plots.innerHTML = `<h2>${id}</h2>` + DATA.tabs[tabs[tab]].map(file =>
        `<img src="${path(file)}" onerror="missing(this)">`
    ).join('') + `<p class="caption">${DATA.captions[tabs[tab]] || ''}</p>`
        + '<div class="placeholder" id="placeholder" hidden>Not available</div>';
}

function missing(img) {
    img.remove();
    if (!document.querySelector('#plots img')) {
        document.getElementById('placeholder').hidden = false;
    }
}

function select(i) {
    if (i < 0 || i >= order.length) return;
    selected = i;
    render();
}

function showTab(t) {
    tab = (t + tabs.length) % tabs.length;
    render();
}

document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
    if (e.key === 'ArrowDown') select(selected + 1);
    else if (e.key === 'ArrowUp') select(selected - 1);
    else if (e.key === 'ArrowRight') showTab(tab + 1);
    else if (e.key === 'ArrowLeft') showTab(tab - 1);
    else return;
    e.preventDefault();
});

update();
"""


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, or return None if it has not been saved."""
    try:
        with open(path) as file:
            return json.load(file)
    except (OSError, ValueError):
        return None


def _session_row(
    id: str,
    info: dict | None,
    qc_dir: Path,
    output_dir: Path | None,
    registrations: dict,
) -> dict:
    """Get the report's row for a session: the numbers its steps saved.

    Parameters
    ----------
    id : str
        Session ID.
    info : dict
        Session info. The surfaces are looked up with its 'subject'.
    qc_dir : Path
        Path to the QC directory.
    output_dir : Path
        Path to the derivatives directory.
    registrations : dict
        MNI registration quality of the subjects read so far, which is
        shared by all the sessions of a subject.

    Returns
    -------
    row : dict
        Session ID, subject, the ID owning the coregistration and the metrics
        that have been saved.
    """
    subject = info.get("subject") if isinstance(info, dict) else None
    row = {"Session": id, "subject": subject, "head_model": id}

    summary = _read_json(qc_dir / "preproc" / id / "summary.json")
    if summary is not None:
        row["Bad segments (%)"] = summary["bad_percent"]
        row["Bad channels"] = summary["n_bad_channels"]
        if "ica_n_excluded" in summary:
            row["ICA excluded"] = summary["ica_n_excluded"]

    if output_dir is None:
        return row

    if subject is not None:
        if subject not in registrations:
            registrations[subject] = _read_json(
                output_dir / "anat_surfaces" / subject / "mni_registration.json"
            )
        registration = registrations[subject]
        if registration is not None:
            used = registration["used"]
            row["MNI registration"] = used
            row["MNI overlap"] = registration[used]["dice"]
            row["MNI MI"] = registration[used]["mutual_information"]

        # The coregistration may be a property of the subject rather than the
        # session, and so shared by all of a subject's sessions (see
        # head_model_id in Session)
        if not (output_dir / "osl" / id / "coreg").is_dir():
            row["head_model"] = subject

    coreg = _read_json(output_dir / "osl" / row["head_model"] / "coreg" / "coreg.json")
    if coreg is not None:
        row["Coreg rms (mm)"] = coreg["rms"]

    return row


def _copy_plots(table: pd.DataFrame, qc_dir: Path, output_dir: Path) -> None:
    """Copy the plots saved in the derivatives directory to the QC directory.

    Surface extraction and coregistration save their plots with their output
    (the preprocessing and parcellation plots are saved in the QC directory).
    The surfaces are copied once per subject and the coregistration once per
    head model, to the paths in TABS. They are saved as WebP files, which are
    several times smaller than the PNG files the steps save and look the same.
    Plots that have not changed since they were last copied are skipped.

    Parameters
    ----------
    table : pd.DataFrame
        Session ID, subject and head model of each session.
    qc_dir : Path
        Path to the QC directory.
    output_dir : Path
        Path to the derivatives directory.
    """
    copies = []
    for subject in table["subject"].dropna().unique():
        for name in ["surfaces", "mni_registration"]:
            source = output_dir / "anat_surfaces" / subject / f"{name}.png"
            destination = qc_dir / "surfaces" / subject / f"{name}.webp"
            copies.append((source, destination))
    for head_model in table["head_model"].unique():
        source = output_dir / "osl" / head_model / "coreg" / "coreg.png"
        copies.append((source, qc_dir / "coreg" / head_model / "coreg.webp"))

    def copy(files):
        source, destination = files
        if not source.exists():
            return
        if (
            destination.exists()
            and destination.stat().st_mtime >= source.stat().st_mtime
        ):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Moved into place when complete, so an interrupted copy is redone
        temporary = destination.with_suffix(".tmp")
        Image.open(source).save(temporary, "WEBP", quality=80)
        temporary.replace(destination)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(copy, copies))


def generate_report(
    qc_dir: str | Path,
    sessions: dict,
    output_dir: str | Path | None = None,
    metrics: pd.DataFrame | None = None,
    output_file: str = "report.html",
) -> None:
    """Generate a QC summary HTML report.

    Builds a table with one row per session from the QC files the pipeline
    steps have saved so far, shown next to the QC plots of the selected
    session. The table can be sorted by each metric and filtered, and values
    far from the rest of their column are highlighted. Sessions are navigated
    with the up/down arrows and the plots with the left/right arrows.

    Parameters
    ----------
    qc_dir : str or Path
        Path to the QC directory. The preprocessing and parcellation QC is
        read from here and the report is written here.
    sessions : dict
        Dictionary of sessions (same format as the pipeline scripts).
    output_dir : str or Path, optional
        Path to the derivatives directory. If provided, the surface
        extraction and coregistration QC is read from here and their plots are
        copied to the QC directory.
    metrics : pd.DataFrame, optional
        More columns for the table (e.g. statistics of the parcellated data),
        indexed by session ID. Numbers and booleans are highlighted like the
        other metrics.
    output_file : str, optional
        Filename for the report. Written to qc_dir/output_file.
    """
    qc_dir = Path(qc_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)

    # One row per session, keeping the metrics that at least one session has.
    # The QC files are read in threads, a large dataset has tens of thousands
    registrations = {}
    columns = [
        "Bad segments (%)",
        "Bad channels",
        "ICA excluded",
        "MNI registration",
        "MNI overlap",
        "MNI MI",
        "Coreg rms (mm)",
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = pool.map(
            lambda item: _session_row(*item, qc_dir, output_dir, registrations),
            sessions.items(),
        )
        table = pd.DataFrame(
            rows, columns=["Session", "subject", "head_model"] + columns
        )
    table = table.drop(columns=[c for c in columns if table[c].isna().all()])
    if output_dir is not None:
        _copy_plots(table, qc_dir, output_dir)
    if metrics is not None:
        table = table.join(metrics, on="Session")

    data = json.loads(table.to_json(orient="split", double_precision=6))
    data = {
        "columns": data["columns"],
        "rows": data["data"],
        "tabs": TABS,
        "captions": CAPTIONS,
    }
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>QC Report</title>
<style>{CSS}</style>
</head>
<body>
<header>
<h1>QC Report</h1>
<input type="text" id="filter" placeholder="Filter sessions..." oninput="update()">
<label><input type="checkbox" id="only-flagged" onchange="update()"> highlighted only</label>
<span id="count"></span>
<span class="info">&#8593;&#8595; sessions &nbsp; &#8592;&#8594; plots &nbsp; | &nbsp; Generated: {timestamp}</span>
</header>
<main>
<div id="table-pane">
<div id="table-scroll"><table id="table"></table></div>
<div id="pager">
<button onclick="select((Math.floor(selected / pageSize) - 1) * pageSize)">&#9664;</button>
<span id="page"></span>
<button onclick="select((Math.floor(selected / pageSize) + 1) * pageSize)">&#9654;</button>
</div>
</div>
<div id="plot-pane">
<div id="tabs"></div>
<div id="plots"></div>
</div>
</main>
<script>const DATA = {json.dumps(data).replace("</", "<\\/")};{JS}</script>
</body>
</html>"""

    output_path = qc_dir / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report saved: {output_path}")
