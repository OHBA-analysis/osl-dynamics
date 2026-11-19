"""Make the figures for the model pages of the User Guide.

The figures are schematics and simulations that illustrate the models, they
are not results from real data. Run from the repository root::

    python doc/user_guide/models/make_figures.py

This writes SVGs to doc/user_guide/models/images. Pass ``--format pdf`` to
write PDFs instead (e.g. for a paper or talk) and ``--only`` to make some of
the figures, e.g. ``--only hmm-graphical-model pipeline``. ``--output-dir``
writes the figures somewhere else.
"""

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath

from osl_dynamics.analysis import post_hoc, spectral
from osl_dynamics.simulation import OscillatoryMVN

OUTPUT_DIR = Path(__file__).resolve().parent / "images"

# -- Style ---------------------------------------------------------------------

# Categorical colors for states/modes. Three hues that stay distinct with
# color vision deficiencies, so every figure uses three states/modes.
STATES = ["#2a78d6", "#eb6834", "#1baf7a"]

INK = "#1f1f1f"
MUTED = "#6b6b6b"
FAINT = "#b5b5b5"
PANEL = "#f3f3f1"
OBSERVED = "#d9d9d9"

SEQUENTIAL = LinearSegmentedColormap.from_list("sequential", ["#ffffff", "#3d3d3d"])
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging", ["#3f6fae", "#f4f4f4", "#c2473f"]
)

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "text.color": INK,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "cmsy10",
    "mathtext.fallback": "stixsans",
    "svg.fonttype": "path",
    "svg.hashsalt": "osl-dynamics",
    "savefig.dpi": 200,
}

RNG_SEED = 42


def save(fig, name, fmt, smooth_images=False):
    """Save a figure to the output directory.

    Pass smooth_images=True to let the viewer smooth images embedded at their
    own resolution (:code:`interpolation="none"`) in an SVG.
    """
    path = OUTPUT_DIR / f"{name}.{fmt}"
    metadata = {"Date": None} if fmt in ["svg", "pdf"] else None
    fig.savefig(
        path, bbox_inches="tight", pad_inches=0.03, facecolor="white", metadata=metadata
    )
    plt.close(fig)
    if fmt == "svg":
        compact_svg(path, smooth_images)
    print(f"Saved {path}")


NUMBER = re.compile(r"-?\d+\.\d+")


def _round_number(match):
    """Round a number to 2 decimal places and remove trailing zeros."""
    value = f"{float(match.group()):.2f}".rstrip("0").rstrip(".")
    return "0" if value == "-0" else value


def _format(value):
    """Format a number with at most 2 decimal places."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ["-0", ""] else text


def _compact_path(d):
    """Write path data with relative coordinates rounded to 0.01 pt."""
    tokens = d.replace(",", " ").split()
    n_args = {"M": 2, "L": 2, "Q": 4, "C": 6, "Z": 0}
    out = []
    current = start = (0.0, 0.0)
    previous = None
    i = 0
    while i < len(tokens):
        command = tokens[i].upper()
        i += 1
        if command not in n_args:
            # Unexpected path data, leave it unchanged
            return d
        if command == "Z":
            out.append("z")
            current = start
            previous = "z"
            continue
        args = [round(float(x), 2) for x in tokens[i : i + n_args[command]]]
        i += n_args[command]
        if not out:
            # The first move is absolute
            out.append("M" + " ".join(_format(x) for x in args))
            current = start = (args[0], args[1])
            previous = "m"
            continue
        deltas = [x - current[j % 2] for j, x in enumerate(args)]
        letter = command.lower()
        text = " ".join(_format(x) for x in deltas)
        if letter == previous and letter in "lcq":
            # A repeated command is implicit
            out.append(" " + text if not text.startswith("-") else text)
        else:
            out.append(letter + text)
        previous = letter
        current = (args[-2], args[-1])
        if command == "M":
            start = current
    return "".join(out)


def compact_svg(path, smooth_images=False):
    """Make an SVG smaller: round path coordinates to 0.01 pt and remove metadata and whitespace."""
    text = path.read_text()
    if smooth_images:
        text = re.sub(r"image-rendering:[a-z-]+;?", "", text)
    text = re.sub(r"<metadata>.*?</metadata>", "", text, flags=re.S)
    text = re.sub(r' d="([^"]*)"', lambda m: f' d="{_compact_path(m.group(1))}"', text)
    text = re.sub(r">\s+<", "><", text)
    path.write_text(text)


# -- Drawing helpers -----------------------------------------------------------


def canvas(fig, rect, xlim, ylim):
    """Axes for drawing in data units, with equal aspect and no axis."""
    ax = fig.add_axes(rect)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def node(ax, xy, label, observed=False, r=0.3, fontsize=12):
    """Circle for a random variable in a graphical model."""
    ax.add_patch(
        Circle(
            xy,
            r,
            facecolor=OBSERVED if observed else "white",
            edgecolor=INK,
            lw=0.9,
            zorder=3,
        )
    )
    ax.text(*xy, label, ha="center", va="center", fontsize=fontsize, zorder=4)


def parameter(ax, xy, label, label_offset=(0.12, 0), ha="left", r=0.06):
    """Small filled dot for a parameter learned as a point estimate."""
    ax.add_patch(Circle(xy, r, facecolor=INK, edgecolor="none", zorder=3))
    ax.text(
        xy[0] + label_offset[0],
        xy[1] + label_offset[1],
        label,
        ha=ha,
        va="center",
        fontsize=11,
    )


def arrow(ax, start, end, color=INK, lw=0.9, style="-|>", ls="-", **kwargs):
    """Straight arrow between two points."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=8,
            lw=lw,
            color=color,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
            **kwargs,
        )
    )


def link(ax, a, ra, b, rb, **kwargs):
    """Arrow from the edge of a circle at a (radius ra) to the edge of b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = (b - a) / np.linalg.norm(b - a)
    arrow(ax, a + ra * u, b - rb * u, **kwargs)


def elbow(ax, points, color=INK, lw=0.9):
    """Arrow along a path of straight segments."""
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    ax.add_patch(
        FancyArrowPatch(
            path=MplPath(points, codes),
            arrowstyle="-|>",
            mutation_scale=8,
            lw=lw,
            color=color,
            zorder=2,
        )
    )


def box(ax, xy, w, h, fc=PANEL, ec="none", lw=0.8, zorder=1):
    """Rectangle with its lower-left corner at xy."""
    ax.add_patch(
        Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder)
    )


def panel_label(ax, letter, x=-0.02, y=1.02):
    """Bold panel letter at the top left of an axes."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def state_band(ax, stc, t, alpha=1.0):
    """Draw a state time course as a band of colored segments."""
    states = stc.argmax(axis=1)
    change = np.flatnonzero(np.diff(states)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(states)]])
    dt = t[1] - t[0]
    for s, e in zip(starts, ends):
        ax.axvspan(
            t[s],
            t[e - 1] + dt,
            color=STATES[states[s]],
            alpha=alpha,
            lw=0,
        )


def stacked(ax, t, alpha, gap=0.6, step=False, max_points=250):
    """Stacked area plot of probabilities, with thin white lines between layers.

    Pass step=True for a one-hot state time course. Otherwise, at most
    max_points time points are drawn.
    """
    t = np.asarray(t)
    alpha = np.asarray(alpha)
    xlim = (t[0], t[-1])
    if step:
        # First time point of each state visit, and the last time point
        change = np.flatnonzero(np.any(np.diff(alpha, axis=0) != 0, axis=1)) + 1
        keep = np.unique(np.concatenate([[0], change, [len(t) - 1]]))
    else:
        keep = np.unique(np.linspace(0, len(t) - 1, min(len(t), max_points)).round().astype(int))
    t, alpha = t[keep], alpha[keep]
    tops = np.cumsum(alpha, axis=1)

    # Each layer is filled from zero, starting with the top layer
    for k in reversed(range(alpha.shape[1])):
        ax.fill_between(t, 0, tops[:, k], color=STATES[k], lw=0, step="post" if step else None)
    for k in range(alpha.shape[1] - 1):
        ax.plot(t, tops[:, k], color="white", lw=gap, drawstyle="steps-post" if step else "default")
    ax.set_ylim(0, 1)
    ax.set_xlim(*xlim)


# -- Simulations ---------------------------------------------------------------


def markov_chain(trans_prob, n_samples, rng, initial_state=0):
    """Sample a state sequence and return it as a one-hot array."""
    n_states = trans_prob.shape[0]
    states = np.empty(n_samples, dtype=int)
    states[0] = initial_state
    for i in range(1, n_samples):
        states[i] = rng.choice(n_states, p=trans_prob[states[i - 1]])
    return np.eye(n_states, dtype=int)[states]


def smooth_noise(n_samples, n_series, rng, width):
    """Gaussian noise smoothed with a Gaussian kernel (width in samples)."""
    kernel = np.exp(-0.5 * (np.arange(-4 * width, 4 * width + 1) / width) ** 2)
    kernel /= kernel.sum()
    x = rng.standard_normal((n_samples + len(kernel), n_series))
    x = np.array([np.convolve(x[:, i], kernel, mode="same") for i in range(n_series)]).T
    x = x[len(kernel) // 2 : len(kernel) // 2 + n_samples]
    return x / x.std(axis=0)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def oscillations(weights, frequencies, amplitudes, peak_widths, fs, seed):
    """One channel with an oscillation for each column of weights.

    The power of each oscillation is proportional to its weight (see
    OscillatoryMVN).
    """
    np.random.seed(seed)
    obs_model = OscillatoryMVN(
        frequencies=frequencies,
        sampling_frequency=fs,
        covariances=np.square(amplitudes)[:, None, None],
        peak_width=peak_widths,
    )
    return obs_model.simulate_data(weights)[:, 0].astype(np.float64)


def example_covariances():
    """Covariances of three states/modes for five channels."""
    corr = [np.eye(5) for _ in range(3)]
    corr[0][0, 1] = corr[0][1, 0] = 0.8
    corr[1][2, 3] = corr[1][3, 2] = -0.7
    corr[2][3, 4] = corr[2][4, 3] = 0.6
    corr[2][2, 4] = corr[2][4, 2] = 0.5
    stds = np.array(
        [
            [2.0, 2.0, 0.6, 0.6, 0.6],
            [0.6, 0.6, 1.8, 1.8, 0.6],
            [0.6, 0.6, 1.4, 1.4, 2.0],
        ]
    )
    return np.array([np.outer(std, std) * c for std, c in zip(stds, corr)])


def covariance_panel(ax, cov, limit, color, label):
    """Heatmap of a covariance matrix with a colored frame."""
    im = ax.imshow(cov, cmap=DIVERGING, vmin=-limit, vmax=limit)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(2)
    ax.set_xlabel(label, labelpad=3)
    return im


# -- Models overview -----------------------------------------------------------


def pipeline(fmt):
    """Steps of a typical analysis, from recordings to statistics."""
    rng = np.random.default_rng(RNG_SEED)
    W, H = 7, 5.0
    fig = plt.figure(figsize=(W, H))
    overlay = fig.add_axes([0, 0, 1, 1])
    overlay.set_xlim(0, W)
    overlay.set_ylim(0, H)
    overlay.axis("off")

    steps = [
        ("Record", "M/EEG or fMRI recordings"),
        ("Parcellate", "Source reconstruction and\nparcellation (M/EEG only)"),
        ("Prepare data", "TDE and PCA (M/EEG),\nstandardization"),
        ("Train a model", "HMM, DyNeMo, DyNeStE,\nM-DyNeMo, HIVE"),
        ("Post-hoc analysis", "Summary statistics,\nspectra, networks"),
        ("Statistics", "Group and condition\ncomparisons"),
    ]
    w, h = 2.0, 2.0
    x0 = [0.05, 2.5, 4.95]
    y0 = [2.6, 0.05]
    corners = [(x0[i % 3], y0[i // 3]) for i in range(6)]

    for i, ((title, caption), (x, y)) in enumerate(zip(steps, corners)):
        box(overlay, (x, y), w, h)
        overlay.add_patch(Circle((x + 0.24, y + h - 0.24), 0.14, color=INK, zorder=3))
        overlay.text(
            x + 0.24,
            y + h - 0.24,
            str(i + 1),
            color="white",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
        )
        overlay.text(
            x + 0.47,
            y + h - 0.24,
            title,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )
        overlay.text(
            x + w / 2,
            y + 0.13,
            caption,
            fontsize=10,
            color=MUTED,
            ha="center",
            va="bottom",
            linespacing=1.35,
        )

    # Arrows between the steps. In the first row the arrows are the M/EEG
    # path, at the height of the M/EEG data in step 1.
    meeg_y = corners[0][1] + 0.84
    for i in [0, 1]:
        x, _ = corners[i]
        arrow(overlay, (x + w + 0.04, meeg_y), (x + w + 0.41, meeg_y))
        overlay.text(
            x + w + 0.225,
            meeg_y + 0.06,
            "M/EEG",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    for i in [3, 4]:
        x, y = corners[i]
        arrow(overlay, (x + w + 0.04, y + h / 2), (x + w + 0.41, y + h / 2))
    x3, y3 = corners[2]
    x4, y4 = corners[3]
    middle = (y3 + y4 + h) / 2
    elbow(
        overlay,
        [
            (x3 + w / 2, y3 - 0.03),
            (x3 + w / 2, middle),
            (x4 + w / 2, middle),
            (x4 + w / 2, y4 + h + 0.03),
        ],
    )

    # fMRI data skip step 2
    x1, y1 = corners[0]
    top = y1 + h + 0.2
    elbow(
        overlay,
        [
            (x1 + w / 2, y1 + h + 0.03),
            (x1 + w / 2, top),
            (x3 + w / 2, top),
            (x3 + w / 2, y3 + h + 0.03),
        ],
        color=MUTED,
        lw=0.8,
    )
    overlay.patches[-1].set_linestyle((0, (3, 2)))
    overlay.text(
        (x1 + x3 + w) / 2,
        top + 0.04,
        "fMRI (already parcellated)",
        color=MUTED,
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    def mini(i, left=0.4, right=0.18, bottom=0.62, top=0.5):
        """Axes inside the box of step i, in figure coordinates."""
        x, y = corners[i]
        rect = [
            (x + left) / W,
            (y + bottom) / H,
            (w - left - right) / W,
            (h - bottom - top) / H,
        ]
        ax = fig.add_axes(rect)
        ax.axis("off")
        return ax

    def side_label(ax, text):
        """Label on the left of a mini plot."""
        ax.text(
            -0.05,
            0.5,
            text,
            transform=ax.transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=9,
            color=MUTED,
        )

    t = np.linspace(0, 1, 300)

    # 1. Recordings: fMRI (slow) above, M/EEG sensors (fast) below
    ax = mini(0, bottom=1.12, top=0.5)
    x = smooth_noise(300, 4, rng, 18)
    for c in range(4):
        ax.plot(t, x[:, c] * 0.25 + c, color=MUTED, lw=0.7)
    ax.set_xlim(0, 1)
    side_label(ax, "fMRI")
    ax = mini(0, bottom=0.62, top=0.98)
    x = smooth_noise(300, 8, rng, 1.5)
    for c in range(8):
        ax.plot(t, x[:, c] * 0.2 + c, color=MUTED, lw=0.4)
    ax.set_xlim(0, 1)
    side_label(ax, "M/EEG")

    # 2. Parcel time courses
    ax = mini(1)
    x = smooth_noise(300, 5, rng, 3)
    for c in range(5):
        ax.plot(t, x[:, c] * 0.25 + c, color=INK, lw=0.6)
    ax.set_xlim(0, 1)
    side_label(ax, "Parcels")

    # 3. Time-delay embedding: a parcel and lagged copies of it
    ax = mini(2)
    x = smooth_noise(400, 1, rng, 6)[:, 0]
    n_lags = 5
    for j in range(n_lags):
        shade = 0.3 + 0.7 * (1 - j / (n_lags - 1))
        ax.plot(t, x[j * 15 : j * 15 + 300] * 0.25 - j, color=INK, alpha=shade, lw=0.6)
    ax.set_xlim(0, 1)
    side_label(ax, "Lagged copies")

    # 4. Inferred state probabilities
    ax = mini(3, bottom=0.68, top=0.55)
    trans_prob = np.full((3, 3), 0.015) + np.eye(3) * 0.955
    stc = markov_chain(trans_prob, 300, rng)
    alpha = 0.9 * stc + 0.1 * softmax(2 * smooth_noise(300, 3, rng, 6))
    stacked(ax, t, alpha)
    side_label(ax, "States")

    # 5. Fractional occupancies and state spectra
    x5, y5 = corners[4]
    ax = mini(4, left=0.25, right=1.4, bottom=0.72, top=0.55)
    for k, fo in enumerate([0.45, 0.2, 0.35]):
        ax.bar(k, fo, color=STATES[k], width=0.7)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 0.5)
    ax.text(0.5, -0.06, "Occupancy", transform=ax.transAxes, ha="center", va="top", fontsize=9, color=MUTED)
    ax = mini(4, left=0.95, right=0.15, bottom=0.72, top=0.55)
    f = np.linspace(1, 40, 200)
    background = 0.4 / f
    for k, (peak, height) in enumerate([(10, 0.5), (20, 0.3), (5, 0.4)]):
        psd = background + height * np.exp(-0.5 * ((f - peak) / 1.8) ** 2)
        ax.plot(f, psd, color=STATES[k], lw=1.2)
    ax.set_xlim(1, 40)
    ax.set_ylim(0, 0.65)
    ax.text(0.5, -0.06, "Spectra", transform=ax.transAxes, ha="center", va="top", fontsize=9, color=MUTED)

    # 6. Group comparison
    ax = mini(5, left=0.45, right=0.45, bottom=0.72, top=0.5)
    for g, (mean, color) in enumerate([(0.0, MUTED), (1.1, INK)]):
        values = rng.normal(mean, 0.45, 14)
        ax.scatter(g + rng.uniform(-0.15, 0.15, 14), values, s=6, color=color, lw=0)
        ax.plot([g - 0.25, g + 0.25], [values.mean()] * 2, color=INK, lw=1.0)
        ax.text(g, -1.45, f"Group {'AB'[g]}", ha="center", va="top", fontsize=9, color=MUTED)
    ax.plot([0, 0, 1, 1], [2.1, 2.3, 2.3, 2.1], color=INK, lw=0.6)
    ax.text(0.5, 2.38, "$p < 0.05$", ha="center", va="bottom", fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1.3, 2.9)

    save(fig, "pipeline", fmt)


def states_vs_modes(fmt):
    """A state time course and a mode time course side by side."""
    rng = np.random.default_rng(RNG_SEED + 1)
    fs = 100
    n_samples = 3 * fs
    t = np.arange(n_samples) / fs

    trans_prob = np.full((3, 3), 0.02) + np.eye(3) * 0.94
    stc = markov_chain(trans_prob, n_samples, rng)
    modes = softmax(2.5 * smooth_noise(n_samples, 3, rng, 8))

    rows = [
        (stc, "States", "HMM, DyNeStE, HIVE", r"$\alpha_{kt}$", "State"),
        (modes, "Modes", "DyNeMo, M-DyNeMo", r"$\alpha_{jt}$", "Mode"),
    ]
    fig, axes = plt.subplots(
        len(rows),
        1,
        figsize=(7, 2.3),
        sharex=True,
        gridspec_kw={"hspace": 0.5, "left": 0.2, "right": 0.86},
    )
    for ax, (alpha, title, models, ylabel, name) in zip(axes, rows):
        stacked(ax, t, alpha, step=name == "State")
        ax.set_yticks([0, 1])
        ax.set_ylabel(ylabel, rotation=0, ha="right", va="center", labelpad=4)
        ax.text(
            -0.115,
            0.62,
            title,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontweight="bold",
        )
        ax.text(
            -0.115,
            0.3,
            models,
            transform=ax.transAxes,
            ha="right",
            va="center",
            color=MUTED,
        )
        ax.spines["left"].set_bounds(0, 1)
        handles = [
            mpl.patches.Patch(color=STATES[k], label=f"{name} {k + 1}") for k in range(3)
        ]
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            handlelength=1,
            handleheight=1,
        )
    axes[-1].set_xlabel("Time (s)")

    save(fig, "states-vs-modes", fmt)


# -- HMM -----------------------------------------------------------------------


def hmm_graphical_model(fmt):
    """Graphical model of the HMM, with its parameters."""
    fig = plt.figure(figsize=(6.2, 2.1))
    ax = canvas(fig, [0, 0, 1, 1], (0, 8.85), (0, 3))
    r = 0.3
    ys, yx = 1.95, 0.85
    xs = [1.5, 2.8, 4.1, 6.2]
    labels = ["1", "2", "3", "T"]

    for x, label in zip(xs, labels):
        node(ax, (x, ys), f"$s_{label}$", r=r)
        node(ax, (x, yx), f"$x_{label}$", observed=True, r=r)
        link(ax, (x, ys), r, (x, yx), r)
    for a, b in zip(xs[:2], xs[1:3]):
        link(ax, (a, ys), r, (b, ys), r)
    ax.text(5.15, ys, r"$\cdots$", ha="center", va="center", fontsize=13)
    arrow(ax, (xs[2] + r, ys), (4.85, ys))
    arrow(ax, (5.45, ys), (xs[3] - r, ys))

    # Parameters
    parameter(ax, (0.45, ys), r"$\pi$", label_offset=(0, 0.22), ha="center")
    link(ax, (0.45, ys), 0.06, (xs[0], ys), r, color=MUTED, lw=0.7)
    parameter(ax, (3.45, 2.85), "$A$", label_offset=(0.12, 0))
    for x in xs[1:]:
        link(ax, (3.45, 2.85), 0.06, (x, ys), r, color=MUTED, lw=0.7)
    parameter(ax, (3.85, 0.12), r"$m_k, C_k$", label_offset=(0.12, 0))
    for x in xs:
        link(ax, (3.85, 0.12), 0.06, (x, yx), r, color=MUTED, lw=0.7)

    # Legend
    lx = 7.0
    node(ax, (lx, 2.45), "", r=0.13)
    ax.text(lx + 0.25, 2.45, "Hidden state", va="center", fontsize=8.5)
    node(ax, (lx, 1.95), "", observed=True, r=0.13)
    ax.text(lx + 0.25, 1.95, "Observed data", va="center", fontsize=8.5)
    ax.add_patch(Circle((lx, 1.45), 0.06, color=INK))
    ax.text(lx + 0.25, 1.45, "Parameter", va="center", fontsize=8.5)

    save(fig, "hmm-graphical-model", fmt)


def hmm_generative_process(fmt):
    """Sampling data from an HMM: transition matrix, covariances, states, data."""
    rng = np.random.default_rng(RNG_SEED + 2)
    fs = 100
    n_samples = 4 * fs
    n_channels = 5
    t = np.arange(n_samples) / fs

    trans_prob = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.02, 0.95, 0.03],
            [0.03, 0.02, 0.95],
        ]
    )

    covs = example_covariances()

    stc = markov_chain(trans_prob, n_samples, rng)
    states = stc.argmax(axis=1)
    chol = np.linalg.cholesky(covs)
    x = np.einsum("tij,tj->ti", chol[states], rng.standard_normal((n_samples, n_channels)))

    fig = plt.figure(figsize=(7, 3.7))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.45], hspace=0.42, top=0.92, bottom=0.1
    )
    top = outer[0].subgridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.45)
    bottom = outer[1].subgridspec(2, 1, height_ratios=[0.11, 1.25], hspace=0.45)
    fig.subplots_adjust(left=0.07, right=0.98)

    # Transition probability matrix
    ax = fig.add_subplot(top[0, 0])
    ax.imshow(trans_prob, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{trans_prob[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if trans_prob[i, j] > 0.5 else INK,
            )
    ax.set_xticks(range(3), ["1", "2", "3"])
    ax.set_yticks(range(3), ["1", "2", "3"])
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Transition probabilities, $A$", loc="left", pad=6)
    panel_label(ax, "a", x=-0.38, y=1.06)

    # State covariances
    limit = np.abs(covs).max()
    for k in range(3):
        ax = fig.add_subplot(top[0, k + 1])
        im = ax.imshow(covs[k], cmap=DIVERGING, vmin=-limit, vmax=limit)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(STATES[k])
            spine.set_linewidth(2)
        ax.set_xlabel(f"$C_{k + 1}$", labelpad=3)
        if k == 0:
            ax.set_title("State covariances, $C_k$", loc="left", pad=6)
            panel_label(ax, "b", x=-0.12, y=1.06)
    cax = fig.add_axes([0.985, 0.6, 0.008, 0.25])
    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7.5, length=2)

    # State time course
    ax = fig.add_subplot(bottom[0])
    state_band(ax, stc, t)
    ax.set_xlim(t[0], t[-1] + 1 / fs)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        r"State time course, $s_t \sim p(s_t \mid s_{t-1})$",
        loc="left",
        pad=4,
    )
    panel_label(ax, "c", x=-0.04, y=1.15)

    # Generated data
    ax = fig.add_subplot(bottom[1])
    state_band(ax, stc, t, alpha=0.12)
    offsets = np.arange(n_channels)[::-1] * 6
    for c in range(n_channels):
        ax.plot(t, x[:, c] + offsets[c], color=INK, lw=0.5)
    ax.set_yticks(offsets, [f"{c + 1}" for c in range(n_channels)])
    ax.set_ylabel("Channel")
    ax.set_xlim(t[0], t[-1] + 1 / fs)
    ax.set_xlabel("Time (s)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(
        r"Generated data, $x_t \sim \mathcal{N}(m_{s_t}, C_{s_t})$",
        loc="left",
        pad=4,
    )
    panel_label(ax, "d", x=-0.04, y=1.02)

    save(fig, "hmm-generative-process", fmt)


def hmm_training(fmt):
    """One training step of the HMM in osl-dynamics."""
    rng = np.random.default_rng(RNG_SEED + 3)
    fig = plt.figure(figsize=(7, 2.6))
    W, H = 7, 2.6
    ax = canvas(fig, [0, 0, 1, 1], (0, W), (0, H))

    def titled_box(x, y, w, h, title, body, fc=PANEL):
        box(ax, (x, y), w, h, fc=fc)
        ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontweight="bold")
        ax.text(
            x + w / 2,
            y + h - 0.38,
            body,
            ha="center",
            va="top",
            fontsize=9,
            linespacing=1.5,
        )

    # Batch of data
    titled_box(0.05, 0.75, 1.3, 1.25, "Batch", "Sequences of data,\n$x_{1:L}$")
    sub = fig.add_axes([0.15 / W, 0.85 / H, 1.1 / W, 0.42 / H])
    sub.axis("off")
    x = smooth_noise(100, 4, rng, 1.5)
    for c in range(4):
        sub.plot(x[:, c] * 0.2 + c, color=MUTED, lw=0.4)

    # E-step
    titled_box(
        1.75,
        0.75,
        1.55,
        1.25,
        "E-step",
        "Baum-Welch\n(forward-backward)\ngiven the current\n" r"$\pi$, $A$, $m_k$, $C_k$",
    )

    # Posterior
    titled_box(
        3.7,
        0.75,
        1.25,
        1.25,
        "Posterior",
        "$q(s_t)$, $q(s_t, s_{t+1})$",
    )
    sub = fig.add_axes([3.8 / W, 0.87 / H, 1.05 / W, 0.42 / H])
    sub.axis("off")
    trans_prob = np.full((3, 3), 0.03) + np.eye(3) * 0.91
    stc = markov_chain(trans_prob, 100, rng)
    alpha = 0.8 * stc + 0.2 * softmax(2 * smooth_noise(100, 3, rng, 2))
    stacked(sub, np.arange(100), alpha, gap=0.4)

    # Updates
    titled_box(
        5.35,
        1.45,
        1.6,
        1.05,
        "Observation model",
        "Gradient step on the\nlog-likelihood of the data,\nweighted by $q(s_t)$",
    )
    titled_box(
        5.35,
        0.25,
        1.6,
        1.05,
        "Markov chain",
        r"$A \leftarrow (1 - \rho) A + \rho \hat{A}$"
        "\n"
        r"$\pi \leftarrow (1 - \rho) \pi + \rho \hat{\pi}$"
        "\n"
        r"$\hat{A}$, $\hat{\pi}$ from the posterior",
    )

    # Arrows
    arrow(ax, (1.38, 1.375), (1.72, 1.375))
    arrow(ax, (3.33, 1.375), (3.67, 1.375))
    elbow(ax, [(4.98, 1.375), (5.15, 1.375), (5.15, 1.97), (5.32, 1.97)])
    elbow(ax, [(5.15, 1.375), (5.15, 0.775), (5.32, 0.775)])

    # Loop back to the E-step for the next batch
    elbow(
        ax,
        [(6.15, 0.22), (6.15, 0.07), (2.525, 0.07), (2.525, 0.72)],
        color=MUTED,
        lw=0.7,
    )
    ax.text(
        3.8,
        0.12,
        "Next batch, with the updated parameters",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )

    save(fig, "hmm-training", fmt)


def hmm_summary_stats(fmt):
    """The summary statistics of a state time course, illustrated for one state."""
    fs = 100
    # (state, duration in samples) for each visit
    visits = [
        (0, 22), (1, 30), (2, 18), (0, 35), (2, 25), (1, 20), (0, 15),
        (1, 28), (2, 30), (0, 27), (1, 22), (2, 16), (0, 12),
    ]
    states = np.concatenate([[s] * n for s, n in visits])
    n_samples = len(states)
    stc = np.eye(3, dtype=int)[states]
    t = np.arange(n_samples) / fs

    fo = post_hoc.fractional_occupancies(stc)[0]
    lt = post_hoc.mean_lifetimes(stc, fs)[0]
    intv = post_hoc.mean_intervals(stc, fs)[0]
    sr = post_hoc.switching_rates(stc, fs)[0]

    fig = plt.figure(figsize=(7, 2.3))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.22, 1],
        width_ratios=[2.6, 1.3],
        hspace=0.5,
        wspace=0.05,
        left=0.09,
        right=0.99,
        top=0.88,
        bottom=0.17,
    )

    # All states
    ax = fig.add_subplot(gs[0, 0])
    state_band(ax, stc, t)
    ax.set_xlim(0, t[-1] + 1 / fs)
    ax.axis("off")
    ax.text(-0.01, 0.5, "All states", transform=ax.transAxes, ha="right", va="center")

    # State 1 only
    ax = fig.add_subplot(gs[1, 0])
    on = (states == 0).astype(float)
    ax.fill_between(t, 0, on, step="post", color=STATES[0], lw=0)
    ax.plot(t, on, drawstyle="steps-post", color=INK, lw=0.6)
    ax.set_xlim(0, t[-1] + 1 / fs)
    ax.set_ylim(-0.75, 1.75)
    ax.set_yticks([0, 1], ["Off", "On"])
    ax.spines["left"].set_bounds(0, 1)
    ax.set_xlabel("Time (s)")
    ax.text(-0.01, 1.08, "State 1", transform=ax.transAxes, ha="right", va="bottom")

    # Brackets for the lifetimes (above) and intervals (below)
    change = np.flatnonzero(np.diff(on)) + 1
    edges = np.concatenate([[0], change, [n_samples]]) / fs
    for start, end in zip(edges[:-1], edges[1:]):
        active = on[int(round(start * fs))] == 1
        y = 1.22 if active else -0.25
        ax.annotate(
            "",
            (start + 0.005, y),
            (end - 0.005, y),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=0.6, shrinkA=0, shrinkB=0),
        )
    ax.text(
        (edges[0] + edges[1]) / 2,
        1.32,
        "Lifetime",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    ax.text(
        (edges[1] + edges[2]) / 2,
        -0.35,
        "Interval",
        ha="center",
        va="top",
        fontsize=8.5,
    )

    # Values for state 1
    ax = fig.add_subplot(gs[:, 1])
    ax.axis("off")
    rows = [
        ("Fractional occupancy", f"{fo:.2f}"),
        ("Mean lifetime", f"{lt * 1000:.0f} ms"),
        ("Mean interval", f"{intv * 1000:.0f} ms"),
        ("Switching rate", f"{sr:.1f} Hz"),
    ]
    ax.text(0.08, 0.95, "State 1", fontweight="bold", va="top", transform=ax.transAxes)
    for i, (name, value) in enumerate(rows):
        y = 0.75 - i * 0.19
        ax.text(0.08, y, name, va="top", transform=ax.transAxes, fontsize=9)
        ax.text(0.98, y, value, va="top", ha="right", transform=ax.transAxes, fontsize=9)

    save(fig, "hmm-summary-stats", fmt)


def hmm_spectra(fmt):
    """Multitaper spectra: the data weighted by each state's time course."""
    rng = np.random.default_rng(RNG_SEED + 4)
    fs = 250
    n_samples = 300 * fs
    trans_prob = np.full((3, 3), 0.0075) + np.eye(3) * 0.9775
    stc = markov_chain(trans_prob, n_samples, rng)

    # An oscillation in each state and a broad background (always on)
    x = oscillations(
        np.column_stack([stc, np.ones(n_samples)]),
        frequencies=[10, 20, 5, 1],
        amplitudes=[1.2, 1.2, 1.2, 0.6],
        peak_widths=[1.2, 1.2, 1.2, 8.4],
        fs=fs,
        seed=RNG_SEED + 4,
    )

    f, psd = spectral.multitaper_spectra(
        x[:, None],
        fs,
        alpha=stc,
        window_length=2 * fs,
        frequency_range=[1, 35],
        calc_coh=False,
    )
    _, static_psd = spectral.multitaper_spectra(
        x[:, None],
        fs,
        alpha=np.ones((n_samples, 1)),
        window_length=2 * fs,
        frequency_range=[1, 35],
        calc_coh=False,
    )
    psd = np.squeeze(psd)
    static_psd = np.squeeze(static_psd)

    fig = plt.figure(figsize=(7, 2.6))
    gs = fig.add_gridspec(
        4,
        2,
        width_ratios=[1.7, 1],
        height_ratios=[1.3, 1, 1, 1],
        hspace=0.25,
        wspace=0.3,
        left=0.09,
        right=0.98,
        top=0.9,
        bottom=0.17,
    )

    start = 61 * fs
    window = slice(start, start + int(2.5 * fs))
    t = np.arange(window.stop - window.start) / fs
    limit = np.abs(x[window]).max()

    ax = fig.add_subplot(gs[0, 0])
    state_band(ax, stc[window], t, alpha=0.15)
    ax.plot(t, x[window], color=INK, lw=0.5)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-limit, limit)
    ax.axis("off")
    ax.text(-0.01, 0.5, "$x_t$", transform=ax.transAxes, ha="right", va="center", fontsize=11)
    ax.set_title("Data before time-delay embedding", loc="left", pad=3)
    panel_label(ax, "a", x=-0.08, y=1.0)

    for k in range(3):
        ax = fig.add_subplot(gs[k + 1, 0])
        ax.plot(t, x[window] * stc[window, k], color=STATES[k], lw=0.5)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-limit, limit)
        ax.text(
            -0.01,
            0.5,
            rf"$x_t \, \alpha_{{{k + 1}t}}$",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=11,
        )
        if k < 2:
            ax.axis("off")
        else:
            ax.spines["left"].set_visible(False)
            ax.set_yticks([])
            ax.set_xlabel("Time (s)")

    ax = fig.add_subplot(gs[:, 1])
    ax.plot(f, static_psd, color=FAINT, lw=1.0, ls="--", label="Static")
    for k in range(3):
        ax.plot(f, psd[k], color=STATES[k], lw=1.3)
        i = np.argmax(psd[k])
        # Labels to the right of the peaks, except state 3 (above its peak)
        if k == 2:
            ax.text(f[i], psd[k][i] + 0.004, "State 3", fontsize=8.5, ha="left", va="bottom")
        else:
            ax.text(f[i] + 1.0, psd[k][i], f"State {k + 1}", fontsize=8.5, va="center")
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(0, 1.2 * psd.max())
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (a.u.)")
    ax.set_title("Multitaper spectra", loc="left", pad=3)
    ax.legend(loc="upper right", frameon=False, handlelength=2)
    panel_label(ax, "b", x=-0.2, y=1.0)

    save(fig, "hmm-spectra", fmt)

# -- DyNeMo --------------------------------------------------------------------


def dynemo_graphical_model(fmt):
    """Graphical model of DyNeMo: each logit depends on all previous logits."""
    fig = plt.figure(figsize=(6.2, 2.33))
    ax = canvas(fig, [0, 0, 1, 1], (0, 8.85), (0, 3.3))
    r = 0.3
    ys, yx = 1.95, 0.85
    xs = [1.5, 2.8, 4.1, 6.2]
    labels = ["1", "2", "3", "T"]

    for x, label in zip(xs, labels):
        node(ax, (x, ys), rf"$\theta_{label}$", r=r)
        node(ax, (x, yx), f"$x_{label}$", observed=True, r=r)
        link(ax, (x, ys), r, (x, yx), r)
    for a, b in zip(xs[:2], xs[1:3]):
        link(ax, (a, ys), r, (b, ys), r)
    ax.text(5.15, ys, r"$\cdots$", ha="center", va="center", fontsize=13)
    arrow(ax, (xs[2] + r, ys), (4.85, ys))
    arrow(ax, (5.45, ys), (xs[3] - r, ys))

    # Long-range dependencies (the model RNN sees all previous logits)
    for a, b, rad in [(0, 2, -0.45), (1, 3, -0.3), (0, 3, -0.3)]:
        arrow(
            ax,
            (xs[a] + 0.1, ys + r - 0.02),
            (xs[b] - 0.1, ys + r - 0.02),
            color=MUTED,
            lw=0.7,
            connectionstyle=f"arc3,rad={rad}",
        )

    parameter(ax, (3.85, 0.12), r"$\mu_j, D_j$", label_offset=(0.12, 0))
    for x in xs:
        link(ax, (3.85, 0.12), 0.06, (x, yx), r, color=MUTED, lw=0.7)

    # Legend
    lx = 7.0
    node(ax, (lx, 2.45), "", r=0.13)
    ax.text(lx + 0.25, 2.45, "Latent logits", va="center", fontsize=8.5)
    node(ax, (lx, 1.95), "", observed=True, r=0.13)
    ax.text(lx + 0.25, 1.95, "Observed data", va="center", fontsize=8.5)
    ax.add_patch(Circle((lx, 1.45), 0.06, color=INK))
    ax.text(lx + 0.25, 1.45, "Parameter", va="center", fontsize=8.5)

    save(fig, "dynemo-graphical-model", fmt)


def dynemo_generative_process(fmt):
    """Sampling data from DyNeMo: logits, mixing coefficients and data."""
    rng = np.random.default_rng(RNG_SEED + 5)
    fs = 100
    n_samples = 4 * fs
    n_channels = 5
    t = np.arange(n_samples) / fs

    covs = example_covariances()
    theta = 2.5 * smooth_noise(n_samples, 3, rng, 10)
    alpha = softmax(theta)
    C = np.einsum("tj,jmn->tmn", alpha, covs)
    x = np.einsum(
        "tij,tj->ti", np.linalg.cholesky(C), rng.standard_normal((n_samples, n_channels))
    )

    # A time point where the modes mix
    mixed = np.flatnonzero((alpha.max(axis=1) < 0.6) & (np.sort(alpha, axis=1)[:, -2] > 0.25))
    i = mixed[np.argmin(np.abs(mixed - n_samples // 2))]

    fig = plt.figure(figsize=(7, 5.0))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 2.6], hspace=0.3, top=0.93, bottom=0.08
    )
    top = outer[0].subgridspec(1, 5, width_ratios=[1, 1, 1, 0.35, 1], wspace=0.35)
    bottom = outer[1].subgridspec(3, 1, height_ratios=[0.8, 0.8, 1.6], hspace=0.5)
    fig.subplots_adjust(left=0.08, right=0.97)

    # Mode covariances and the covariance at one time point
    limit = np.abs(covs).max()
    for j in range(3):
        ax = fig.add_subplot(top[0, j])
        covariance_panel(ax, covs[j], limit, STATES[j], f"$D_{j + 1}$")
        if j == 0:
            ax.set_title("Mode covariances, $D_j$", loc="left", pad=6)
            panel_label(ax, "a", x=-0.12, y=1.06)
    ax = fig.add_subplot(top[0, 4])
    covariance_panel(
        ax,
        C[i],
        limit,
        INK,
        rf"${alpha[i, 0]:.2f} D_1 + {alpha[i, 1]:.2f} D_2 + {alpha[i, 2]:.2f} D_3$",
    )
    ax.xaxis.label.set_fontsize(8.5)
    ax.set_title(r"Covariance at $t^*$, $C_{t^*}$", loc="left", pad=6)
    panel_label(ax, "b", x=-0.12, y=1.06)

    # Logits
    ax = fig.add_subplot(bottom[0])
    for j in range(3):
        ax.plot(t, theta[:, j], color=STATES[j], lw=1.2)
    ax.set_xlim(t[0], t[-1])
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(r"Logits, $\theta_t \sim p(\theta_t \mid \theta_{1:t-1})$", loc="left", pad=4)
    panel_label(ax, "c", x=-0.04, y=1.05)
    ax.axvline(t[i], color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.text(t[i] + 0.03, ax.get_ylim()[1], "$t^*$", ha="left", va="top")

    # Mixing coefficients
    ax = fig.add_subplot(bottom[1])
    stacked(ax, t, alpha)
    ax.set_yticks([0, 1])
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title(r"Mixing coefficients, $\alpha_t = \mathrm{softmax}(\theta_t)$", loc="left", pad=4)
    panel_label(ax, "d", x=-0.04, y=1.05)
    ax.axvline(t[i], color=INK, lw=0.8, ls=(0, (3, 2)))

    # Generated data
    ax = fig.add_subplot(bottom[2])
    offsets = np.arange(n_channels)[::-1] * 6
    for c in range(n_channels):
        ax.plot(t, x[:, c] + offsets[c], color=INK, lw=0.5)
    ax.set_yticks(offsets, [f"{c + 1}" for c in range(n_channels)])
    ax.set_ylabel("Channel")
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(
        r"Generated data, $x_t \sim \mathcal{N}(m_t, C_t)$, $C_t = {\sum}_j \alpha_{jt} D_j$",
        loc="left",
        pad=4,
    )
    panel_label(ax, "e", x=-0.04, y=1.02)
    ax.axvline(t[i], color=INK, lw=0.8, ls=(0, (3, 2)))

    save(fig, "dynemo-generative-process", fmt)


def dynemo_training(fmt):
    """Amortized variational inference in DyNeMo."""
    rng = np.random.default_rng(RNG_SEED + 6)
    W, H = 7, 3.8
    fig = plt.figure(figsize=(W, H))
    ax = canvas(fig, [0, 0, 1, 1], (0, W), (0, H))

    def titled_box(x, y, w, h, title, body, top=False):
        """Box with a title, and the body centred or (top=True) under the title."""
        box(ax, (x, y), w, h)
        ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontweight="bold")
        ax.text(
            x + w / 2,
            y + h - 0.42 if top else y + (h - 0.35) / 2,
            body,
            ha="center",
            va="top" if top else "center",
            fontsize=9,
            linespacing=1.5,
        )

    def mini(x, y, w, h):
        """Axes for an illustration inside a box, in canvas units."""
        sub = fig.add_axes([x / W, y / H, w / W, h / H])
        sub.axis("off")
        return sub

    def gaussians(sub, means, std):
        """A normal density for each mode."""
        z = np.linspace(-4, 4, 200)
        for j, mean in enumerate(means):
            density = np.exp(-0.5 * ((z - mean) / std) ** 2)
            sub.fill_between(z, density, color=STATES[j], alpha=0.25, lw=0)
            sub.plot(z, density, color=STATES[j], lw=1.0)
        sub.set_xlim(-4, 4)
        sub.set_ylim(0, 1.1)

    y1, h1 = 2.25, 1.5
    y2, h2 = 0.1, 1.3
    xs = {
        "data": (0.03, 0.95),
        "inf": (1.2, 1.25),
        "post": (2.67, 1.2),
        "alpha": (4.09, 1.1),
        "obs": (5.41, 1.56),
    }

    def centre(key):
        x, w = xs[key]
        return x + w / 2

    # Top row: from the data to the observation model
    titled_box(xs["data"][0], y1, xs["data"][1], h1, "Data", "$x_{1:L}$", top=True)
    sub = mini(xs["data"][0] + 0.1, y1 + 0.12, xs["data"][1] - 0.2, 0.75)
    x = smooth_noise(100, 4, rng, 1.5)
    for c in range(4):
        sub.plot(x[:, c] * 0.2 + c, color=MUTED, lw=0.4)
    titled_box(xs["inf"][0], y1, xs["inf"][1], h1, "Inference RNN", "Bidirectional,\nsees $x_{1:L}$")
    titled_box(xs["post"][0], y1, xs["post"][1], h1, "Posterior", r"$q(\theta_t \mid x_{1:L})$," "\nsample $\\theta_t$", top=True)
    gaussians(mini(xs["post"][0] + 0.1, y1 + 0.1, xs["post"][1] - 0.2, 0.5), [-1.8, 0.2, 1.9], 0.45)
    titled_box(xs["alpha"][0], y1, xs["alpha"][1], h1, "Softmax", r"$\alpha_t = \mathrm{softmax}(\theta_t)$", top=True)
    sub = mini(xs["alpha"][0] + 0.12, y1 + 0.15, xs["alpha"][1] - 0.24, 0.65)
    stacked(sub, np.arange(100), softmax(2.5 * smooth_noise(100, 3, rng, 6)), gap=0.4)
    sub.axis("off")
    titled_box(
        xs["obs"][0],
        y1,
        xs["obs"][1],
        h1,
        "Observation model",
        r"$m_t = {\sum}_j \alpha_{jt} \mu_j$" "\n" r"$C_t = {\sum}_j \alpha_{jt} D_j$",
        top=True,
    )
    covs = example_covariances()
    limit = np.abs(covs).max()
    size = 0.38
    gap = (xs["obs"][1] - 3 * size) / 4
    for j in range(3):
        sub = fig.add_axes([(xs["obs"][0] + gap + j * (size + gap)) / W, (y1 + 0.12) / H, size / W, size / H])
        covariance_panel(sub, covs[j], limit, STATES[j], "")
        for spine in sub.spines.values():
            spine.set_linewidth(1)
    order = ["data", "inf", "post", "alpha", "obs"]
    for a, b in zip(order[:-1], order[1:]):
        xa = xs[a][0] + xs[a][1]
        arrow(ax, (xa + 0.02, y1 + h1 / 2), (xs[b][0] - 0.02, y1 + h1 / 2))

    # Bottom row: the prior, the cost and the log-likelihood
    titled_box(xs["inf"][0], y2, xs["inf"][1], h2, "Model RNN", r"Sees $\theta_{1:t-1}$")
    titled_box(xs["post"][0], y2, xs["post"][1], h2, "Prior", r"$p(\theta_t \mid \theta_{1:t-1})$", top=True)
    gaussians(mini(xs["post"][0] + 0.1, y2 + 0.1, xs["post"][1] - 0.2, 0.5), [-1.3, 0.6, 1.5], 0.8)
    titled_box(xs["alpha"][0], y2, xs["alpha"][1], h2, "Cost", "$-$log-likelihood\n+ KL divergence")
    titled_box(xs["obs"][0], y2, xs["obs"][1], h2, "Log-likelihood", r"$\log \mathcal{N}(x_t \mid m_t, C_t)$")

    middle = (y1 + y2 + h2) / 2
    # Samples of the logits go to the model RNN
    xp = xs["post"][0] + 0.3
    elbow(ax, [(xp, y1 - 0.02), (xp, middle + 0.1), (centre("inf"), middle + 0.1), (centre("inf"), y2 + h2 + 0.02)])
    arrow(ax, (xs["inf"][0] + xs["inf"][1] + 0.02, y2 + h2 / 2), (xs["post"][0] - 0.02, y2 + h2 / 2))
    # KL divergence between the posterior and the prior
    xk = xs["post"][0] + 0.9
    arrow(ax, (xk, y2 + h2 + 0.02), (xk, y1 - 0.02), style="<|-|>")
    ax.text(xk + 0.08, middle + 0.1, "KL divergence", fontsize=9, fontweight="bold", ha="left", va="bottom")
    elbow(ax, [(xk, middle - 0.05), (centre("alpha"), middle - 0.05), (centre("alpha"), y2 + h2 + 0.02)])
    # Observation model to the log-likelihood and the log-likelihood to the cost
    arrow(ax, (centre("obs"), y1 - 0.02), (centre("obs"), y2 + h2 + 0.02))
    arrow(ax, (xs["obs"][0] - 0.02, y2 + h2 / 2), (xs["alpha"][0] + xs["alpha"][1] + 0.02, y2 + h2 / 2))

    save(fig, "dynemo-training", fmt)


def dynemo_regression_spectra(fmt):
    """Regression spectra: regress the spectrogram on the mixing coefficients."""
    from scipy import signal

    rng = np.random.default_rng(RNG_SEED + 7)
    fs = 250
    n_samples = 300 * fs
    alpha = softmax(3 * smooth_noise(n_samples, 3, rng, int(0.4 * fs)))
    # The power of each oscillation is proportional to the mixing coefficient,
    # plus a broad background (always on)
    x = oscillations(
        np.column_stack([alpha, np.ones(n_samples)]),
        frequencies=[10, 20, 5, 1],
        amplitudes=[1.5, 1.5, 1.5, 0.5],
        peak_widths=[1.2, 1.2, 1.2, 8.4],
        fs=fs,
        seed=RNG_SEED + 7,
    )

    f, psd = spectral.regression_spectra(
        x[:, None],
        alpha,
        fs,
        window_length=fs,
        step_size=fs // 10,
        frequency_range=[1, 30],
        calc_coh=False,
        return_coef_int=True,
    )
    psd = np.squeeze(psd)  # (2, n_modes, n_freq): coefficients and intercept
    coefs = psd[0]
    static_psd = psd[1][0]

    # Spectrogram of an excerpt
    start = 100 * fs
    excerpt = slice(start, start + 10 * fs)
    fs_, ts_, S = signal.spectrogram(
        x[start - fs : start + 11 * fs], fs, nperseg=fs // 2, noverlap=fs // 2 - fs // 20
    )
    ts_ = ts_ - 1
    keep = (fs_ >= 2) & (fs_ <= 30)
    t = np.arange(10 * fs) / fs

    fig = plt.figure(figsize=(7, 2.9))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.7, 1],
        height_ratios=[1.6, 0.6],
        hspace=0.15,
        wspace=0.3,
        left=0.08,
        right=0.98,
        top=0.88,
        bottom=0.16,
    )
    ax = fig.add_subplot(gs[0, 0])
    # The spectrogram is embedded at its own resolution and smoothed by the viewer
    df = fs_[1] - fs_[0]
    dt = ts_[1] - ts_[0]
    ax.imshow(
        S[keep],
        cmap=SEQUENTIAL,
        vmin=0,
        vmax=np.percentile(S[keep], 99),
        origin="lower",
        aspect="auto",
        interpolation="none",
        extent=[ts_[0] - dt / 2, ts_[-1] + dt / 2, fs_[keep][0] - df / 2, fs_[keep][-1] + df / 2],
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(2, 30)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram of the data before time-delay embedding", loc="left", pad=4)
    panel_label(ax, "a", x=-0.1, y=1.0)

    ax = fig.add_subplot(gs[1, 0])
    stacked(ax, t, alpha[excerpt])
    ax.set_yticks([0, 1])
    ax.set_ylabel(r"$\alpha_{jt}$", rotation=0, ha="right", va="center")
    ax.set_xlabel("Time (s)")

    right = gs[:, 1].subgridspec(2, 1, height_ratios=[1, 1.6], hspace=0.55)
    ax = fig.add_subplot(right[0])
    ax.plot(f, static_psd, color=MUTED, lw=1.3)
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(0, None)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Intercept (static PSD)", loc="left", pad=4)
    panel_label(ax, "b", x=-0.08, y=1.0)

    ax = fig.add_subplot(right[1])
    ax.axhline(0, color=FAINT, lw=0.6)
    for j in range(3):
        ax.plot(f, coefs[j], color=STATES[j], lw=1.3, label=f"Mode {j + 1}")
    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.2,
        borderaxespad=0,
    )
    ax.set_xlim(f[0], f[-1])
    ax.set_yticks([0])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title("Regression coefficients", loc="left", pad=4)
    panel_label(ax, "c", x=-0.08, y=1.0)

    save(fig, "dynemo-regression-spectra", fmt, smooth_images=True)


# -- DyNeStE -------------------------------------------------------------------


def chain_with_arcs(ax, xs, ys, yx, label, r=0.3, arcs=True):
    """A chain of latent nodes with observed nodes below, as in DyNeMo/DyNeStE."""
    labels = ["1", "2", "3", "T"]
    for x, n in zip(xs, labels):
        node(ax, (x, ys), label.format(n), r=r)
        node(ax, (x, yx), f"$x_{n}$", observed=True, r=r)
        link(ax, (x, ys), r, (x, yx), r)
    for a, b in zip(xs[:2], xs[1:3]):
        link(ax, (a, ys), r, (b, ys), r)
    ax.text((xs[2] + xs[3]) / 2, ys, r"$\cdots$", ha="center", va="center", fontsize=13)
    arrow(ax, (xs[2] + r, ys), ((xs[2] + xs[3]) / 2 - 0.3, ys))
    arrow(ax, ((xs[2] + xs[3]) / 2 + 0.3, ys), (xs[3] - r, ys))
    if arcs:
        for a, b, rad in [(0, 2, -0.45), (1, 3, -0.3), (0, 3, -0.3)]:
            arrow(
                ax,
                (xs[a] + 0.1, ys + r - 0.02),
                (xs[b] - 0.1, ys + r - 0.02),
                color=MUTED,
                lw=0.7,
                connectionstyle=f"arc3,rad={rad}",
            )


def legend_nodes(ax, lx, y0, latent_label):
    """Legend for the graphical models."""
    node(ax, (lx, y0), "", r=0.13)
    ax.text(lx + 0.25, y0, latent_label, va="center", fontsize=8.5)
    node(ax, (lx, y0 - 0.5), "", observed=True, r=0.13)
    ax.text(lx + 0.25, y0 - 0.5, "Observed data", va="center", fontsize=8.5)
    ax.add_patch(Circle((lx, y0 - 1.0), 0.06, color=INK))
    ax.text(lx + 0.25, y0 - 1.0, "Parameter", va="center", fontsize=8.5)


def dyneste_graphical_model(fmt):
    """Graphical model of DyNeStE: each state depends on all previous states."""
    fig = plt.figure(figsize=(6.2, 2.33))
    ax = canvas(fig, [0, 0, 1, 1], (0, 8.85), (0, 3.3))
    xs = [1.5, 2.8, 4.1, 6.2]
    chain_with_arcs(ax, xs, 1.95, 0.85, "$s_{}$")
    parameter(ax, (3.85, 0.12), r"$m_k, C_k$", label_offset=(0.12, 0))
    for x in xs:
        link(ax, (3.85, 0.12), 0.06, (x, 0.85), 0.3, color=MUTED, lw=0.7)
    legend_nodes(ax, 7.0, 2.45, "Hidden state")
    save(fig, "dyneste-graphical-model", fmt)


def dyneste_training(fmt):
    """Amortized variational inference in DyNeStE."""
    rng = np.random.default_rng(RNG_SEED + 8)
    W, H = 7, 3.0
    fig = plt.figure(figsize=(W, H))
    ax = canvas(fig, [0, 0, 1, 1], (0, W), (0, H))

    def titled_box(x, y, w, h, title, body):
        box(ax, (x, y), w, h)
        ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontweight="bold")
        ax.text(
            x + w / 2, y + (h - 0.35) / 2, body, ha="center", va="center",
            fontsize=9, linespacing=1.5,
        )

    y1, h1 = 1.85, 1.1
    y2, h2 = 0.1, 0.9
    xs = {
        "data": (0.03, 0.95),
        "inf": (1.2, 1.25),
        "post": (2.67, 1.2),
        "obs": (4.09, 2.88),
        "cost": (4.09, 1.1),
        "ll": (5.41, 1.56),
    }

    def centre(key):
        x, w = xs[key]
        return x + w / 2

    titled_box(xs["data"][0], y1, xs["data"][1], h1, "Data", "")
    sub = fig.add_axes([0.13 / W, (y1 + 0.12) / H, 0.75 / W, 0.6 / H])
    sub.axis("off")
    x = smooth_noise(100, 4, rng, 1.5)
    for c in range(4):
        sub.plot(x[:, c] * 0.2 + c, color=MUTED, lw=0.4)
    titled_box(xs["inf"][0], y1, xs["inf"][1], h1, "Inference RNN", "Bidirectional,\nsees $x_{1:L}$")
    titled_box(xs["post"][0], y1, xs["post"][1], h1, "Posterior", r"$q(s_t \mid x_{1:L})$," "\ncategorical")
    titled_box(xs["obs"][0], y1, xs["obs"][1], h1, "Observation model", r"$p(x_t \mid s_t = k) = \mathcal{N}(m_k, C_k)$")
    for a, b in [("data", "inf"), ("inf", "post"), ("post", "obs")]:
        xa = xs[a][0] + xs[a][1]
        arrow(ax, (xa + 0.02, y1 + h1 / 2), (xs[b][0] - 0.02, y1 + h1 / 2))

    titled_box(xs["inf"][0], y2, xs["inf"][1], h2, "Model RNN", r"Sees $s_{1:t-1}$")
    titled_box(xs["post"][0], y2, xs["post"][1], h2, "Prior", r"$p(s_t \mid s_{1:t-1})$")
    titled_box(xs["cost"][0], y2, xs["cost"][1], h2, "Cost", "$-$log-likelihood\n+ KL divergence")
    titled_box(xs["ll"][0], y2, xs["ll"][1], h2, "Log-likelihood", r"${\sum}_k q(s_t{=}k) \log \mathcal{N}(x_t \mid m_k, C_k)$")
    ax.texts[-1].set_fontsize(7.5)

    middle = (y1 + y2 + h2) / 2
    xp = xs["post"][0] + 0.3
    elbow(ax, [(xp, y1 - 0.02), (xp, middle + 0.1), (centre("inf"), middle + 0.1), (centre("inf"), y2 + h2 + 0.02)])
    ax.text(xs["inf"][0] + 0.15, middle + 0.16, "Gumbel-Softmax samples", fontsize=8.5, ha="left", va="bottom", color=MUTED)
    arrow(ax, (xs["inf"][0] + xs["inf"][1] + 0.02, y2 + h2 / 2), (xs["post"][0] - 0.02, y2 + h2 / 2))
    xk = xs["post"][0] + 0.9
    arrow(ax, (xk, y2 + h2 + 0.02), (xk, y1 - 0.02), style="<|-|>")
    ax.text(xk + 0.08, middle + 0.1, "KL divergence", fontsize=9, fontweight="bold", ha="left", va="bottom")
    elbow(ax, [(xk, middle - 0.05), (centre("cost"), middle - 0.05), (centre("cost"), y2 + h2 + 0.02)])
    arrow(ax, (centre("ll"), y1 - 0.02), (centre("ll"), y2 + h2 + 0.02))
    arrow(ax, (xs["ll"][0] - 0.02, y2 + h2 / 2), (xs["cost"][0] + xs["cost"][1] + 0.02, y2 + h2 / 2))

    save(fig, "dyneste-training", fmt)


def dyneste_gumbel_softmax(fmt):
    """Gumbel-Softmax samples of the states at different temperatures."""
    rng = np.random.default_rng(RNG_SEED + 9)
    n = 300
    t = np.arange(n) / 100
    trans_prob = np.full((3, 3), 0.01) + np.eye(3) * 0.97
    stc = markov_chain(trans_prob, n, rng)
    q = softmax(np.log(0.85 * stc + 0.15 * softmax(2 * smooth_noise(n, 3, rng, 5))))
    gumbel = -np.log(-np.log(rng.uniform(size=(n, 3))))

    rows = [(q, r"Posterior, $q(s_t)$")]
    for tau in [1.0, 0.5, 0.1]:
        rows.append((softmax((np.log(q) + gumbel) / tau), rf"Sample, $\tau = {tau:g}$"))

    fig, axes = plt.subplots(
        len(rows), 1, figsize=(7, 2.9), sharex=True,
        gridspec_kw={"hspace": 0.45, "left": 0.2, "right": 0.86},
    )
    for ax, (y, title) in zip(axes, rows):
        stacked(ax, t, y, max_points=len(t))
        ax.set_yticks([0, 1])
        ax.spines["left"].set_bounds(0, 1)
        ax.text(-0.04, 0.5, title, transform=ax.transAxes, ha="right", va="center")
    axes[-1].set_xlabel("Time (s)")
    handles = [mpl.patches.Patch(color=STATES[k], label=f"State {k + 1}") for k in range(3)]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.1), frameon=False, handlelength=1, handleheight=1)
    save(fig, "dyneste-gumbel-softmax", fmt)


# -- M-DyNeMo ------------------------------------------------------------------


def mdynemo_graphical_model(fmt):
    """Graphical model of M-DyNeMo: separate logits for power and FC."""
    fig = plt.figure(figsize=(6.2, 2.5))
    ax = canvas(fig, [0, 0, 1, 1], (0, 8.85), (0, 3.55))
    r = 0.3
    xs = [1.5, 2.8, 4.1, 6.2]
    labels = ["1", "2", "3", "T"]
    ya, yb, yx = 3.0, 1.95, 0.85
    for x, n in zip(xs, labels):
        node(ax, (x, ya), rf"$\theta^{{\alpha}}_{n}$", r=r, fontsize=9.5)
        node(ax, (x, yb), rf"$\theta^{{\beta}}_{n}$", r=r, fontsize=9.5)
        node(ax, (x, yx), f"$x_{n}$", observed=True, r=r)
        link(ax, (x, yb), r, (x, yx), r)
        # Power logits to the data, passing behind the FC node
        arrow(ax, (x - 0.38, ya - 0.2), (x - 0.25, yx + 0.17), connectionstyle="arc3,rad=0.25")
    for y in [ya, yb]:
        for a, b in zip(xs[:2], xs[1:3]):
            link(ax, (a, y), r, (b, y), r)
        ax.text((xs[2] + xs[3]) / 2, y, r"$\cdots$", ha="center", va="center", fontsize=13)
        arrow(ax, (xs[2] + r, y), ((xs[2] + xs[3]) / 2 - 0.3, y))
        arrow(ax, ((xs[2] + xs[3]) / 2 + 0.3, y), (xs[3] - r, y))
    # The model RNN sees both histories
    for a, b in zip(xs[:2], xs[1:3]):
        link(ax, (a, ya), r, (b, yb), r, color=MUTED, lw=0.7)
        link(ax, (a, yb), r, (b, ya), r, color=MUTED, lw=0.7)
    parameter(ax, (3.85, 0.12), r"$\mu_j, E_j, R_j$", label_offset=(0.12, 0))
    for x in xs:
        link(ax, (3.85, 0.12), 0.06, (x, yx), r, color=MUTED, lw=0.7)
    legend_nodes(ax, 7.0, 3.0, "Latent logits")
    save(fig, "mdynemo-graphical-model", fmt)


def mdynemo_generative_process(fmt):
    """Sampling data from M-DyNeMo: power and FC modes with separate dynamics."""
    rng = np.random.default_rng(RNG_SEED + 10)
    fs = 100
    n_samples = 4 * fs
    n_channels = 5
    t = np.arange(n_samples) / fs

    stds = np.array([[2.0, 2.0, 0.7, 0.7, 0.7], [0.7, 0.7, 1.8, 1.8, 0.7], [0.7, 0.7, 0.7, 1.4, 2.0]])
    corrs = [np.eye(n_channels) for _ in range(3)]
    corrs[0][0, 1] = corrs[0][1, 0] = 0.8
    corrs[1][2, 3] = corrs[1][3, 2] = -0.7
    corrs[2][3, 4] = corrs[2][4, 3] = 0.6
    corrs[2][2, 4] = corrs[2][4, 2] = 0.5
    corrs = np.array(corrs)

    alpha = softmax(2.5 * smooth_noise(n_samples, 3, rng, 8))
    beta = softmax(2.5 * smooth_noise(n_samples, 3, rng, 25))
    E = alpha @ stds  # (time, channels)
    R = np.einsum("tj,jmn->tmn", beta, corrs)
    C = E[:, :, None] * R * E[:, None, :]
    x = np.einsum("tij,tj->ti", np.linalg.cholesky(C), rng.standard_normal((n_samples, n_channels)))

    fig = plt.figure(figsize=(7, 5.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.6], hspace=0.32, top=0.93, bottom=0.08)
    top = outer[0].subgridspec(1, 7, width_ratios=[1, 1, 1, 0.3, 1, 1, 1], wspace=0.3)
    bottom = outer[1].subgridspec(3, 1, height_ratios=[0.8, 0.8, 1.6], hspace=0.5)
    fig.subplots_adjust(left=0.08, right=0.97)

    for j in range(3):
        ax = fig.add_subplot(top[0, j])
        ax.bar(np.arange(n_channels) + 1, stds[j], color=STATES[j], width=0.7)
        ax.set_ylim(0, 2.2)
        ax.set_xticks(np.arange(n_channels) + 1)
        ax.set_yticks([] if j else [0, 1, 2])
        ax.set_xlabel(f"$E_{j + 1}$", labelpad=2)
        if j == 0:
            ax.set_title("Power modes (std), $E_j$", loc="left", pad=6)
            panel_label(ax, "a", x=-0.3, y=1.06)
    for j in range(3):
        ax = fig.add_subplot(top[0, 4 + j])
        covariance_panel(ax, corrs[j], 1, STATES[j], f"$R_{j + 1}$")
        if j == 0:
            ax.set_title("FC modes (correlation), $R_j$", loc="left", pad=6)
            panel_label(ax, "b", x=-0.12, y=1.06)

    ax = fig.add_subplot(bottom[0])
    stacked(ax, t, alpha)
    ax.set_yticks([0, 1])
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title(r"Power mixing coefficients, $\alpha_t$", loc="left", pad=4)
    panel_label(ax, "c", x=-0.04, y=1.05)

    ax = fig.add_subplot(bottom[1])
    stacked(ax, t, beta)
    ax.set_yticks([0, 1])
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title(r"FC mixing coefficients, $\beta_t$", loc="left", pad=4)
    panel_label(ax, "d", x=-0.04, y=1.05)

    ax = fig.add_subplot(bottom[2])
    offsets = np.arange(n_channels)[::-1] * 6
    for c in range(n_channels):
        ax.plot(t, x[:, c] + offsets[c], color=INK, lw=0.5)
    ax.set_yticks(offsets, [f"{c + 1}" for c in range(n_channels)])
    ax.set_ylabel("Channel")
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(
        r"Generated data, $x_t \sim \mathcal{N}(m_t, E_t R_t E_t)$, "
        r"$E_t = {\sum}_j \alpha_{jt} E_j$, $R_t = {\sum}_j \beta_{jt} R_j$",
        loc="left",
        pad=4,
    )
    panel_label(ax, "e", x=-0.04, y=1.02)
    save(fig, "mdynemo-generative-process", fmt)


# -- HIVE ----------------------------------------------------------------------


def hive_graphical_model(fmt):
    """Graphical model of HIVE: an HMM for each session with session-specific parameters."""
    fig = plt.figure(figsize=(6.2, 2.9))
    ax = canvas(fig, [0, 0, 1, 1], (0, 8.85), (0, 4.1))
    r = 0.3
    ys, yx = 2.3, 1.2
    xs = [1.5, 2.8, 4.1, 6.2]
    labels = ["1", "2", "3", "T"]

    # Plate over sessions
    ax.add_patch(Rectangle((0.85, 0.15), 6.05, 3.0, facecolor="none", edgecolor=MUTED, lw=0.8))
    ax.text(6.85, 0.22, "Session $n$", ha="right", va="bottom", color=MUTED, fontsize=8.5)

    for x, n in zip(xs, labels):
        node(ax, (x, ys), f"$s_{n}$", r=r)
        node(ax, (x, yx), f"$x_{n}$", observed=True, r=r)
        link(ax, (x, ys), r, (x, yx), r)
    for a, b in zip(xs[:2], xs[1:3]):
        link(ax, (a, ys), r, (b, ys), r)
    ax.text(5.15, ys, r"$\cdots$", ha="center", va="center", fontsize=13)
    arrow(ax, (xs[2] + r, ys), (4.85, ys))
    arrow(ax, (5.45, ys), (xs[3] - r, ys))

    # Shared transition probabilities
    parameter(ax, (3.45, 3.7), "$A$", label_offset=(0.12, 0))
    for x in xs[1:]:
        link(ax, (3.45, 3.7), 0.06, (x, ys), r, color=MUTED, lw=0.7)

    # Session parameters from the group parameters and the session embedding
    sp = (3.85, 0.45)
    parameter(ax, sp, r"$m^{(n)}_k, C^{(n)}_k$", label_offset=(0.12, -0.02))
    for x in xs:
        link(ax, sp, 0.06, (x, yx), r, color=MUTED, lw=0.7)
    parameter(ax, (1.2, 0.45), r"$e_n$", label_offset=(0, 0.2), ha="center")
    arrow(ax, (1.28, 0.45), (sp[0] - 0.08, sp[1]), color=MUTED, lw=0.7)
    parameter(ax, (0.4, 0.45), r"$m_k, C_k$", label_offset=(0, -0.25), ha="center")
    ax.annotate(
        "", (sp[0] - 0.05, sp[1] - 0.05), (0.45, 0.4),
        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, mutation_scale=8,
                        connectionstyle="arc3,rad=0.12", shrinkA=0, shrinkB=0),
    )

    lx = 7.4
    node(ax, (lx, 2.8), "", r=0.13)
    ax.text(lx + 0.25, 2.8, "Hidden state", va="center", fontsize=8.5)
    node(ax, (lx, 2.3), "", observed=True, r=0.13)
    ax.text(lx + 0.25, 2.3, "Observed data", va="center", fontsize=8.5)
    ax.add_patch(Circle((lx, 1.8), 0.06, color=INK))
    ax.text(lx + 0.25, 1.8, "Parameter", va="center", fontsize=8.5)
    save(fig, "hive-graphical-model", fmt)


def hive_embeddings(fmt):
    """Session embeddings and the session-specific deviations of a state covariance."""
    rng = np.random.default_rng(RNG_SEED + 11)
    centres = np.array([[-1.2, 0.6], [1.1, 0.9], [0.1, -1.1]])
    n_per_site = 12
    emb = np.concatenate([c + 0.3 * rng.standard_normal((n_per_site, 2)) for c in centres])
    site = np.repeat(np.arange(3), n_per_site)

    group = example_covariances()[0]
    n_channels = group.shape[0]

    def deviation(direction):
        """Deviation of the group covariance for a session from a site."""
        u = np.zeros(n_channels)
        u[direction] = 1.0
        return 0.9 * np.outer(u, u) + 0.15 * rng.standard_normal((n_channels, n_channels))

    devs = [deviation([2, 3]), deviation([4])]
    devs = [(d + d.T) / 2 for d in devs]
    example_sessions = [np.flatnonzero(site == 0)[0], np.flatnonzero(site == 1)[0]]

    fig = plt.figure(figsize=(7, 2.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.45, left=0.06, right=0.97, top=0.85, bottom=0.15)

    ax = fig.add_subplot(gs[0, 0])
    for k in range(3):
        sel = site == k
        ax.scatter(emb[sel, 0], emb[sel, 1], s=14, color=STATES[k], lw=0, label=f"Site {'ABC'[k]}")
        ax.text(*(centres[k] + [0, 0.62]), f"Site {'ABC'[k]}", ha="center", va="bottom", fontsize=8.5)
    for i, sess in enumerate(example_sessions):
        ax.scatter(*emb[sess], s=40, facecolor="none", edgecolor=INK, lw=1)
        ax.text(emb[sess, 0] + 0.15, emb[sess, 1] - 0.15, f"{i + 1}", fontsize=8.5, va="top")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Embedding dimension 1")
    ax.set_ylabel("Embedding dimension 2")
    lim = np.abs(emb).max() + 0.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("Session embeddings, $e_n$", loc="left", pad=6)
    panel_label(ax, "a", x=-0.12, y=1.04)

    limit = np.abs(group).max()
    ax = fig.add_subplot(gs[0, 1])
    covariance_panel(ax, group, limit, INK, "$C_1$")
    ax.set_title("Group covariance", loc="left", pad=6)
    panel_label(ax, "b", x=-0.12, y=1.04)
    dlim = max(np.abs(d).max() for d in devs)
    for i, d in enumerate(devs):
        ax = fig.add_subplot(gs[0, 2 + i])
        covariance_panel(ax, d, dlim, INK, rf"$C^{{({i + 1})}}_1 - C_1$")
        ax.set_title(f"Session {i + 1} deviation", loc="left", pad=6)
        if i == 0:
            panel_label(ax, "c", x=-0.12, y=1.04)
    save(fig, "hive-embeddings", fmt)


FIGURES = {
    "pipeline": pipeline,
    "states-vs-modes": states_vs_modes,
    "hmm-graphical-model": hmm_graphical_model,
    "hmm-generative-process": hmm_generative_process,
    "hmm-training": hmm_training,
    "hmm-summary-stats": hmm_summary_stats,
    "hmm-spectra": hmm_spectra,
    "dynemo-graphical-model": dynemo_graphical_model,
    "dynemo-generative-process": dynemo_generative_process,
    "dynemo-training": dynemo_training,
    "dynemo-regression-spectra": dynemo_regression_spectra,
    "dyneste-graphical-model": dyneste_graphical_model,
    "dyneste-training": dyneste_training,
    "dyneste-gumbel-softmax": dyneste_gumbel_softmax,
    "mdynemo-graphical-model": mdynemo_graphical_model,
    "mdynemo-generative-process": mdynemo_generative_process,
    "hive-graphical-model": hive_graphical_model,
    "hive-embeddings": hive_embeddings,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--format", default="svg", choices=["svg", "pdf", "png"])
    parser.add_argument("--only", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        for name in args.only:
            FIGURES[name](args.format)


if __name__ == "__main__":
    main()
