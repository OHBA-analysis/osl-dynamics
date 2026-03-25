"""Preprocessing functions."""

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage.filters import uniform_filter1d

# -------------------------------------------------------------------------
# Artefact detection
# -------------------------------------------------------------------------


def detect_bad_segments(
    raw: mne.io.Raw,
    picks: Union[str, List[str]],
    mode: Optional[str] = None,
    metric: str = "std",
    window_length: Optional[int] = None,
    significance_level: float = 0.05,
    maximum_fraction: float = 0.1,
    ref_meg: str = "auto",
) -> mne.io.Raw:
    """Bad segment detection using the G-ESD algorithm.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object.
    picks : str or list of str
        Channel type to pick.
    mode : str, optional
        None or 'diff' to take the difference fo the time series
        before detecting bad segments.
    metric : str, optional
        Either 'std' (for standard deivation) or 'kurtosis'.
    window_length : int, optional
        Window length to used to calculate statistics.
        Defaults to twice the sampling frequency.
    significance_level : float, optional
        Significance level (p-value) to consider as an outlier.
    maximum_fraction : float, optional
        Maximum fraction of time series to mark as bad.
    ref_meg : str, optional
        ref_meg argument to pass to mne.pick_types.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object.
    """
    print()
    print("Bad segment detection")
    print("---------------------")

    if metric not in ["std", "kurtosis"]:
        raise ValueError("metric must be 'std' or 'kurtosis'.")

    if metric == "kurtosis":

        def _kurtosis(inputs):
            return stats.kurtosis(inputs, axis=None)

        metric_func = _kurtosis
    else:
        metric_func = np.std

    if window_length is None:
        window_length = int(raw.info["sfreq"] * 2)

    # Pick channels
    if picks == "eeg":
        chs = mne.pick_types(raw.info, eeg=True, exclude="bads")
    else:
        chs = mne.pick_types(raw.info, meg=picks, ref_meg=ref_meg, exclude="bads")

    # Get data
    data, times = raw.get_data(
        picks=chs, reject_by_annotation="omit", return_times=True
    )
    if mode == "diff":
        data = np.diff(data, axis=1)
        times = times[1:]

    # Calculate metric for each window
    metrics = []
    indices = []
    starts = np.arange(0, data.shape[1], window_length)
    for i in range(len(starts)):
        start = starts[i]
        if i == len(starts) - 1:
            stop = None
        else:
            stop = starts[i] + window_length
        m = metric_func(data[:, start:stop])
        metrics.append(m)
        indices += [i] * data[:, start:stop].shape[1]

    # Detect outliers
    bad_metrics_mask = _gesd(metrics, alpha=significance_level, p_out=maximum_fraction)
    bad_metrics_indices = np.where(bad_metrics_mask)[0]

    # Look up what indices in the original data are bad
    bad = np.isin(indices, bad_metrics_indices)

    # Make lists containing the start and end (index) of end bad segment
    onsets = np.where(np.diff(bad.astype(float)) == 1)[0] + 1
    if bad[0]:
        onsets = np.r_[0, onsets]
    offsets = np.where(np.diff(bad.astype(float)) == -1)[0] + 1
    if bad[-1]:
        offsets = np.r_[offsets, len(bad) - 1]
    assert len(onsets) == len(offsets)

    # Timing of the bad segments in seconds
    onsets = raw.first_samp / raw.info["sfreq"] + times[onsets.astype(int)]
    offsets = raw.first_samp / raw.info["sfreq"] + times[offsets.astype(int)]
    durations = offsets - onsets

    # Description for the annotation of the Raw object
    descriptions = np.repeat(f"bad_segment_{picks}", len(onsets))

    # Annotate the Raw object
    raw.annotations.append(onsets, durations, descriptions)

    # Summary statistics
    n_bad_segments = len(onsets)
    total_bad_time = durations.sum()
    total_time = raw.n_times / raw.info["sfreq"]
    percentage_bad = (total_bad_time / total_time) * 100

    # Print useful summary information
    print(f"Modality: {picks}")
    print(f"Mode: {mode}")
    print(f"Metric: {metric}")
    print(f"Significance level: {significance_level}")
    print(f"Maximum fraction: {maximum_fraction}")
    print(
        f"Found {n_bad_segments} bad segments: "
        f"{total_bad_time:.1f}/{total_time:.1f} "
        f"seconds rejected ({percentage_bad:.1f}%)"
    )

    return raw


def detect_bad_channels(
    raw: mne.io.Raw,
    picks: str,
    ref_meg: str = "auto",
    significance_level: float = 0.05,
    log10: bool = True,
) -> mne.io.Raw:
    """Detect bad channels using the G-ESD algorithm based on standard deviation.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object.
    picks : str
        Channel types to pick. See Notes for recommendations.
    ref_meg : str, optional
        ref_meg argument to pass with mne.pick_types.
    significance_level : float, optional
        Significance level for detecting outliers. Must be between 0-1.
    log10 : bool, optional
        Should we apply a log10 transform to the standard deviations?
        This is normally a good idea to make sure the standard deviations
        are normally distributed.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object with bad channels marked.

    Notes
    -----
    For Elekta/MEGIN data, we recommend using picks='mag' or picks='grad'
    separately (in no particular order).

    Note that with CTF data, mne.pick_types will return:
        ~274 axial grads (as magnetometers) if picks='mag', ref_meg=False
        ~28 reference axial grads if picks='grad'.
    Thus, it is recommended to use picks='mag' in combination with ref_mag=False,
    and picks='grad' separately (in no particular order).
    """
    print()
    print("Bad channel detection")
    print("---------------------")

    # Select channels
    if (picks == "mag") or (picks == "grad"):
        ch_inds = mne.pick_types(raw.info, meg=picks, ref_meg=ref_meg, exclude="bads")
    elif picks == "meg":
        ch_inds = mne.pick_types(raw.info, meg=True, ref_meg=ref_meg, exclude="bads")
    elif picks == "eeg":
        ch_inds = mne.pick_types(raw.info, eeg=True, ref_meg=ref_meg, exclude="bads")
    elif picks == "eog":
        ch_inds = mne.pick_types(raw.info, eog=True, ref_meg=ref_meg, exclude="bads")
    elif picks == "ecg":
        ch_inds = mne.pick_types(raw.info, ecg=True, ref_meg=ref_meg, exclude="bads")
    elif picks == "misc":
        ch_inds = mne.pick_types(raw.info, misc=True, exclude="bads")
    else:
        raise NotImplementedError(f"picks={picks} not available.")

    # Calculate standard deviation for each channel
    data = raw.get_data(picks=ch_inds)
    std = np.std(data, axis=-1)
    if log10:
        std = np.log10(std)

    # Detect outliers
    mask = _gesd(std, alpha=significance_level)
    chs = np.array(raw.ch_names)[ch_inds]
    bads = list(chs[mask])

    # Mark as bad
    for bad in bads:
        if bad not in raw.info["bads"]:
            raw.info["bads"].append(bad)

    # Print useful summary information
    print(f"{len(bads)} bad channels:")
    print(np.array(bads))

    return raw


def _gesd(
    X: np.ndarray,
    alpha: float,
    p_out: float = 0.1,
    outlier_side: int = 0,
) -> np.ndarray:
    """Generalised-ESD (Rosner) test for outliers.

    Parameters
    ----------
    X : array-like, 1D
        Data to test. NaNs are ignored (treated as non-tested).
    alpha : float
        Significance level (0 < alpha < 1).
    p_out : float
        Maximum fraction of points that may be flagged as outliers (0..1).
    outlier_side : int
        -1 -> look for small outliers
         0 -> two-sided (both small and large) -- default
         1 -> look for large outliers

    Returns
    -------
    mask : np.ndarray (bool)
        Boolean array of same length as X. True indicates an outlier.

    Notes
    -----
    B. Rosner (1983). Percentage Points for a Generalized ESD
    Many-Outlier Procedure. Technometrics 25(2), pp. 165-172.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 1:
        raise ValueError("_gesd expects a 1D array-like input.")
    if not (0 <= p_out <= 1):
        raise ValueError("p_out must be between 0 and 1.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1).")
    if outlier_side not in (-1, 0, 1):
        raise ValueError("outlier_side must be -1, 0, or 1.")

    finite_mask = np.isfinite(X)
    Xf = X[finite_mask]
    n = Xf.size
    if n == 0:
        return np.zeros_like(X, dtype=bool)

    # maximum number of outliers to consider
    n_out = int(np.floor(n * float(p_out)))
    if n_out <= 0:
        return np.zeros_like(X, dtype=bool)

    # Arrays to hold statistics for each removal step
    R = np.zeros(n_out, dtype=float)
    lam = np.zeros(n_out, dtype=float)
    rm_order = (
        []
    )  # stores the original indices of removed points (relative to finite subset)

    # Work on a working copy and an index map to original finite indices
    arr = Xf.copy()
    idx_map = np.arange(n)

    for i in range(n_out):
        # compute the current mean (ignoring NaNs)
        mean_val = np.nanmean(arr)
        # choose removal index based on outlier_side
        if outlier_side == -1:
            rm = int(np.nanargmin(arr))
            dev = mean_val - arr[rm]
        elif outlier_side == 1:
            rm = int(np.nanargmax(arr))
            dev = arr[rm] - mean_val
        else:  # two-sided
            diffs = np.abs(arr - mean_val)
            rm = int(np.nanargmax(diffs))
            dev = diffs[rm]
        # store the original index of the removed element
        rm_order.append(int(idx_map[rm]))

        sigma = np.nanstd(arr, ddof=0)
        if sigma == 0 or np.isnan(sigma):
            R[i] = 0.0
        else:
            R[i] = dev / sigma

        # remove the element from arr and idx_map for next iteration
        arr = np.delete(arr, rm)
        idx_map = np.delete(idx_map, rm)

        # compute lambda (critical value) for this iteration
        m = n - i  # remaining sample size before removal
        # if there are too few degrees of freedom, set critical to +inf so no detection
        if m - 2 <= 0:
            lam[i] = np.inf
        else:
            if outlier_side == 0:
                # two-sided: adjust alpha/2 per Rosner's guidance
                p = 1 - alpha / (2 * m)
            else:
                p = 1 - alpha / m
            t = stats.t.ppf(p, m - 2)
            lam[i] = ((m - 1) * t) / (np.sqrt((m - 2 + t**2) * m))

    # Determine largest k (0-based) where R[k] > lam[k]
    k_candidates = np.where(R > lam)[0]
    if k_candidates.size == 0:
        out_mask_finite = np.zeros(n, dtype=bool)
    else:
        k = int(k_candidates.max())
        # the first k+1 entries of rm_order are flagged as outliers
        out_idx = np.array(rm_order[: k + 1], dtype=int)
        out_mask_finite = np.zeros(n, dtype=bool)
        out_mask_finite[out_idx] = True

    # Map back to original full-length mask (NaNs are False)
    out_mask = np.zeros_like(X, dtype=bool)
    out_mask[np.where(finite_mask)[0]] = out_mask_finite
    return out_mask


# -------------------------------------------------------------------------
# ICA artefact rejection
# -------------------------------------------------------------------------


def ica_label(
    raw: mne.io.Raw,
    picks: str = "mag",
    n_components: int = 20,
    method: str = "megnet",
    threshold: float = 0.5,
    random_state: int = 42,
) -> Tuple[mne.io.Raw, Any, dict]:
    """Automatic ICA artefact rejection using mne-icalabel.

    Fits ICA on a bandpass-filtered copy of the data, labels components
    using a pre-trained classifier, and removes artefact components from
    the original data.

    For MEG data, uses the MEGNet classifier. For EEG data, uses ICLabel.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object.
    picks : str, optional
        Channel type to use for ICA. For Elekta MEG data, use ``"mag"``
        (MEGNet was trained on magnetometer topographies).
    n_components : int, optional
        Number of ICA components. Should not exceed the data rank.
        For MaxFiltered Elekta data (rank ~60), 20 is a safe default.
    method : str, optional
        Labelling method: ``"megnet"`` for MEG or ``"iclabel"`` for EEG.
    threshold : float, optional
        Probability threshold (0–1) for excluding a component. Components
        labelled as artefact with probability above this threshold are
        removed.
    random_state : int, optional
        Random seed for ICA reproducibility.

    Returns
    -------
    raw : mne.io.Raw
        Cleaned MNE Raw object.
    ica : mne.preprocessing.ICA
        Fitted ICA object with ``exclude`` set.
    ic_labels : dict
        Dictionary with keys ``"labels"`` (list of str) and
        ``"y_pred_proba"`` (array of float).

    Notes
    -----
    Requires ``mne-icalabel >= 0.4``. Install with::

        pip install mne-icalabel

    For EEG data, use ``picks="eeg"`` and ``method="iclabel"``::

        raw, ica, ic_labels = preproc.ica_label(
            raw, picks="eeg", method="iclabel", n_components=30,
        )
    """
    try:
        from mne_icalabel import label_components
    except ImportError:
        raise ImportError(
            "mne-icalabel is required for ICA artefact rejection. "
            "Install it with: pip install mne-icalabel"
        )
    from mne.preprocessing import ICA

    print()
    print("ICA artefact rejection")
    print("----------------------")
    print(f"Method: {method}")
    print(f"Picks: {picks}")
    print(f"Components: {n_components}")
    print(f"Threshold: {threshold}")

    # Filter a copy for ICA fitting (classifiers expect 1-100 Hz)
    print("Filtering data copy (1-100 Hz) for ICA fitting...")
    raw_fit = raw.copy().filter(l_freq=1.0, h_freq=100.0, verbose=False)

    # ICLabel (EEG) requires average reference
    if method == "iclabel":
        raw_fit.set_eeg_reference("average", verbose=False)

    # Fit ICA
    print("Fitting ICA...")
    ica = ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=random_state,
        verbose=False,
    )
    ica.fit(raw_fit, picks=picks)

    # Label components
    print("Labelling components...")
    ic_labels = label_components(raw_fit, ica, method=method)

    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]

    # Identify artefact components (exclude everything except brain/other)
    exclude_idx = []
    for idx, (label, prob) in enumerate(zip(labels, probs)):
        if label not in ["brain", "other"]:
            if prob > threshold:
                exclude_idx.append(idx)
                print(f"  ICA{idx:03d}: {label} ({prob:.2f}) -> excluded")
            else:
                print(
                    f"  ICA{idx:03d}: {label} ({prob:.2f}) -> kept "
                    f"(below threshold)"
                )

    # Apply to original data
    if len(exclude_idx) > 0:
        print(f"Removing {len(exclude_idx)} artefact component(s)...")
        ica.exclude = exclude_idx
        ica.apply(raw)
    else:
        print("No artefact components found.")

    return raw, ica, ic_labels


def ica_ecg_eog_correlation(
    raw: mne.io.Raw,
    picks: str = "meg",
    n_components: int = 40,
    ecg: bool = True,
    eog: bool = True,
    ecg_method: str = "ctps",
    ecg_threshold: Union[str, float] = "auto",
    eog_threshold: float = 3.0,
    random_state: int = 42,
) -> Tuple[mne.io.Raw, Any, dict]:
    """ICA artefact rejection using ECG/EOG correlation.

    Fits ICA on a bandpass-filtered copy of the data, identifies artefact
    components by correlating with ECG and EOG signals, and removes them
    from the original data.

    Uses ``picks="meg"`` by default so both magnetometers and gradiometers
    are denoised. Does not require mne-icalabel.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object.
    picks : str, optional
        Channel type to use for ICA. ``"meg"`` fits on both mags and grads.
    n_components : int, optional
        Number of ICA components. Should not exceed the data rank.
        For MaxFiltered Elekta data (rank ~60), 40 is a safe default.
    ecg : bool, optional
        Whether to detect ECG artefact components. If no ECG channel is
        present, a synthetic ECG is created from the magnetometers.
    eog : bool, optional
        Whether to detect EOG artefact components. Requires an EOG
        channel; skipped with a warning if absent.
    ecg_method : str, optional
        Method for ECG detection: ``"ctps"`` (cross-trial phase
        statistics) or ``"correlation"``.
    ecg_threshold : str or float, optional
        Threshold for ECG component detection.
    eog_threshold : float, optional
        Z-score threshold for EOG component detection.
    random_state : int, optional
        Random seed for ICA reproducibility.

    Returns
    -------
    raw : mne.io.Raw
        Cleaned MNE Raw object.
    ica : mne.preprocessing.ICA
        Fitted ICA object with ``exclude`` set.
    ic_labels : dict
        Dictionary with keys ``"labels"`` (list of str) and
        ``"y_pred_proba"`` (array of float), compatible with
        :func:`plot_ica_components`.

    Notes
    -----
    For EEG data, use ``picks="eeg"``. Note that synthetic ECG detection
    only works with MEG magnetometers — for EEG data a dedicated ECG
    channel must be present, otherwise set ``ecg=False``::

        raw, ica, ic_labels = preproc.ica_ecg_eog_correlation(
            raw, picks="eeg", n_components=30, ecg=False,
        )
    """
    from mne.preprocessing import ICA

    print()
    print("ICA artefact rejection (ECG/EOG correlation)")
    print("---------------------------------------------")
    print(f"Picks: {picks}")
    print(f"Components: {n_components}")
    print(f"ECG: {ecg}, EOG: {eog}")

    # Filter a copy for ICA fitting
    print("Filtering data copy (1-100 Hz) for ICA fitting...")
    raw_fit = raw.copy().filter(l_freq=1.0, h_freq=100.0, verbose=False)

    # Fit ICA
    print("Fitting ICA...")
    ica = ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=random_state,
        verbose=False,
    )
    ica.fit(raw_fit, picks=picks)

    # Detect ECG components
    ecg_indices = []
    ecg_scores = np.zeros(ica.n_components_)
    if ecg:
        print("Detecting ECG components...")
        try:
            ecg_indices, ecg_scores = ica.find_bads_ecg(
                raw_fit,
                method=ecg_method,
                threshold=ecg_threshold,
                verbose=False,
            )
            for idx in ecg_indices:
                print(f"  ICA{idx:03d}: ecg (score={ecg_scores[idx]:.2f})")
            if not ecg_indices:
                print("  No ECG components found.")
        except Exception as e:
            print(f"  ECG detection failed: {e}")

    # Detect EOG components
    eog_indices = []
    eog_scores = np.zeros(ica.n_components_)
    if eog:
        # Check if EOG channels exist
        eog_chs = mne.pick_types(raw_fit.info, eog=True)
        if len(eog_chs) == 0:
            print("No EOG channel found, skipping EOG detection.")
        else:
            print("Detecting EOG components...")
            try:
                eog_indices, eog_scores = ica.find_bads_eog(
                    raw_fit,
                    threshold=eog_threshold,
                    verbose=False,
                )
                # eog_scores can be a list of arrays if multiple EOG channels
                if isinstance(eog_scores, list):
                    eog_scores = np.max(np.abs(eog_scores), axis=0)
                for idx in eog_indices:
                    print(f"  ICA{idx:03d}: eog (score={eog_scores[idx]:.2f})")
                if not eog_indices:
                    print("  No EOG components found.")
            except Exception as e:
                print(f"  EOG detection failed: {e}")

    # Combine and exclude
    exclude_idx = sorted(set(ecg_indices + eog_indices))

    if len(exclude_idx) > 0:
        print(f"Removing {len(exclude_idx)} artefact component(s)...")
        ica.exclude = exclude_idx
        ica.apply(raw)
    else:
        print("No artefact components found.")

    # Build ic_labels dict (same structure as ica_label output)
    labels = []
    probs = []
    for idx in range(ica.n_components_):
        if idx in ecg_indices and idx in eog_indices:
            labels.append("ecg+eog")
            probs.append(max(abs(ecg_scores[idx]), abs(eog_scores[idx])))
        elif idx in ecg_indices:
            labels.append("ecg")
            probs.append(abs(ecg_scores[idx]))
        elif idx in eog_indices:
            labels.append("eog")
            probs.append(abs(eog_scores[idx]))
        else:
            labels.append("brain")
            probs.append(0.0)

    ic_labels = {"labels": labels, "y_pred_proba": np.array(probs)}

    return raw, ica, ic_labels


def plot_ica_components(
    ica: Any,
    ic_labels: dict,
) -> plt.Figure:
    """Plot ICA component topographies with labels.

    Creates a composite figure showing all ICA components with their
    classification labels and probabilities. Excluded components are
    highlighted in red.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    ic_labels : dict
        Dictionary with keys ``"labels"`` and ``"y_pred_proba"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The composite figure.
    """
    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]
    n_components = len(labels)

    # Create composite figure
    n_cols = min(n_components, 5)
    n_rows = (n_components + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n_components):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Plot individual component topography
        comp_figs = ica.plot_components(picks=[idx], show=False)
        if isinstance(comp_figs, list):
            comp_fig = comp_figs[0]
        else:
            comp_fig = comp_figs
        comp_fig.canvas.draw()
        buf = comp_fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        plt.close(comp_fig)

        ax.imshow(img)
        ax.set_axis_off()

        # Title with label and probability
        label = labels[idx]
        prob = probs[idx]
        excluded = idx in ica.exclude
        color = "red" if excluded else "black"
        marker = " *" if excluded else ""
        ax.set_title(
            f"ICA{idx:03d}: {label} ({prob:.2f}){marker}",
            fontsize=12,
            color=color,
            fontweight="bold" if excluded else "normal",
        )

    # Hide unused axes
    for idx in range(n_components, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_axis_off()

    fig.suptitle(
        f"ICA Components — {len(ica.exclude)} excluded " f"(marked in red with *)",
        fontsize=16,
    )
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Headshape decimation
# -------------------------------------------------------------------------


def decimate_headshape_points(
    raw: mne.io.Raw,
    decimate_amount: float = 0.01,
    include_facial_info: bool = True,
    remove_zlim: Optional[float] = -0.02,
    angle: float = 0,
    method: str = "gridaverage",
    face_Z: Optional[List[float]] = None,
    face_Y: Optional[List[float]] = None,
    face_X: Optional[List[float]] = None,
    decimate_facial_info: bool = True,
    decimate_facial_info_amount: float = 0.01,
) -> mne.io.Raw:
    """Decimate headshape points.

    Useful for reducing the number of headshape points collected using an
    EinScan for OPM recordings.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object.
    decimate_amount : float, optional
        Bin width in metres to decimate.
    include_facial_info : bool, optional
        Should we keep facial headshape points?
    remove_zlim : float, optional
        Remove headshape points below this z-value (in metres).
    angle : float, optional
        How much should we rotate the headshape points?
    method : str, optional
        What method should we use for decimation?
    face_Z : list, optional
        Keep headshape points within these z-values (in metres).
    face_Y : list, optional
        Keep headshape points within these y-values (in metres).
    face_X : list, optional
        Keep headshape points within these x-values (in metres).
    decimate_facial_info : bool, optional
        Should we decimate facial headshape points?
    decimate_facial_info_amount : float, optional
        Bin width in metres to decimate.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object.
    """
    if face_Z is None:
        face_Z = [-0.06, 0.02]
    if face_Y is None:
        face_Y = [0.06, 0.15]
    if face_X is None:
        face_X = [-0.03, 0.03]

    print()
    print("Decimate headshape points")
    print("-------------------------")

    dig = raw.info["dig"]
    headshape = np.array([d["r"] for d in dig if "r" in d])
    print("Digitization points:", headshape.shape)

    decimated_headshape = _decimate_headshape(
        headshape,
        decimate_amount=decimate_amount,
        include_facial_info=include_facial_info,
        remove_zlim=remove_zlim,
        angle=angle,
        method=method,
        face_Z=face_Z,
        face_Y=face_Y,
        face_X=face_X,
        decimate_facial_info=decimate_facial_info,
        decimate_facial_info_amount=decimate_facial_info_amount,
    )

    # Initialize fiducial positions
    fid_positions = {"nasion": None, "lpa": None, "rpa": None}

    # Extract fiducials from the dig points
    for f in dig:
        if f["coord_frame"] == 4:  # Ensure head coordinate frame
            if f["ident"] == 2 and fid_positions["nasion"] is None:
                fid_positions["nasion"] = f["r"]
            elif f["ident"] == 1 and fid_positions["lpa"] is None:
                fid_positions["lpa"] = f["r"]
            elif f["ident"] == 3 and fid_positions["rpa"] is None:
                fid_positions["rpa"] = f["r"]

    # Verify the extracted fiducials
    if any(v is None for v in fid_positions.values()):
        raise RuntimeError(
            "One or more fiducials (nasion, LPA, RPA) not found in "
            "the head coordinate frame."
        )

    # Create a DigMontage using the extracted fiducials
    # and decimated headshape points
    montage = mne.channels.make_dig_montage(
        hsp=decimated_headshape,
        nasion=fid_positions["nasion"],
        lpa=fid_positions["lpa"],
        rpa=fid_positions["rpa"],
        coord_frame="head",
    )

    # Set the new montage
    return raw.set_montage(montage)


def _decimate_headshape(
    headshape: np.ndarray,
    decimate_amount: float = 0.015,
    include_facial_info: bool = True,
    remove_zlim: Optional[float] = 0.02,
    angle: float = 10,
    method: str = "gridaverage",
    face_Z: Optional[List[float]] = None,
    face_Y: Optional[List[float]] = None,
    face_X: Optional[List[float]] = None,
    decimate_facial_info: bool = True,
    decimate_facial_info_amount: float = 0.008,
) -> np.ndarray:
    """Decimate headshape points.

    Parameters
    ----------
    - headshape : np.ndarray
        Nx3 array of headshape points in meters.
    - include_facial_info : bool
        Include facial points if True.
    - remove_zlim : float
        Remove points above nasion on the z-axis in meters.
    - method : str
        Downsampling method. Note: only method supported is 'gridaverage'.
    - facial_info_above_z (float): float
        Max z-value for facial points in meters.
    - facial_info_below_z : float
        Min z-value for facial points in meters.
    - facial_info_above_y : float
        Max y-value for facial points in meters.
    - facial_info_below_y : float
        Min y-value for facial points in meters.
    - facial_info_below_x : float
        Min x-value for facial points in meters.
    - decimate_facial_info : bool
        Whether to decimate facial points.
    - decimate_facial_info_amount : float
        Grid size for downsampling facial info in meters.

    Returns
    -------
    decimated_headshape : np.ndarray
        Decimated headshape points.
    """
    if face_Z is None:
        face_Z = [-0.08, 0.02]
    if face_Y is None:
        face_Y = [0.06, 0.15]
    if face_X is None:
        face_X = [-0.07, 0.07]

    if include_facial_info:
        facial_mask = (
            (headshape[:, 2] > face_Z[0])
            & (headshape[:, 2] < face_Z[1])
            & (headshape[:, 1] > face_Y[0])
            & (headshape[:, 1] < face_Y[1])
            & (headshape[:, 0] > face_X[0])
            & (headshape[:, 0] < face_X[1])
        )
        facial_points = headshape[facial_mask]
        if decimate_facial_info:
            facial_points = _grid_average_decimate(
                facial_points, decimate_facial_info_amount
            )
    if remove_zlim is not None:
        print("Removing points below zlim")
        rotated_headshape = _rotate_pointcloud(headshape, angle, "x")
        z_mask = rotated_headshape[:, 2] > remove_zlim
        filtered_rotated_points = rotated_headshape[z_mask]
        headshape = _rotate_pointcloud(filtered_rotated_points, -angle, "x")
    if method == "gridaverage":
        print(f"Using {method}")
        headshape = _grid_average_decimate(headshape, decimate_amount)
    else:
        raise ValueError(f"Unsupported decimation method: {method}")
    if include_facial_info:
        headshape = np.vstack((headshape, facial_points))
    return headshape


def _rotate_pointcloud(
    points: np.ndarray,
    angle_degrees: float,
    axis: str = "x",
) -> np.ndarray:
    """
    Rotates the point cloud around a specified axis.

    Parameters
    ----------
    points : np.ndarray
        Headshape points
    angle_degrees : float
        Amount to rotate in degrees.
    axis : str
        Axis to rotate.
    """
    angle_radians = np.radians(angle_degrees)
    if axis == "x":
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_radians), -np.sin(angle_radians)],
                [0, np.sin(angle_radians), np.cos(angle_radians)],
            ]
        )
    elif axis == "y":
        rotation_matrix = np.array(
            [
                [np.cos(angle_radians), 0, np.sin(angle_radians)],
                [0, 1, 0],
                [-np.sin(angle_radians), 0, np.cos(angle_radians)],
            ]
        )
    elif axis == "z":
        rotation_matrix = np.array(
            [
                [np.cos(angle_radians), -np.sin(angle_radians), 0],
                [np.sin(angle_radians), np.cos(angle_radians), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
    return np.dot(points, rotation_matrix.T)


def _grid_average_decimate(
    point_cloud: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Decimate a point cloud using grid averaging.

    This function divides the space into a voxel grid, computes the average
    position of points within each voxel, and returns a decimated point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray
        A numpy array of shape (N, 3) representing the point cloud, where N
        is the number of points, and each point has (x, y, z) coordinates.

    voxel_size : float
        The size of the voxel grid. Points within a grid cell are averaged
        to compute the decimated point.

    Returns
    -------
    decimated_cloud : np.ndarray
        A numpy array of shape (M, 3) representing the decimated point cloud,
        where M is the number of voxels containing points.

    Notes
    -----
    - This method assumes the input point cloud is dense and unstructured.
    - For very large point clouds, consider optimizing memory usage.
    """
    voxel_indices = np.floor(point_cloud / voxel_size).astype(np.int32)
    voxel_dict = {}
    for idx, point in zip(voxel_indices, point_cloud):
        key = tuple(idx)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(point)
    return np.array([np.mean(voxel_dict[key], axis=0) for key in voxel_dict])


# -------------------------------------------------------------------------
# QC plots
# -------------------------------------------------------------------------


def save_qc_plots(
    raw: mne.io.Raw,
    output_dir: Union[str, Path],
    show: bool = False,
    ica: Any = None,
    ic_labels: Optional[dict] = None,
) -> None:
    """Save preprocessing QC plots and summary.

    Saves the following files to output_dir:

    - ``1_summary.json``: preprocessing summary stats
    - ``1_psd.png``: sensor-level PSD
    - ``1_sum_square.png``: sum-square time series
    - ``1_sum_square_exclude_bads.png``: sum-square excluding bad
      segments/channels
    - ``1_channel_stds.png``: channel standard deviation distributions
    - ``1_ica_components.png``: ICA component topographies (if ``ica``
      and ``ic_labels`` are provided)

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed MNE Raw object.
    output_dir : str or Path
        Directory to save plots to.
    show : bool, optional
        Whether to display the plots interactively. Default is False.
    ica : mne.preprocessing.ICA, optional
        Fitted ICA object. If provided along with ``ic_labels``, saves
        ICA component topography plot.
    ic_labels : dict, optional
        ICA label dictionary from ``ica_label``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save preprocessing summary
    total_duration = raw.times[-1]
    bad_duration = sum(
        a["duration"] for a in raw.annotations if a["description"].startswith("bad")
    )
    summary = {
        "total_duration_s": round(total_duration, 1),
        "bad_duration_s": round(bad_duration, 1),
        "bad_percent": round(100 * bad_duration / total_duration, 1),
        "bad_channels": raw.info["bads"],
        "n_bad_channels": len(raw.info["bads"]),
    }
    if ica is not None and ic_labels is not None:
        summary["ica_n_components"] = ica.n_components_
        summary["ica_n_excluded"] = len(ica.exclude)
        summary["ica_excluded_labels"] = [
            f"{ic_labels['labels'][i]} ({ic_labels['y_pred_proba'][i]:.2f})"
            for i in ica.exclude
        ]
    with open(output_dir / "1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # PSD
    raw.compute_psd(fmax=45).plot()
    plt.savefig(output_dir / "1_psd.png", dpi=150, bbox_inches="tight")
    if not show:
        plt.close("all")

    # Sum-square time series
    plot_sum_square_time_series(raw)
    plt.savefig(output_dir / "1_sum_square.png", dpi=150, bbox_inches="tight")
    if not show:
        plt.close("all")

    # Sum-square excluding bads
    plot_sum_square_time_series(raw, exclude_bads=True)
    plt.savefig(
        output_dir / "1_sum_square_exclude_bads.png", dpi=150, bbox_inches="tight"
    )
    if not show:
        plt.close("all")

    # Channel standard deviations
    plot_channel_stds(raw)
    plt.savefig(output_dir / "1_channel_stds.png", dpi=150, bbox_inches="tight")
    if not show:
        plt.close("all")

    # ICA component topographies
    if ica is not None and ic_labels is not None:
        plot_ica_components(ica, ic_labels)
        plt.savefig(output_dir / "1_ica_components.png", dpi=150, bbox_inches="tight")
        if not show:
            plt.close("all")


def plot_sum_square_time_series(
    raw: mne.io.Raw,
    exclude_bads: bool = False,
) -> None:
    """Plot sum-square time series.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object.
    exclude_bads : bool, optional
        Whether to exclude bad channels and bad segments.
    """
    if exclude_bads:
        # excludes bad channels and bad segments
        exclude = "bads"
    else:
        # includes bad channels and bad segments
        exclude = []

    is_ctf = raw.info["dev_ctf_t"] is not None

    if is_ctf:
        # Note that with CTF mne.pick_types will return:
        # ~274 axial grads (as magnetometers) if {picks: 'mag', ref_meg: False}
        # ~28 reference axial grads if {picks: 'grad'}

        channel_types = {
            "Axial Grads (chtype=mag)": mne.pick_types(
                raw.info, meg="mag", ref_meg=False, exclude=exclude
            ),
            "Ref Axial Grad (chtype=ref_meg)": mne.pick_types(
                raw.info, meg="grad", exclude=exclude
            ),
            "EEG": mne.pick_types(raw.info, eeg=True),
            "CSD": mne.pick_types(raw.info, csd=True),
        }
    else:
        channel_types = {
            "Magnetometers": mne.pick_types(raw.info, meg="mag", exclude=exclude),
            "Gradiometers": mne.pick_types(raw.info, meg="grad", exclude=exclude),
            "EEG": mne.pick_types(raw.info, eeg=True),
            "CSD": mne.pick_types(raw.info, csd=True),
        }

    t = raw.times
    x = raw.get_data()

    # Number of subplots, i.e. the number of different channel types in the fif file
    nrows = 0
    for _, c in channel_types.items():
        if len(c) > 0:
            nrows += 1

    if nrows == 0:
        return None

    # Make sum-square plots
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(16, 4))
    if nrows == 1:
        ax = [ax]
    row = 0
    for name, chan_inds in channel_types.items():
        if len(chan_inds) == 0:
            continue
        ss = np.sum(x[chan_inds] ** 2, axis=0)

        # calculate ss value to give to bad segments for plotting purposes
        good_data = raw.get_data(picks=chan_inds, reject_by_annotation="NaN")
        # get indices of good data
        good_inds = np.where(~np.isnan(good_data[0, :]))[0]
        ss_bad_value = np.mean(ss[good_inds])

        if exclude_bads:
            # set bad segs to mean
            for aa in raw.annotations:
                if "bad_segment" in aa["description"]:
                    time_inds = np.where(
                        (raw.times >= aa["onset"] - raw.first_time)
                        & (raw.times <= (aa["onset"] + aa["duration"] - raw.first_time))
                    )[0]
                    ss[time_inds] = ss_bad_value

        ss = uniform_filter1d(ss, int(raw.info["sfreq"]))

        ax[row].plot(t, ss)
        ax[row].legend([name], frameon=False, fontsize=16)
        ax[row].set_xlim(t[0], t[-1])
        for a in raw.annotations:
            if "bad_segment" in a["description"]:
                ax[row].axvspan(
                    a["onset"] - raw.first_time,
                    a["onset"] + a["duration"] - raw.first_time,
                    color="red",
                    alpha=0.8,
                )
        row += 1
    ax[0].set_title("Sum-Square Across Channels")
    ax[-1].set_xlabel("Time (seconds)")

    plt.show()


def plot_channel_stds(
    raw: mne.io.Raw,
    exclude_bad_segments: bool = True,
) -> None:
    """Plot distribution of standard deviations across channels.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object.
    exclude_bad_segments : bool
        Whether to exclude bad segments.
    """

    if exclude_bad_segments:
        reject_by_annotation = "omit"
    else:
        reject_by_annotation = None

    # --- NEW: get bad channel indices ---
    bad_inds = [raw.ch_names.index(ch) for ch in raw.info["bads"]]

    # Get all channels
    is_ctf = raw.info["dev_ctf_t"] is not None
    if is_ctf:
        channel_types = {
            "Axial Grads (chtype=mag)": mne.pick_types(
                raw.info, meg="mag", ref_meg=False, exclude=[]
            ),
            "Ref Axial Grad (chtype=ref_meg)": mne.pick_types(
                raw.info, meg="grad", exclude=[]
            ),
            "EEG": mne.pick_types(raw.info, eeg=True, exclude=[]),
            "CSD": mne.pick_types(raw.info, csd=True, exclude=[]),
        }
    else:
        channel_types = {
            "Magnetometers": mne.pick_types(raw.info, meg="mag", exclude=[]),
            "Gradiometers": mne.pick_types(raw.info, meg="grad", exclude=[]),
            "EEG": mne.pick_types(raw.info, eeg=True, exclude=[]),
            "CSD": mne.pick_types(raw.info, csd=True, exclude=[]),
        }

    # Get data
    x = raw.get_data(reject_by_annotation=reject_by_annotation)

    # Number of subplots
    ncols = sum(len(c) > 0 for c in channel_types.values())

    if ncols == 0:
        return

    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(9, 3.5))
    if ncols == 1:
        ax = [ax]

    row = 0
    for name, chan_inds in channel_types.items():
        if len(chan_inds) == 0:
            continue

        # Compute stds
        stds = x[chan_inds, :].std(axis=1)

        # Plot histogram
        ax[row].hist(stds, bins=24, histtype="step")
        bad_in_type = np.intersect1d(chan_inds, bad_inds)
        if len(bad_in_type) > 0:
            bad_stds = x[bad_in_type, :].std(axis=1)
            for s in bad_stds:
                ax[row].axvline(s, linestyle="--", color="tab:red")

        ax[row].set_xlabel("Standard Deviation")
        ax[row].set_ylabel("Channel Count")
        ax[row].set_title(name)
        row += 1

    plt.tight_layout()
    plt.show()
