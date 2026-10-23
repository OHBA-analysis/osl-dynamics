"""Fit a standard EEG montage to the MNI152 standard brain.

If we don't have digitised electrode positions or a structural MRI, we can
use a standard montage and a standard brain to source reconstruct EEG data.
However, the electrodes of a standard montage don't sit on the scalp of the
standard brain.

The coregistration aligns the fiducials then refines the fit with ICP, using
the electrodes as headshape points. If the montage doesn't fit the scalp, ICP
slides the electrodes away from the fiducials to fit them to the scalp.

Here, we scale the electrode positions (rather than the MRI) by hand until
the electrodes fit the scalp without ICP moving them away from the fiducials.
For the colin27_1005 montage, scaling the vertical axis by 1.2 works well.
"""

from pathlib import Path

import numpy as np
import mne

from osl_dynamics import files
from osl_dynamics.meeg import preproc, rhino, Session

# ----------------------------------------------------------------------------
outdir = "standard_montage_coreg"
montage_name = "colin27_1005"
scale = (1, 1, 1.2)  # (x, y, z) scale factors for the electrode positions
# ----------------------------------------------------------------------------

# Channels of a 64-channel EEG cap (10-10 system)
cap_ch_names = [
    "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


def coregister(id, montage):
    """Coregister a montage to the MNI152 standard brain."""

    # Dummy Raw object containing the EEG positions. Note, we order the
    # channels as they are in the montage so the headshape points (which
    # are in the order of the montage) match ch_names
    ch_names = [ch for ch in montage.ch_names if ch in cap_ch_names]
    info = mne.create_info(ch_names, sfreq=250, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros([len(ch_names), 1]), info)
    raw.set_montage(montage)

    preproc_file = Path(f"{outdir}/{id}_preproc-raw.fif")
    preproc_file.parent.mkdir(parents=True, exist_ok=True)
    raw.save(preproc_file, overwrite=True)

    # Coregistration
    session = Session(
        outdir=outdir,
        id=id,
        preproc_file=preproc_file,
        surfaces_dir=files.mni152_surfaces.directory,
    )
    rhino.extract_fiducials_and_headshape_from_fif(
        session, include_eeg_as_headshape=True
    )
    rhino.coregister_head_and_mri(session, use_nose=False, show=True)

    # How far did ICP move the montage fiducials from the MRI fiducials?
    print()
    for fid in ["nasion", "lpa", "rpa"]:
        head = np.loadtxt(getattr(session.coreg, f"head_{fid}_file"))
        mri = np.loadtxt(getattr(session.coreg, f"mri_{fid}_file"))
        print(f"{fid} mismatch: {np.linalg.norm(head - mri):.1f} mm")

    # Distance from each electrode to the scalp (positive is outside)
    dist = rhino.coreg_error(session)
    print(f"Mean electrode to scalp distance: {np.mean(dist):+.1f} mm")
    print("Furthest electrodes from the scalp:")
    for i in np.argsort(-np.abs(dist))[:5]:
        print(f"    {ch_names[i]}: {dist[i]:+.1f} mm")


#%% Standard montage

montage = mne.channels.make_standard_montage(montage_name)
coregister("unscaled", montage)

#%% Scaled montage

montage = preproc.scale_eeg_positions(montage, scale)
coregister("scaled", montage)
