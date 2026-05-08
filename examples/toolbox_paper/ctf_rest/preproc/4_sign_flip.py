"""Nottingham MEGUK: Sign flipping.

Aligns the sign of each parcel time course across sessions using the
covariance-based search in :code:`Data.align_channel_signs`, then writes
the sign-flipped data back to a fif file alongside the original.
"""

from glob import glob

import mne

from osl_dynamics.data import Data
from osl_dynamics.meeg import parcellation

# ----------------------------------------------------------------------------
outdir = "data/preproc"
n_jobs = 8
# ----------------------------------------------------------------------------

files = sorted(glob(f"{outdir}/*/lcmv-parc-raw.fif"))

data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=n_jobs)
data.align_channel_signs(
    n_init=3,
    n_iter=3000,
    max_flips=20,
    n_embeddings=15,
    standardize=True,
)

# Save each aligned session back as a fif. parcellation.save_as_fif copies
# annotations and the stim channel from the reference raw, and re-inserts
# zeros at bad-segment positions so the time axis matches the original.
for src_file, aligned in zip(files, data.arrays):
    print(f"Sign-flipping {src_file}")
    raw = mne.io.read_raw_fif(src_file, preload=True)
    out_file = src_file.replace("lcmv-parc-raw.fif", "sflip-lcmv-parc-raw.fif")
    parcellation.save_as_fif(
        aligned.T,
        raw,
        filename=out_file,
        extra_chans="stim",
    )
