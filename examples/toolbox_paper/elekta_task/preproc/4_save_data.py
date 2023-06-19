"""Wakeman-Henson: Save Training Data.

The data saved by this script can be downloaded from https://osf.io/by2tc.
"""

from glob import glob

from osl_dynamics.data import Data

# Save data as standard numpy files
src_dir = "data/src"
files = sorted(glob(f"{src_dir}/*/sflip_parc-raw.fif"))
data = Data(files, picks="misc", reject_by_annotation="omit")
data.save("data/training_data")
data.delete_dir()
