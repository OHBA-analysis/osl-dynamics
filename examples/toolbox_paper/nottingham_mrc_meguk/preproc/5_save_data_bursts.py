"""Nottingham MRC MEGUK: Save Training Data for Burst Detection Analysis.

"""

import mne
from glob import glob
from osl_dynamics.data import Data

def get_downsampled_data(filename):
    raw = mne.io.read_raw_fif(filename)
    raw = raw.resample(100)  # Downsample to 100 Hz
    x = raw.get_data(picks="misc", reject_by_annotation="omit")
    x = x[[8]]  # Only keep one parcel in the left motor cortex
    return x.T

# Source reconstructed data files
src_dir = "data/src"
files = sorted(glob(f"{src_dir}/*/sflip_parc-raw.fif"))

# Get data
time_series = [get_downsampled_data(file) for file in files]

# Save as normal numpy files
data = Data(time_series)
data.save("training_data/bursts")
data.delete_dir()
