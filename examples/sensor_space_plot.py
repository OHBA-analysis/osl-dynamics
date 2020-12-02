from pathlib import Path

import numpy as np
from vrad.utils import plotting

# Get example files
files_dir = Path(__file__).parent / "files"
data_file = str(files_dir / "ctf275_data.npy")
channel_file = str(files_dir / "ctf275_channel_names.npy")

# Load example files
ctf275_data = np.load(data_file)
ctf275_channel_names = np.load(channel_file)

# Produce the figure using the "CTF275_helmet" layout.
# Layouts follow the naming of those in FieldTrip.
plotting.topoplot(
    layout="CTF275_helmet",
    data=np.load("ctf275_data.npy"),
    channel_names=np.load("ctf275_channel_names.npy"),
    plot_boxes=False,
    show_deleted_sensors=True,
    show_names=False,
    title="Example Plot",
    colorbar=True,
)
