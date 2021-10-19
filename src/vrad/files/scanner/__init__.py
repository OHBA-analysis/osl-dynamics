from pathlib import Path

import numpy as np

path = Path(__file__).parent
directory = str(path)

layouts = path / "layouts"

ctf275_channel_names = np.load(path / "ctf275_channel_names.npy")
neuromag306_channel_names = np.load(path / "neuromag306_channel_names.npy")
