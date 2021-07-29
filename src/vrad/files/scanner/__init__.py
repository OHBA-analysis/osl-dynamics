import numpy as np
from pathlib import Path

path = Path(__file__).parent
directory = str(path)

ctf275_channel_names = np.load(path / "ctf275_channel_names.npy")
neuromag306_channel_names = np.load(path / "neuromag306_channel_names.npy")
