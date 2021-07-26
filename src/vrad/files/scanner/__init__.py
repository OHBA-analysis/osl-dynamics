from pathlib import Path

path = Path(__file__).parent
directory = str(path)

ctf275_channel_names = str(path / "ctf275_channel_names.npy")
neuromag306_channel_names = str(path / "neuromag306_channel_names.npy")
