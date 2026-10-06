"""MEG scanner layouts and channel names.

These files are downloaded on first use, see
:py:mod:`osl_dynamics.files._fetch`.
"""

import numpy as np

from osl_dynamics.files import _fetch

_SUBDIRECTORY = "scanner"

_CHANNEL_NAMES = {
    "ctf275_channel_names": "ctf275_channel_names.npy",
    "neuromag306_channel_names": "neuromag306_channel_names.npy",
}

_CACHE = {}


def file(name: str):
    """Path to one scanner file, downloading it if needed."""
    return _fetch.fetch_file(f"{_SUBDIRECTORY}/{name}")


def __getattr__(name):
    if name in _CHANNEL_NAMES:
        if name not in _CACHE:
            _CACHE[name] = np.load(file(_CHANNEL_NAMES[name]))
        return _CACHE[name]
    if name == "layouts":
        return _fetch.fetch_directory(f"{_SUBDIRECTORY}/layouts")
    if name == "path":
        return _fetch.fetch_directory(_SUBDIRECTORY)
    if name == "directory":
        return str(_fetch.fetch_directory(_SUBDIRECTORY))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
