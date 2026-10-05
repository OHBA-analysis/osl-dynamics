"""Volumetric parcellations.

These files are downloaded on first use, see
:py:mod:`osl_dynamics.files._fetch`. For more information see
:ref:`parcellations`.
"""

from osl_dynamics.files import _fetch

_SUBDIRECTORY = "parcellation"


def file(name: str) -> str:
    """Path to one parcellation file, downloading it if needed."""
    return str(_fetch.fetch_file(f"{_SUBDIRECTORY}/{name}"))


def __getattr__(name):
    if name == "path":
        return _fetch.fetch_directory(_SUBDIRECTORY)
    if name == "directory":
        return str(_fetch.fetch_directory(_SUBDIRECTORY))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
