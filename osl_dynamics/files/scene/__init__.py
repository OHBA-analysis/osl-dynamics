"""HCP Workbench scene files.

These files are downloaded on first use, see
:py:mod:`osl_dynamics.files._fetch`.
"""

from osl_dynamics.files import _fetch

_SUBDIRECTORY = "scene"


def file(name: str):
    """Path to one scene file, downloading it if needed."""
    return _fetch.fetch_file(f"{_SUBDIRECTORY}/{name}")


def __getattr__(name):
    if name == "mode_scene":
        return file("mode_scene.scene")
    if name == "path":
        return _fetch.fetch_directory(_SUBDIRECTORY)
    if name == "directory":
        return str(_fetch.fetch_directory(_SUBDIRECTORY))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
