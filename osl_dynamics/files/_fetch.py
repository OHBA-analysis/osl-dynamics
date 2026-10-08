"""Downloads and caches the data files used by osl-dynamics.

The parcellations, masks, surfaces, scanner layouts and scenes are not shipped
with the package. They live in `osl-files
<https://github.com/OHBA-analysis/osl-files>`_ and are downloaded the first
time they are needed, then cached locally.

The cache lives in the operating system's cache directory, which is
:code:`~/Library/Caches/osl-files` on macOS and :code:`~/.cache/osl-files`
on Linux. Set the :code:`OSL_DATA` environment variable to cache them
somewhere else, for example on a cluster where your home directory has a quota.

The files are always the ones currently in osl-files: the list of files and
their checksums is downloaded from there too, and a cached file is downloaded
again when its checksum changes. A cache directory that cannot be written to
is used as it is, so a read-only copy shared by a group stays at the version
its owner last downloaded with :code:`osl-dynamics-download-data`.
"""

import os
import time
import urllib.request
from pathlib import Path

import pooch

BASE_URL = "https://raw.githubusercontent.com/OHBA-analysis/osl-files/main/"
"""Where the data files are downloaded from."""

REGISTRY_URL = BASE_URL + "registry.txt"
"""Lists every data file and its checksum. Kept beside the data, not here, so
that adding or changing a file in osl-files reaches users without osl-dynamics
needing to be released."""

REGISTRY_MAX_AGE = 3600
"""Seconds before the cached copy of the registry is refreshed."""

_POOCH = None
_REGISTRY_MTIME = None
_REGISTRY_CHECKED = None


class DownloadError(RuntimeError):
    """Raised when a data file could not be downloaded."""


def cache_directory() -> Path:
    """Directory the data files are cached in.

    Returns
    -------
    directory : pathlib.Path
        :code:`OSL_DATA` if it is set, otherwise the operating
        system's cache directory.
    """
    directory = os.environ.get("OSL_DATA")
    if directory:
        return Path(directory).expanduser()
    return Path(pooch.os_cache("osl-files"))


def _registry_file(refresh: bool = False) -> Path:
    """Local copy of the registry, downloaded again once it is stale.

    Falls back to the cached copy if it cannot be downloaded, so a machine
    with a warm cache keeps working offline, and if the cache directory
    cannot be written to.

    Parameters
    ----------
    refresh : bool, optional
        Download the registry even if the cached copy is not stale yet.
    """
    global _REGISTRY_CHECKED
    path = cache_directory() / "registry.txt"
    now = time.time()
    if path.exists() and not refresh:
        if now - path.stat().st_mtime < REGISTRY_MAX_AGE:
            return path
        # The cached copy is stale but could not be replaced the last time
        # this was tried, so don't try again on every call
        if _REGISTRY_CHECKED is not None and now - _REGISTRY_CHECKED < REGISTRY_MAX_AGE:
            return path
        # A read-only cache is a copy kept up to date by whoever owns it
        if not os.access(path.parent, os.W_OK):
            return path

    _REGISTRY_CHECKED = now
    try:
        with urllib.request.urlopen(REGISTRY_URL, timeout=30) as response:
            registry = response.read()
    except Exception as error:
        if path.exists():
            return path
        raise DownloadError(
            f"Could not download the list of data files from {REGISTRY_URL}.\n"
            "osl-dynamics downloads its parcellations, masks and surfaces the "
            "first time they are used, so this step needs network access. On "
            "a machine without it, run 'osl-dynamics-download-data' somewhere "
            "that does and copy the cache across, or point OSL_DATA at a "
            f"directory that already has the files. The cache is currently "
            f"{cache_directory()}."
        ) from error

    # Written beside the registry and moved into place, so that another
    # process never reads a partly written registry
    temporary = path.with_name(f"registry.txt.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(registry)
        os.replace(temporary, path)
    except OSError as error:
        if path.exists():
            return path
        raise DownloadError(
            f"Could not write to the cache directory {cache_directory()}. Set "
            "OSL_DATA to a directory you can write to, or to one that already "
            "has the files."
        ) from error
    return path


def _pooch(refresh: bool = False):
    """The pooch instance, rebuilt whenever the registry changes.

    Parameters
    ----------
    refresh : bool, optional
        Download the registry even if the cached copy is not stale yet.
    """
    global _POOCH, _REGISTRY_MTIME
    path = _registry_file(refresh)
    mtime = path.stat().st_mtime
    if _POOCH is None or mtime != _REGISTRY_MTIME:
        _POOCH = pooch.create(
            path=cache_directory(),
            base_url=BASE_URL,
            registry=None,
        )
        _POOCH.load_registry(path)
        _REGISTRY_MTIME = mtime
    return _POOCH


def fetch_file(path: str) -> Path:
    """Local path to one data file, downloading it if it isn't cached.

    Parameters
    ----------
    path : str
        Path of the file within osl-files, e.g.
        :code:`"parcellation/atlas-AAL_nparc-78_space-MNI_res-8x8x8.nii.gz"`.

    Returns
    -------
    path : pathlib.Path
        Full path to the cached file.

    Raises
    ------
    FileNotFoundError
        If osl-dynamics does not provide a file with that name.
    DownloadError
        If the file is not cached and could not be downloaded.
    """
    path = str(path)
    for refresh in (False, True):
        pup = _pooch(refresh)
        if path not in pup.registry:
            raise FileNotFoundError(
                f"'{path}' is not a data file provided by osl-dynamics."
            )
        try:
            return Path(pup.fetch(path))
        except ValueError as error:
            if "hash" not in str(error).lower():
                raise
            # The download does not match the registry. The file has probably
            # changed in osl-files since the registry was cached, so get the
            # registry again and have one more go
            mismatch = error
        except Exception as error:
            raise DownloadError(
                f"Could not download '{path}' from {BASE_URL}.\n"
                "osl-dynamics downloads its parcellations, masks and surfaces the "
                "first time they are used, so this step needs network access. On "
                "a machine without it, run 'osl-dynamics-download-data' somewhere "
                "that does and copy the cache across, or point "
                "OSL_DATA at a directory that already has the files. The "
                f"cache is currently {cache_directory()}."
            ) from error
    raise DownloadError(
        f"The download of '{path}' does not match its checksum in the "
        "osl-files registry. Either it was cut short, or the file has just "
        "been changed in osl-files and the registry has not caught up yet. "
        "Try again in a few minutes."
    ) from mismatch


def fetch_directory(subdirectory: str) -> Path:
    """Local path to a directory of data files, downloading any not cached.

    Parameters
    ----------
    subdirectory : str
        Directory within osl-files, e.g. :code:`"parcellation"`.

    Returns
    -------
    path : pathlib.Path
        Full path to the cached directory.
    """
    prefix = f"{subdirectory}/"
    paths = [path for path in _pooch().registry if path.startswith(prefix)]
    if not paths:
        raise FileNotFoundError(
            f"'{subdirectory}' is not a data directory provided by osl-dynamics."
        )
    for path in paths:
        fetch_file(path)
    return cache_directory() / subdirectory


def fetch_all() -> Path:
    """Downloads every data file, so they are available offline later.

    Returns
    -------
    path : pathlib.Path
        The cache directory the files were downloaded to.
    """
    # Start from the current registry, not a cached copy up to an hour old
    paths = sorted(_pooch(refresh=True).registry)
    for i, path in enumerate(paths, start=1):
        print(f"[{i}/{len(paths)}] {path}")
        fetch_file(path)
    return cache_directory()


def download_data_cli() -> None:
    """Command line interface for downloading every data file up front.

    Useful on a machine that will later be offline, or to populate a directory
    shared by several users.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Download the parcellations, masks, surfaces, scanner layouts and "
            "scenes that osl-dynamics uses, so they are available offline."
        ),
    )
    parser.parse_args()

    print(f"Downloading the osl-dynamics data files to {cache_directory()}")
    path = fetch_all()
    print(f"Done. Set OSL_DATA to {path} to use this cache elsewhere.")
