"""Downloads and caches the data files used by osl-dynamics.

The parcellations, masks, surfaces, scanner layouts and scenes are not shipped
with the package. They live in `osl-files
<https://github.com/OHBA-analysis/osl-files>`_ and are downloaded the first
time they are needed, then cached locally.

The cache lives in the operating system's cache directory, which is
:code:`~/Library/Caches/osl-files` on macOS and :code:`~/.cache/osl-files`
on Linux. Set the :code:`OSL_DATA` environment variable to cache them
somewhere else, for example on a cluster where your home directory has a quota.
"""

import os
from pathlib import Path

import pooch

BASE_URL = "https://raw.githubusercontent.com/OHBA-analysis/osl-files/main/"
"""Where the data files are downloaded from."""

_REGISTRY_FILE = Path(__file__).parent / "registry.txt"

_POOCH = None


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


def _pooch():
    """The pooch instance, created on first use."""
    global _POOCH
    if _POOCH is None:
        _POOCH = pooch.create(
            path=cache_directory(),
            base_url=BASE_URL,
            registry=None,
        )
        _POOCH.load_registry(_REGISTRY_FILE)
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
    pup = _pooch()
    if path not in pup.registry:
        raise FileNotFoundError(
            f"'{path}' is not a data file provided by osl-dynamics."
        )
    try:
        return Path(pup.fetch(path))
    except ValueError as error:
        if "hash" not in str(error).lower():
            raise
        raise DownloadError(
            f"'{path}' has changed in osl-files since this version of "
            "osl-dynamics was released, so it no longer matches the checksum "
            "recorded in files/registry.txt. Upgrade osl-dynamics, which will "
            "carry the new checksum."
        ) from error
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
    paths = sorted(_pooch().registry)
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
