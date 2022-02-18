"""Functions used to manage files kept in DyNeMo source code.

"""

from os import path


def check_exists(filename, dynemo_directory):
    """Looks for a file in the current working directory and in a OHBA-Models directory."""
    if not path.exists(filename):
        if path.exists(f"{dynemo_directory}/{filename}"):
            filename = f"{dynemo_directory}/{filename}"
        else:
            raise FileNotFoundError(filename)
    return filename
