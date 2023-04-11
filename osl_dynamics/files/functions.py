"""Functions used to manage files kept within osl-dynamics.

"""

from os import path


def check_exists(filename, directory):
    """Looks for a file in the current working directory and in a osl-dynamics directory."""
    if not path.exists(filename):
        if path.exists(f"{directory}/{filename}"):
            filename = f"{directory}/{filename}"
        else:
            raise FileNotFoundError(filename)
    return filename
