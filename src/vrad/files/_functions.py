"""Functions used to manage files kept in VRAD source code.

"""

from os import path

def check_exists(filename, vrad_directory):
    """Looks for a file in the current working directory and in a VRAD directory."""
    if not path.exists(filename):
        if path.exists(f"{vrad_directory}/{filename}"):
            filename = f"{vrad_directory}/{filename}"
        else:
            raise FileNotFoundError(filename)
    return filename
