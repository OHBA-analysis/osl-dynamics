"""Functions used to manage files kept within osl-dynamics.

"""

from os import path


def check_exists(filename, directory):
    """Looks for a file in the current working directory and in osl-dynamics.

    Parameters
    ----------
    filename : str
        Name of file to look for or a path to a file.
    directory : str
        Path to directory to look in.

    Returns
    -------
    filename : str
        Full path to the file if found.

    Raises
    ------
    FileNotFoundError
        If the file could not be found.
    """
    if not path.exists(filename):
        if path.exists(f"{directory}/{filename}"):
            filename = f"{directory}/{filename}"
        else:
            raise FileNotFoundError(filename)
    return filename
