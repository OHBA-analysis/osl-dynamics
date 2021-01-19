# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from pkg_resources import DistributionNotFound, get_distribution

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "VRAD"
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

logging.basicConfig(level=logging.WARNING, format=logging.BASIC_FORMAT)
_logger = logging.getLogger("VRAD")

example_files = str(Path(__file__).parent.parent.parent / "examples" / "files")
