import logging
import os
from importlib.metadata import PackageNotFoundError, version

# Hide the INFO messages of TensorFlow's C++ code (it must be set before
# TensorFlow is imported)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

from osl_dynamics.utils.logger import (
    _setup_logger,
    _setup_tensorflow_import_message,
    set_verbosity,
)

_setup_logger()
_setup_tensorflow_import_message()

from osl_dynamics.config_api.pipeline import run_pipeline
from osl_dynamics import meeg

# Setup the version
try:
    __version__ = version("osl-dynamics")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

logging.getLogger("osl-dynamics").debug("Version %s", __version__)
del logging, os, _setup_logger, _setup_tensorflow_import_message
