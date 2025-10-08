import logging
from importlib.metadata import PackageNotFoundError, version

from osl_dynamics.config_api.pipeline import run_pipeline


# Setup the version
try:
    __version__ = version("osl-dynamics")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d:%(funcName)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("osl-dynamics")
logger.debug("Version %s", __version__)
del logger, logging
