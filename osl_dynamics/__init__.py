from pkg_resources import DistributionNotFound, get_distribution
import logging

from .pipeline import run_pipeline

try:
    __version__ = get_distribution("osl-dynamics").version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

# Configure logging for package
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("osl-dynamics")
logger.debug("Version %s", __version__)

del logger, logging
