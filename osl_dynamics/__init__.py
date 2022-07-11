from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("osl-dynamics").version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
