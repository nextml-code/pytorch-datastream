from datastream.dataset import Dataset
from datastream.datastream import Datastream

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution('pytorch-datastream').version
except DistributionNotFound:
    pass
