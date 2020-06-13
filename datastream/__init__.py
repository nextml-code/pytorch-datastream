from datastream.star import star
from datastream.starcompose import starcompose
from datastream.repeat_map_chain import repeat_map_chain

from datastream.dataset import Dataset
from datastream.datastream import Datastream

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution('ml-workflow').version
except DistributionNotFound:
    pass
