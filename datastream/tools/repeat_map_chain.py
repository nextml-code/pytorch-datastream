from itertools import repeat, chain
from functools import partial

from datastream.tools import starcompose


def mk_repeat_map_chain(fn):
    return starcompose(
        repeat,
        partial(map, fn),
        chain.from_iterable,
    )


def repeat_map_chain(fn, iterable):
    return mk_repeat_map_chain(fn)(iterable)
