

def starcompose(*transforms):
    '''
    left compose functions together and expand tuples to args

    Use starcompose.debug for verbose output when debugging
    '''

    # TODO: consider doing starcompose with inner function calls rather than
    # a loop
    def _compose(*x):
        for t in transforms:
            if type(x) is tuple:
                x = t(*x)
            else:
                x = t(x)
        return x
    return _compose


def starcompose_debug(*transforms):
    '''
    verbose starcompose for debugging
    '''
    print('starcompose debug')
    def _compose(*x):
        for index, t in enumerate(transforms):
            print(f'{index}:, fn={t}, x={x}')
            if type(x) is tuple:
                x = t(*x)
            else:
                x = t(x)
        return x
    return _compose

starcompose.debug = starcompose_debug


def test_starcompose():
    from functools import partial

    test = starcompose(lambda x, y: x + y)
    if test(3, 5) != 8:
        raise Exception('Two args inputs failed')

    test = starcompose(lambda x: sum(x))
    if test((3, 5)) != 8:
        raise Exception('Tuple input failed')

    test = starcompose(
        lambda x: (x, x),
        lambda x, y: x + y,
        lambda x: x * 2,
    )
    if test(10) != 40:
        raise Exception('Expanded tuple for inner function failed')
