from functools import wraps


def star(fn):
    '''Wrap function to expand input to arguments'''
    @wraps(fn)
    def wrapper(args):
        return fn(*args)
    return wrapper
