
def star(fn):
    '''Wrap function to expand input to arguments'''
    return lambda args: fn(*args)
