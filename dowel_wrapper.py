import sys

assert 'dowel' not in sys.modules, 'dowel must be imported after dowel_wrapper.'

# https://stackoverflow.com/a/6985648/2182622
import dowel
dowel_eval = dowel
del sys.modules['dowel']

import dowel
dowel_plot = dowel
del sys.modules['dowel']

import dowel
all_dowels = [dowel, dowel_eval, dowel_plot]
assert len(set(id(d) for d in all_dowels)) == len(all_dowels)

import global_context
def get_dowel(phase=None):
    if (phase or global_context.get_context().get('phase')).lower() == 'plot':
        return dowel_plot
    if (phase or global_context.get_context().get('phase')).lower() == 'eval':
        return dowel_eval
    return dowel
def get_logger(phase=None):
    return get_dowel(phase).logger
def get_tabular(phase=None):
    return get_dowel(phase).tabular
